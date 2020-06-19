import sys
import os
import torch
import math
import numpy as np
from data import QEDataset, collate_fn
from model import QE
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--src', default="en")
parser.add_argument('--tgt', default="de")
parser.add_argument('--model', default="bert")
parser.add_argument('--output_dir', required=True)
parser.add_argument('--use_word_probs', nargs="?", const=True, default=False)
parser.add_argument('--num_features', default=None, type=int)
parser.add_argument('--encode_separately', nargs="?", const=True, default=False)
parser.add_argument('--use_secondary_loss', nargs="?", const=True, default=False)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--run_id', type=str, default=1)
args = parser.parse_args()
print(args)

output_dir = os.path.join(args.output_dir, 'run_{}'.format(args.run_id))
try:
    assert not os.path.exists(output_dir)
except AssertionError:
    print('Fatal! Output directory exists:'.format(output_dir))
    raise
os.mkdir(output_dir)


src_lcode = args.src
tgt_lcode = args.tgt

epochs = args.epochs
#model specific configuration
if args.model.lower() == "xlm":
    model_name = "xlm-mlm-100-1280"
    model_dim = 1280
    learning_rate = 1e-6
    batch_size = 6 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
elif args.model.lower() == "xlm_roberta":
    model_name = "xlm-roberta-base"
    model_dim = 768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
elif args.model.lower() == "xlm_roberta_large":
    model_name = "xlm-roberta-large"
    model_dim = 1024
    learning_rate = 1e-6
    batch_size = 8 * args.num_gpus
    eval_interval = 100
    if src_lcode == "all":
        eval_interval=600
    accum_grad = 1
else:
    model_name = "bert-base-multilingual-cased"
    model_dim = 768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
    if args.use_word_probs:
        batch_size = 12

#load model and optimizer
gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

model = QE(transformer,
        model_dim,
        use_word_probs = args.use_word_probs,
        num_features=args.num_features,
        encode_separately=args.encode_separately,
        use_secondary_loss=args.use_secondary_loss)

n_param = 0
for p in model.parameters():
    if p.requires_grad:
        n_param += p.numel()

print(n_param)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(gpu)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

filedir = "data/%s-%s"%(src_lcode, tgt_lcode) if src_lcode != "all" else "data/*"
train_file = glob("%s/train.*.tsv" % filedir)
train_mt_file = glob("%s/word-probas/mt.train.*" % filedir)
train_wp_file = glob("%s/word-probas/word_probas.train.*" % filedir)
train_features_file = glob("%s/features.train.tsv" % filedir) if args.num_features else None
train_dataset = QEDataset(train_file, train_mt_file, train_wp_file, features_path=train_features_file)
assert os.path.exists(train_file)
assert os.path.exists(train_mt_file)
assert os.path.exists(train_wp_file)
if args.num_features:
    assert os.path.exists(train_features_file)

if src_lcode == "all":
    lcodes = [("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en")]
else:
    lcodes = [(src_lcode, tgt_lcode)]

dev_datasets, test_datasets = [], []
for src_lcode, tgt_lcode in lcodes:
    filedir = "data/%s-%s"%(src_lcode, tgt_lcode)
    dev_file = glob("%s/traindev*.tsv" % filedir)
    dev_mt_file = glob("%s/word-probas/mt.traindev*" % filedir)
    dev_wp_file = glob("%s/word-probas/word_probas.traindev*" % filedir)
    dev_features_file = glob("%s/*features.traindev.tsv" % filedir) if args.num_features else None
    dev_datasets.append(((src_lcode, tgt_lcode), QEDataset(dev_file, dev_mt_file, dev_wp_file, features_path=dev_features_file)))

    test_file = glob("%s/dev*.tsv" % filedir)
    test_mt_file = glob("%s/word-probas/mt.dev*" % filedir)
    test_wp_file = glob("%s/word-probas/word_probas.dev*" % filedir)
    test_features_file = glob("%s/*features.dev.tsv" % filedir) if args.num_features else None
    test_datasets.append(((src_lcode, tgt_lcode), QEDataset(test_file, test_mt_file, test_wp_file, features_path=test_features_file)))

log_file = os.path.join(output_dir, "log")
flog = open(log_file, "w")

def eval(dataset, get_metrics=False):
    model.eval()
    predicted_scores, actual_scores = [], []
    for batch, wps, z_scores, _, feats in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_word_probs=args.use_word_probs,
            encode_separately=args.encode_separately,
            num_features=args.num_features), shuffle=False)):
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps
        feats = feats.to(gpu) if feats is not None else feats

        #force nan to be 0, this deals with bad inputs from si-en dataset
        z_score_outputs, _ = model(batch, wps, feats=feats)
        z_score_outputs[torch.isnan(z_score_outputs)] = 0
        predicted_scores += z_score_outputs.flatten().tolist()

        actual_scores += z_scores
    if get_metrics:
        predicted_scores = np.array(predicted_scores)
        actual_scores = np.array(actual_scores)
        pearson = pearsonr(predicted_scores, actual_scores)[0]
        mse = np.square(np.subtract(predicted_scores, actual_scores)).mean()
    else:
        pearson, mse = None, None
    model.train()
    return predicted_scores, pearson, mse

global_steps = 0
best_eval = 0
early_stop = 0
for epoch in range(epochs):
    print("Epoch ", epoch)
    total_loss = 0
    total_batches = 0
    for batch, wps, z_scores, da_scores, feats in tqdm(DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_word_probs=args.use_word_probs,
            num_features=args.num_features,
            encode_separately=args.encode_separately), shuffle=True)):
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps
        feats = feats.to(gpu) if feats is not None else feats
        z_scores = torch.tensor(z_scores).to(gpu)
        z_score_outputs, da_score_outputs  = model(batch, wps, feats=feats)

        #drop batch with nan
        if torch.isnan(z_score_outputs).any():
            del batch, wps, z_scores, da_scores, z_score_outputs, da_score_outputs
            continue
        elif da_score_outputs is not None and torch.isnan(da_score_outputs).any():
            del batch, wps, z_scores, da_scores, z_score_outputs, da_score_outputs
            continue


        loss = loss_fn(z_score_outputs.squeeze(), z_scores)
        cur_batch_size = z_score_outputs.size(0)

        #if we are using a secondary loss
        if args.use_secondary_loss:
            da_scores = torch.tensor(da_scores).to(gpu)
            loss += loss_fn(da_score_outputs.squeeze(), da_scores)

        total_loss += loss.item() * cur_batch_size
        total_batches += cur_batch_size
        loss.backward()
        del batch, wps, z_scores, da_scores, z_score_outputs, da_score_outputs, loss

        if global_steps % accum_grad == 0:
            optimizer.step()
            model.zero_grad()

        global_steps += 1

        with torch.no_grad():
            if global_steps % eval_interval == 0:
                dev_results = []
                total_pearson, total = 0, 0
                print("\nCalculating results on dev set(s)...")
                for lcodes, dev_dataset in dev_datasets:
                    predicted_scores, pearson, mse =  eval(dev_dataset, get_metrics=True)
                    dev_results.append((lcodes, predicted_scores, pearson, mse))
                    total_pearson += pearson
                    total += 1

                avg_pearson = total_pearson/total
                if avg_pearson > best_eval:
                    best_eval = avg_pearson
                    print()
                    for lcodes, predicted_scores, _, _ in dev_results:
                        best_dev_file = os.path.join(output_dir, "%s%s.dev.best.scores" % lcodes)
                        print("Saving best dev results to: %s" % best_dev_file)
                        with open(best_dev_file, "w") as fout:
                            for score in predicted_scores:
                                print(score, file=fout)

                    test_results = []
                    print("\nCalculating results on test set(s)...")
                    for lcodes, test_dataset in test_datasets:
                        predicted_scores, _, _ = eval(test_dataset)
                        test_results.append((lcodes, predicted_scores))

                    for lcodes, predicted_scores in test_results:
                        best_test_file = os.path.join(output_dir, "%s%s.test.best.scores" % lcodes)
                        print("Saving best test results to: %s" % best_test_file)
                        with open(best_test_file, "w") as fout:
                            for score in predicted_scores:
                                print(score, file=fout)
                    early_stop = 0
                else:
                    early_stop += 1

                log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, global_steps, total_loss/total_batches)
                #reset total loss 
                total_loss, total_batches = 0, 0
                for lcodes, _, pearson, mse in dev_results:
                    log +="%s-%s Dev loss: %.4f r:%.4f\n" % (lcodes[0], lcodes[1], mse, pearson)
                log +="Current avg r:%.4f Best avg r: %.4f" % (avg_pearson, best_eval)
                print(log)
                print(log, file=flog)
        if early_stop > 25:
            break
    if early_stop > 25:
        break
