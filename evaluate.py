import numpy as np
import torch

from tqdm import tqdm
from functools import partial
from scipy.stats import pearsonr

from data import DataLoader, collate_fn


def evaluate(
        dataset, model, tokenizer, batch_size, gpu, get_metrics=False, use_word_probs=False,
        num_features=None, encode_separately=False):
    model.eval()
    predicted_scores, actual_scores = [], []
    for batch, wps, z_scores, _, feats in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            use_word_probs=use_word_probs,
            encode_separately=encode_separately,
            num_features=num_features), shuffle=False)):
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
