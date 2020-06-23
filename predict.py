import argparse
import torch

from transformers import AutoTokenizer, AutoModel

from model import QE
from data import QEDataset
from evaluate import evaluate


def predict(args):
    # load model and optimizer
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    transformer = AutoModel.from_pretrained(args.model_name)
    model = QE(transformer, args.model_dim, num_features=args.num_features)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    dataset = QEDataset(
        args.test_file, args.test_mt_file, args.test_wp_file, features_path=args.test_features_file)
    predicted_scores, pearson, mse = evaluate(
        dataset, model, tokenizer, args.batch_size, gpu, num_features=args.num_features, encode_separately=args.encode_separately,
        use_word_probs=args.use_word_probs, get_metrics=True)
    print(predicted_scores)
    print(pearson)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_dim', required=True, type=int)
    parser.add_argument('--num_features', required=False, default=None, type=int)
    parser.add_argument('--test_file', required=True)
    parser.add_argument('--test_mt_file', required=True)
    parser.add_argument('--test_wp_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--test_features_file', required=False, default=None)
    parser.add_argument('--batch_size', required=False, default=16)
    parser.add_argument('--use_word_probs', action='store_true', required=False, default=False)
    parser.add_argument('--encode_separately', action='store_true', required=False, default=False)
    args = parser.parse_args()
    print(args)
    predict(args)


if __name__ == '__main__':
    main()
