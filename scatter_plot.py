import argparse

import pandas as pd
import seaborn as sns

from matplotlib import pyplot
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--labels')  # MLQE format
    parser.add_argument('-p', '--predictions')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('--lang', default=None)
    args = parser.parse_args()

    test_df = pd.read_csv(args.labels, sep='\t', quoting=3)
    preds = [float(l.strip()) for l in open(args.predictions)]
    test_df['preds'] = preds
    print(pearsonr(test_df['preds'], test_df['z_mean'])[0])
    regplt = sns.jointplot(test_df['preds'], test_df['z_mean'], size=8)
    regplt.annotate(pearsonr)
    pyplot.title(args.lang)
    if args.output is not None:
        pyplot.savefig(args.output)
    else:
        pyplot.show()


if __name__ == '__main__':
    main()
