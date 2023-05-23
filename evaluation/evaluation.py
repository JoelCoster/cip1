import pandas as pd
import argparse

from algorithm_1_diversity import diversity_percentage


def parse_args() -> argparse.ArgumentParser():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Evaluate CIP logfile.')
    parser.add_argument('-l', '--logfile', dest='logfile', help='Path to logfile', required=True, )
    return parser.parse_args()


def main():
    args = parse_args()

    df_logs = pd.read_csv(args.logfile, index_col=0)

    # Algorithm 1
    diversity_percentage(df_logs)

    # Calculate consistency score (Algorithm 2)

    # Calculate relevance score (Algorithm 3)


if __name__ == "__main__":
    main()
