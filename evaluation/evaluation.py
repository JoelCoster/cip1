import argparse

import pandas as pd

from algorithm_1_diversity import diversity_percentage
from algorithm_2_consistency import consistency_percentage


def parse_args() -> argparse.ArgumentParser():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Evaluate CIP logfile.')
    parser.add_argument('-l', '--logfile', dest='logfile', help='Path to logfile', required=True, )
    return parser.parse_args()


def divider():
    print('-' * 50)


def main():
    args = parse_args()

    df_logs = pd.read_csv(args.logfile, index_col=0)

    speaker_1, speaker_2 = 'JAM', 'SNO'

    ### Algorithm 1
    # Speaker 1
    rep_questions_speaker_1 = diversity_percentage(df_logs, speaker_1)

    # Speaker 2
    rep_questions_speaker_2 = diversity_percentage(df_logs, speaker_2)

    divider()
    ### Algorithm 1
    # Speaker 1
    consistency_percentage(df_logs, speaker_1, rep_questions_speaker_2)

    # Speaker 2
    consistency_percentage(df_logs, speaker_2, rep_questions_speaker_1)

    # Calculate relevance score (Algorithm 3)


if __name__ == "__main__":
    main()
