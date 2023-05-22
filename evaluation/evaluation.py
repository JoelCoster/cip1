import pandas as pd
import argparse

from algorithm_1_diversity import diversity
from utils import get_tfidf_sim


SPEAKER = 'SNO'


def parse_args() -> argparse.ArgumentParser():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(description='Evaluate CIP logfile.')
    parser.add_argument('-l', '--logfile', dest='logfile', help='Path to logfile', required=True, )
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.logfile, index_col=0)

    # Calculate the TF-IDF cosine similarity matrix
    mat_tf = get_tfidf_sim(df)

    # Calculate diversity score (Algorithm 1)
    rep_threshold = 0.8
    diversity_score = 0
    for i in range(df.shape[0]):
        # Determine current turn and previous turns
        current_turn = df.iloc[i]
        previous_turns = df.iloc[:i]

        # Check if utterance is said by current speaker
        if SPEAKER in current_turn['speaker']:
            diversity_score += diversity(current_turn, previous_turns, mat_tf, rep_threshold)
    print('Diversity score: {0}'.format(diversity_score))

    # Calculate consistency score (Algorithm 2)

    # Calculate relevance score (Algorithm 3)


if __name__ == "__main__":
    main()
