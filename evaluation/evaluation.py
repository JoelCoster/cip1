import pandas as pd

from algorithm_1_diversity import diversity
from utils import get_tfidf_sim


def load_csv(df_file):
    # Load a CSV transcript as a Pandas Dataframe
    return pd.read_csv(df_file, index_col=0)


def main():
    # Load the DataFrame from a CSV file
    df_file = '../src/logs/05-15-2023 13_42_36.csv'
    df = load_csv(df_file)

    # Set speaker
    speaker = 'SNO'

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
        if speaker in current_turn['speaker']:
            diversity_score += diversity(current_turn, previous_turns, mat_tf, rep_threshold)
    print('Diversity score: {0}'.format(diversity_score))

    # Calculate consistency score (Algorithm 2)

    # Calculate relevance score (Algorithm 3)


if __name__ == "__main__":
    main()
