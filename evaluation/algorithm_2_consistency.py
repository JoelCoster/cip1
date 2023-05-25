import pandas as pd

from constants import CONSISTENCY_THRESHOLD
from utils import get_cosine_sim
from utils import get_tfidf_sim


def consistency_percentage(df: pd.DataFrame, speaker, repeated_questions):
    total_penalty = 0
    mat_tf = get_tfidf_sim(df)
    utterance_count = df[df['speaker'].str.contains(speaker)].shape[0]

    for i in range(df.shape[0]):
        current_turn = df.iloc[i]

        # Check if utterance is said by current speaker
        if speaker in current_turn['speaker']:
            total_penalty += consistency_penalty(mat_tf, current_turn, repeated_questions)

    print(f'Consistency score (\"{speaker}\"): {total_penalty}')
    print(f'Consistency percentage (\"{speaker}\"): {total_penalty / utterance_count:.2%}')
    print()


def consistency_penalty(mat_tf, current_turn, repeated_questions):
    penalty = 0

    # Check if utterance preceding the current turn is a repeated question
    if current_turn.name - 1 not in repeated_questions:
        return penalty

    # Check if the repeating question was not said by the other person
    elif repeated_questions[current_turn.name - 1].speaker == current_turn.speaker:
        return penalty

    # Find IDs for old and new answer to the question
    old_a_id = repeated_questions[current_turn.name - 1].name + 1
    new_a_id = current_turn.name

    # See if answers are similar. Difference will be penalized
    if get_cosine_sim(old_a_id, new_a_id, mat_tf) < CONSISTENCY_THRESHOLD:
        penalty += 1

    return penalty
