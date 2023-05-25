import pandas as pd

from constants import DIVERSITY_THRESHOLD
from utils import get_tfidf_sim
from utils import has_question, get_cosine_sim


def diversity_percentage(df: pd.DataFrame, speaker):
    total_penalty = 0
    mat_tf = get_tfidf_sim(df)
    utterance_count = df[df['speaker'].str.contains(speaker)].shape[0]

    total_repeated_questions = dict()
    for i in range(df.shape[0]):
        current_turn = df.iloc[i]
        previous_turns = df.iloc[:i]

        # Check if utterance is said by current speaker
        if speaker in current_turn['speaker']:
            penalty, repeated_questions = diversity_penalty(mat_tf, current_turn, previous_turns)
            for repeated_question in repeated_questions:
                if repeated_question.name not in [total_repeated_question.name for total_repeated_question in total_repeated_questions]:
                    total_repeated_questions[i] = repeated_question
            total_penalty += penalty

    print(f'Diversity score (\"{speaker}\"): {total_penalty}')
    print(f'Diversity percentage (\"{speaker}\"): {total_penalty / utterance_count:.2%}')
    print()

    return total_repeated_questions


def diversity_penalty(mat_tf, current_turn, previous_turns):
    turn_penalty = 0

    repeated_turns = []
    repeated_questions = []

    # Line 3 in pseudocode
    for index, previous_turn in previous_turns.iterrows():

        # Check if two utterances are similar (line 4 and 5 in pseudocode)
        if get_cosine_sim(current_turn.name, previous_turn.name, mat_tf) >= DIVERSITY_THRESHOLD:
            repeated_turns.append(previous_turn)

    # Check if there are repeated utterances (line 6 in pseudocode)
    if len(repeated_turns) == 0:
        return turn_penalty, repeated_questions

    flag_question, flag_rep = False, False

    for repeated_turn in repeated_turns:
        # Check if the current turn and repeated turn is a question (line 7 and 8 in pseudocode)
        if has_question(repeated_turn['utterance']) and has_question(current_turn['utterance']):
            repeated_questions.append(repeated_turn)
            flag_question = True

        # Check previous turn is not a repetitive question (line 9, 10 and 11 in pseudocode)
        elif current_turn.name > 0:
            previous_turn = previous_turns.iloc[current_turn.name - 1]
            if len(repeated_questions):
                for repeated_question in repeated_questions:
                    if get_cosine_sim(previous_turn.name, repeated_question.name, mat_tf) <= DIVERSITY_THRESHOLD:
                        flag_rep = True
            else:
                flag_rep = True

    if flag_question or flag_rep:
        turn_penalty += 1

    return turn_penalty, repeated_questions
