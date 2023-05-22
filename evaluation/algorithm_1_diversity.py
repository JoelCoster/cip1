from utils import has_question, get_cosine_sim


def diversity(current_turn, previous_turns, mat_tf, repetition_threshold):
    # Create a list of repeated turns and questions
    repeated_turns = []
    repeated_questions = []

    # Set r to 0 (line 2 in pseudocode)
    r = 0

    # Loop through turns said by speaker (line 3 in pseudocode)
    for index, previous_turn in previous_turns.iterrows():
        # Check if two utterances are similar (line 4 and 5 in pseudocode)
        if get_cosine_sim(current_turn.name, previous_turn.name,
                          mat_tf) >= repetition_threshold:
            repeated_turns.append(previous_turn)

    # Check if there are repeated utterances (line 6 in pseudocode;
    # pseudocode and GitHub differ, taken GitHub)
    if len(repeated_turns) > 0:
        flag_question, flag_rep = False, False
        for repeated_turn in repeated_turns:
            # Check if the current turn and repeated turn is a question (line 7
            # and 8 in pseudocode)
            if has_question(repeated_turn['utterance']) and has_question(current_turn['utterance']):
                repeated_questions.append(repeated_turn)
                flag_question = True

            # Check previous turn is not a repetitive question (line
            # 9, 10 and 11 in pseudocode)
            elif len(repeated_questions) > 0 and current_turn.name >= 1:
                previous_turn = previous_turns.iloc[current_turn.name - 1]
                for repeated_question in repeated_questions:
                    if get_cosine_sim(previous_turn.name,
                                      repeated_question.name,
                                      mat_tf) <= repetition_threshold:
                        flag_rep = True
        if flag_question or flag_rep:
            r += 1
    return r
