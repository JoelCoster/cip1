import string

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

from utilities import load_csv


def get_cosine_sim(i, j, mat_tf):
    # Calculate cosine similarity between two vectors in the matrix
    return mat_tf[i, j]


def has_question(sentence):
    # Check if a sentence contains a question mark
    return "?" in sentence


def diversity_detector(i, df, speaker, mat_tf, repetition_threshold):
    # Create a list of repeated turns and questions
    repeated_turns = []
    repeated_questions = []

    # Set r to 0 (line 2 in pseudocode)
    r = 0

    # Determine current turn and previous turns
    current_turn = df.iloc[i]
    previous_turns = df.iloc[:i]

    # Loop through turns said by speaker (line 3 in pseudocode)
    if speaker in current_turn['speaker']:
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
            if has_question(repeated_turn['utterance']) and has_question(
                    current_turn['utterance']):
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


def get_tfidf_cosine(document):
    # Calculate the TF-IDF cosine similarity matrix for a document
    vect = TfidfVectorizer(tokenizer=normalize, min_df=1)
    tfidf = vect.fit_transform(document)
    pairwise_similarity = (tfidf * tfidf.T).toarray()
    return pairwise_similarity


def get_tokens(text):
    # Tokenize text into words
    lower = text.lower()
    remove_punctuation_map = dict(
        (ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    # Apply stemming to a list of tokens
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def normalize(text):
    # Normalize text by tokenizing and stemming
    return stem_tokens(get_tokens(text), PorterStemmer())


def get_tfidf_sim(df):
    # Get the TF-IDF cosine similarity matrix for a DataFrame
    document = list(df['utterance'])
    return get_tfidf_cosine(document)


def main():
    # Load the DataFrame from a CSV file
    df_file = '../src/logs/05-15-2023 13_42_36.csv'
    df = load_csv(df_file)

    # Calculate the TF-IDF cosine similarity matrix
    mat_tf = get_tfidf_sim(df)

    repetition_threshold = 0.8
    speaker = 'SNO'

    diversity_score = 0
    for i in range(df.shape[0]):
        # Calculate diversity score for each turn
        diversity_score += diversity_detector(i, df, speaker, mat_tf,
                                              repetition_threshold)
    print('Diversity score: {0}'.format(diversity_score))


if __name__ == "__main__":
    main()
