import string

import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer

from utilities import load_csv


def get_cosine_sim(i, j, mat_tf):
    return mat_tf[i, j]


def diversity_detector(current_turn, previous_turns, mat_tf,
                       repetition_threshold):
    repetition_penalty = 0
    repeated_turns = []

    for index, previous_turn in previous_turns.iterrows():
        if get_cosine_sim(current_turn.name, previous_turn.name,
                          mat_tf) >= repetition_threshold:
            repeated_turns.append(previous_turn)



def get_tfidf_cosine(document):
    vect = TfidfVectorizer(tokenizer=normalize, min_df=1)
    tfidf = vect.fit_transform(document)
    pairwise_similarity = (tfidf * tfidf.T).toarray()
    # print(pairwise_similarity.shape)
    return pairwise_similarity


def get_tokens(text):
    lower = text.lower()
    remove_punctuation_map = dict(
        (ord(char), None) for char in string.punctuation)
    no_punctuation = lower.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def normalize(text):
    return stem_tokens(get_tokens(text), PorterStemmer())


def get_tfidf_sim(df):
    document = list(df['utterance'])
    return get_tfidf_cosine(document)


def main():
    df_file = '../src/logs/05-15-2023 13_42_36.csv'
    df = load_csv(df_file)

    mat_tf = get_tfidf_sim(df)

    repetition_threshold = 0.95
    for i in range(3, df.shape[0]):
        current_turn = df.iloc[i]
        previous_turns = df.iloc[:i]
        diversity_detector(current_turn, previous_turns, mat_tf,
                           repetition_threshold)


if __name__ == "__main__":
    main()
