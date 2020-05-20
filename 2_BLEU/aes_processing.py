import nltk
import pandas as pd
from pandas import DataFrame
import re
import numpy as np


# Data Processing
def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words
    Args: A list of words to be processed
    Returns: A list of processed words"""

    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):

    """Remove punctuation from list of tokenized words
    Args: A list of words to be processed
    Returns: A list of processed words"""

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]','', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def cand_data(df):

    """
    """
    df = df[df['essay_set'] == 3]
    cand_id = df['essay_id'].tolist()

    cand_list = df['essay'].tolist()
    human_score = df['domain1_score'].tolist()

    cand_corpus = []

    for i in range(len(cand_list)):
        lst = cand_list[i].split()
        lst = remove_punctuation(to_lowercase(lst))
        cand_corpus.append(lst)
    return cand_id, cand_corpus, human_score

def ref_data(df):

    """
    """
    ref_list = df['Reference'].tolist()
    ref_corpus = []
    for i in range(len(ref_list)):
        lst = ref_list[i].split()
        lst = remove_punctuation(to_lowercase(lst))
        ref_corpus.append(lst)
    return ref_corpus
