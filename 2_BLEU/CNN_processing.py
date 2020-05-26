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

def get_ref_and_cand(df):

    """
    """
    ref_list = df['reference'].tolist()
    cand_list = df['candidate'].tolist()
    human_score_list = df['score'].tolist()
    length = len(ref_list)

    ref_corpus = []
    cand_corpus = []

    for i in range(length):
        ref = ref_list[i].split()
        cand = cand_list[i].split()
        ref = remove_punctuation(to_lowercase(ref))
        cand = remove_punctuation(to_lowercase(cand))
        ref_corpus.append(ref)
        cand_corpus.append(cand)
    return cand_corpus, ref_corpus, human_score_list
