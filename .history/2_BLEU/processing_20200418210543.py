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

def data(df):

    """
    """
    essay_set_list = (df['EssaySet'].unique())
    max_score_list = []
    for i in essay_set_list:
        max_score_list.append(df[df['EssaySet']==i]['Score1'].max())
    #     print('Max score for essay {} is {}'.format(i, max_score_list[i-1]))


    reference = pd.DataFrame()
    candidates = pd.DataFrame()

    for i in essay_set_list:
        ref = df[(df['EssaySet']==i) & (df['Score1']==max_score_list[i-1])]
        ref_list = [reference, ref]
        reference = pd.concat(ref_list)
        cands = df[(df['EssaySet']==i) & (df['Score1']!=max_score_list[i-1])]
        cand_list = [candidates, cands]
        candidates = pd.concat(cand_list)


    total_ref = reference.count()[0]
    total_cand = candidates.count()[0]
    # print(total_ref, total_cand, total_ref+total_cand)

    essay_set_list_ref = (reference['EssaySet'].unique())
    essay_set_list_cand = (candidates['EssaySet'].unique())
    # print(essay_set_list_ref, essay_set_list_cand)


    ref = df.loc[(df['Score1']==1) & (df['EssaySet']!=3)]
    # ref.head(5)
    # ref.loc[0]['EssaySet']


    # Genearting the corpus

    reference_corpus = []
    candidate_corpus = []
    reference_id = []
    candidate_id = []
    candidate_scores = []

    for i in essay_set_list:
        ref = reference.loc[reference['EssaySet']==i]
        cand = candidates.loc[candidates['EssaySet']==i]
        
        count_ref = ref.count()[0]
        count_cand = cand.count()[0]
        
        ref_list = []
        cand_list = []
        cand_id = []
        ref_id = []
        score = []
        
        for j in range(count_ref):
            ref_list.append(list(ref.iloc[j]['EssayText'].split()))
            ref_id.append(ref.iloc[j]['Id'])
        ref_tuple = (i, ref_list)
        ref_id_tuple = (i, ref_id)
        reference_corpus.append(ref_tuple)
        reference_id.append(ref_id_tuple)
        
        for j in range(count_cand):
            cand_list.append(list(cand.iloc[j]['EssayText'].split()))
            cand_id.append(cand.iloc[j]['Id'])
            score.append(cand.iloc[j]['Score1'])
        cand_tuple = (i, cand_list)
        cand_id_tuple = (i, cand_id)
        score_tuple = (i, score)

        candidate_corpus.append(cand_tuple)
        candidate_id.append(cand_id_tuple)
        candidate_scores.append(score_tuple)

    reference_corpus = dict(reference_corpus)
    candidate_corpus = dict(candidate_corpus)
    reference_id = dict(reference_id)
    candidate_id = dict(candidate_id)
    candidate_scores = dict(candidate_scores)


    reference_corpus = list(reference_corpus.values())
    candidate_corpus = list(candidate_corpus.values())
    reference_id = list(reference_id.values())
    candidate_id = list(candidate_id.values())
    candidate_scores = list(candidate_scores.values())



    new_reference_corpus = []
    new_candidate_corpus = []

    for i in essay_set_list:
        ref_list = []
        cand_list = []
        for j in range(len(reference_corpus[i-1])):
            ref_list.append(to_lowercase(remove_punctuation(reference_corpus[i-1][j])))
        for j in range(len(candidate_corpus[i-1])):
            cand_list.append(to_lowercase(remove_punctuation(candidate_corpus[i-1][j])))
        ref_tuple = (i, ref_list)
        cand_tuple = (i, cand_list)
        new_reference_corpus.append(ref_tuple)
        new_candidate_corpus.append(cand_tuple)

    new_reference_corpus = dict(new_reference_corpus)
    new_candidate_corpus = dict(new_candidate_corpus)


    reference_corpus = list(new_reference_corpus.values())
    candidate_corpus = list(new_candidate_corpus.values())

    return reference_corpus, candidate_corpus