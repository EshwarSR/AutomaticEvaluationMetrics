import pandas as pd
import numpy as np
from pandas import DataFrame


from processing import data
from bleu import BLEU


df = pd.read_csv('../data/train.tsv', sep='\t')
n_gram = 4

reference_corpus, candidate_corpus = data(df)

print(BLEU(reference_corpus[1], candidate_corpus[1][50], 4))

bleu_scores = []
bleu = []

for i in range(len(candidate_corpus[2])):
    bleu.append(BLEU(reference_corpus[i], candidate_corpus[i][j], n_gram))
    bleu_scores.append(list(bleu))

print(bleu_scores)
