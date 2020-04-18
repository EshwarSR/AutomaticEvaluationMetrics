import pandas as pd
import numpy as np
from pandas import DataFrame


from processing import data
from bleu import BLEU


df = pd.read_csv('../data/train.tsv', sep='\t')
n_gram = 4

reference_corpus, candidate_corpus, reference_id, candidate_id, candidate_scores, max_scores = data(df)

print(reference_id[0], candidate_id[0], candidate_scores[0], max_scores)
# print(BLEU(reference_corpus[1], candidate_corpus[1][50], 4))

bleu_scores = []
bleu = []

for i in range(100):
    bleu.append(BLEU(reference_corpus[2], candidate_corpus[2][i], n_gram))
    # print(bleu)
    # bleu_scores.append(list(bleu))

for i in range(len(bleu)):
    print(bleu[i])
