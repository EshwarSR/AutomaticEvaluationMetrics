import pandas as pd
import numpy as np
from pandas import DataFrame


from CNN_processing import get_ref_and_cand
from bleu import BLEU


df = pd.read_csv('../data/CNN_DailyMail.tsv', sep='\t')

n_gram = 4

candidate_corpus, reference_corpus, human_scores = get_ref_and_cand(df)
print(reference_corpus[0:10])
print(candidate_corpus[0:10])

bleu_scores = []

for i in range(len(candidate_corpus)):
    bleu = BLEU(reference_corpus[i], candidate_corpus[i], n_gram)
    # print(bleu)
    bleu_scores.append(bleu)
print(max(bleu_scores))
print(bleu_scores[0:500])

filename = "../results/CNN_DailyMail_results/BLEU_scores_CNN.txt"
with open(filename, 'w') as f:
    f.write("candidate_id\tsimilarity\tscore\n")
    for i in range(len(candidate_corpus)):
        f.write("{0}\t{1}\t{2}\n".format(i+1, bleu_scores[i], human_scores[i]))