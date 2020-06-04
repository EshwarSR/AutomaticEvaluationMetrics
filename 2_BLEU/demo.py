import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import spearmanr
from texttable import Texttable

from CNN_processing import get_ref_and_cand
from bleu import BLEU


df = pd.read_csv('../data/demo.tsv', sep='\t')

n_gram = 4

candidate_corpus, reference_corpus, human_scores = get_ref_and_cand(df)
# print(reference_corpus[0:10])
# print(candidate_corpus[0:10])

bleu_scores = []

for i in range(len(candidate_corpus)):
    bleu = BLEU(reference_corpus[i], candidate_corpus[i], n_gram)
    # print(bleu)
    bleu_scores.append(bleu)
# print(max(bleu_scores))
# print(bleu_scores[0:500])

scorr = spearmanr(human_scores, bleu_scores)
corr = "{0:6.3f}".format(scorr.correlation)
if (scorr.pvalue >= 0.001):
        pval = "{0:6.3f}".format(scorr.pvalue)
else:
    pval = "{0:10.3e}".format(scorr.pvalue)


correlations = [["Method", "Correlation", "P value"]]
correlations.append(['BLEU', corr, pval])

print("\nCorrelations from various EMD based metrics\n")
t = Texttable()
t.add_rows(correlations)
print(t.draw())
# filename = "../results/CNN_DailyMail_results/BLEU_scores_CNN.txt"
# with open(filename, 'w') as f:
#     f.write("candidate_id\tsimilarity\tscore\n")
#     for i in range(len(candidate_corpus)):
#         f.write("{0}\t{1}\t{2}\n".format(i+1, bleu_scores[i], human_scores[i]))