from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd


results_file = 'data_as_per_author_elmo_s+wms.tsv'
results = pd.read_csv(results_file, sep="\t")
score = results['score'].tolist()

cory = []
similarity = results['similarity'].tolist()
scorr = spearmanr(similarity, score)
pcorr = pearsonr(similarity, score)
kentau = kendalltau(similarity, score)
cory.append([ 'elmo_s+wms', scorr.correlation, scorr.pvalue, pcorr[0], pcorr[1], kentau[0], kentau[1] ])

fdf = pd.DataFrame(cory, columns=['Score', 'Spearman Rank Correlation', 'Spearman p-value',\
																 'Pearson Rank Correlation', 'Pearson p-value',  \
																 'Kendall Rank Correlation', 'Kendall p-value'])
fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_elmo_s+wms_sms.tsv", sep="\t", index=False, header=True)