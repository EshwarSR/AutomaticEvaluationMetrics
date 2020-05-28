from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import os

dire = '../results/CNN_DailyMail_results/cnn_all.tsv'
odf = pd.read_csv(dire, delimiter='\t')

metrics = odf.columns
metrics = metrics[1:-1]
score = odf[odf.columns[-1]].tolist()

out = []
for one in metrics:
	# print(one + ' started')
	similarity = odf[one].tolist()
	score = score[:len(similarity)]
	scorr = spearmanr(similarity, score)
	pcorr = pearsonr(similarity, score)
	kentau = kendalltau(similarity, score)
	out.append([one, scorr.correlation, scorr.pvalue, pcorr[0], pcorr[1], kentau[0], kentau[1]])
	# print(one + ' done')

fdf = pd.DataFrame(out, columns=['Score', 'Spearman Rank Correlation', 'Spearman p-value',\
																 'Pearson Rank Correlation', 'Pearson p-value',  \
																 'Kendall Rank Correlation', 'Kendall p-value'])

fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_all.tsv", sep="\t", index=False, header=True)