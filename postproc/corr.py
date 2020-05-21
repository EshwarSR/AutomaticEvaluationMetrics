from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import os

dire = '../results/asap_aes_results/'
fn = os.listdir(dire)

out = []
for an in fn:
	inpath = dire + an
	odf = pd.read_csv(inpath, delimiter='\t')
	similarity = odf['similarity score'].tolist()
	score = odf['score'].tolist()

	scorr = spearmanr(similarity, score)
	pcorr = pearsonr(similarity, score)
	kentau = kendalltau(similarity, score)
	out.append([an[:-4],scorr.correlation, scorr.pvalue, pcorr[0], pcorr[1], kentau[0], kentau[1]])
	print(inpath + ' done')

fdf = pd.DataFrame(out, columns=['Score', 'Spearman Rank Correlation', 'Spearman p-value',\
																 'Pearson Rank Correlation', 'Pearson p-value',  \
																 'Kendall Rank Correlation', 'Kendall p-value'])

fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_all.tsv", sep="\t", index=False, header=True)