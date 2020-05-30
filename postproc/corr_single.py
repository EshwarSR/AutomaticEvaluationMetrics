from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import os

# dire = '../results/SentBERT/CNN_DailyMail_SentBERT.tsv'
dire = 'asap_aes_sentbert.tsv'
odf = pd.read_csv(dire, delimiter='\t')

score = odf['score'].tolist()
similarity = odf['similarity'].tolist()
out = []
scorr = spearmanr(similarity, score)
pcorr = pearsonr(similarity, score)
kentau = kendalltau(similarity, score)
out.append(['SentBERT', scorr.correlation, scorr.pvalue, pcorr[0], pcorr[1], kentau[0], kentau[1]])

fdf = pd.DataFrame(out, columns=['Score', 'Spearman Rank Correlation', 'Spearman p-value',\
																 'Pearson Rank Correlation', 'Pearson p-value',  \
																 'Kendall Rank Correlation', 'Kendall p-value'])

fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_all.tsv", sep="\t", index=False, header=True)