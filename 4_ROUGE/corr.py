from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd

results_file = 'outs.tsv'
results = pd.read_csv(results_file, sep="\t")
score = results['score'].tolist()

cory = []
for one in results.columns[1:-1]:
	similarity = results[one].tolist()
	scorr = spearmanr(similarity, score)
	# print("File:", results_file)
	# print("Spearman Rank Correlation:{0:6.3f}".format(scorr.correlation))
	# if (scorr.pvalue >= 0.001):
	# 	print("P-value:{0:6.3f}".format(scorr.pvalue))
	# else:
	# 	print("P-value:{0:10.3e}".format(scorr.pvalue))
	pcorr = pearsonr(similarity, score)
	# print("Pearson Correlation Coefficient:{0:6.3f}".format(pcorr[0]))
	# if (pcorr[1] >= 0.001):
	# 	print("P-value:{0:6.3f}".format(pcorr[1]))
	# else:
	# 	print("P-value:{0:10.3e}".format(pcorr[1]))
	kentau = kendalltau(similarity, score)
	# print("Kendall Rank Correlation :{0:6.3f}".format(kentau[0]))
	# if (kentau[1] >= 0.001):
	# 	print("P-value:{0:6.3f}".format(kentau[1]))
	# else:
	# 	print("P-value:{0:10.3e}".format(kentau[1]))
	cory.append([ one, scorr.correlation, scorr.pvalue, pcorr[0], pcorr[1], kentau[0], kentau[1] ])

fdf = pd.DataFrame(cory, columns=['Score', 'Spearman Rank Correlation', 'Spearman p-value',\
																 'Pearson Rank Correlation', 'Pearson p-value',  \
																 'Kendall Rank Correlation', 'Kendall p-value'])
fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_rouge.tsv", sep="\t", index=False, header=True)