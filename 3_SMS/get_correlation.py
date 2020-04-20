from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
results_file = sys.argv[1]

results = pd.read_csv(results_file, sep="\t")

similarity = results["similarity score"].tolist()
score = results["score"].tolist()

scorr = spearmanr(similarity, score)
# print("File:", results_file)
print("Spearman Rank Correlation:{0:6.3f}".format(scorr.correlation))
print("P-value:{0:5.3f}".format(scorr.pvalue))
pcorr = pearsonr(similarity, score)
print("Pearson Correlation Coefficient:{0:6.3f}".format(pcorr[0]))
print("P-value:{0:6.3f}".format(pcorr[1]))
kentau = kendalltau(similarity, score)
print("Kendall Rank Correlation :{0:6.3f}".format(kentau[0]))
print("P-value:{0:6.3f}".format(kentau[1]))
