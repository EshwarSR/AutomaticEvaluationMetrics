from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd

if len(sys.argv) > 3:
    results_file = sys.argv[1]
    field = sys.argv[2]
    agg = sys.argv[3]
    results = pd.read_csv(results_file, sep="\t")
    groups = results.groupby("candidate_id")
    similarity = []
    score = []
    for candidate_id, group in groups:
        score.append(group["score"].mean())
        if agg == "mean" or agg == "avg":
            similarity.append(group[field].mean())
        elif agg == "max":
            similarity.append(group[field].max())

else:
    results_file = sys.argv[1]
    field = sys.argv[2]
    results = pd.read_csv(results_file, sep="\t")
    similarity = results[field].tolist()
    score = results["score"].tolist()


scorr = spearmanr(similarity, score)
# print("File:", results_file)
print("Spearman Rank Correlation:{0:6.3f}".format(scorr.correlation))
if (scorr.pvalue >= 0.001):
	print("P-value:{0:6.3f}".format(scorr.pvalue))
else:
	print("P-value:{0:10.3e}".format(scorr.pvalue))
pcorr = pearsonr(similarity, score)
print("Pearson Correlation Coefficient:{0:6.3f}".format(pcorr[0]))
if (pcorr[1] >= 0.001):
	print("P-value:{0:6.3f}".format(pcorr[1]))
else:
	print("P-value:{0:10.3e}".format(pcorr[1]))
kentau = kendalltau(similarity, score)
print("Kendall Rank Correlation :{0:6.3f}".format(kentau[0]))
if (kentau[1] >= 0.001):
	print("P-value:{0:6.3f}".format(kentau[1]))
else:
	print("P-value:{0:10.3e}".format(kentau[1]))
