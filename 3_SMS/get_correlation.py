from scipy.stats import spearmanr
import sys
import pandas as pd
results_file = sys.argv[1]

results = pd.read_csv(results_file, sep="\t")

similarity = results["similarity score"].tolist()
score = results["score"].tolist()

corr = spearmanr(similarity, score)
print("File:", results_file)
print("Correlation:", corr.correlation)
print("P-value:", corr.pvalue)
