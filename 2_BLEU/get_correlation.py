from scipy.stats import spearmanr
import sys
import pandas as pd
results_file = sys.argv[1]

results = pd.read_csv(results_file, sep="\t")

similarity = results["similarity"].tolist()
score = results["score"].tolist()
new_score = [x/2 for x in score]

corr = spearmanr(similarity, new_score)
print("File:", results_file, "Correlation:",
      corr.correlation, "P-value:", corr.pvalue)
