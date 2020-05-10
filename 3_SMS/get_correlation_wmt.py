from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd

results_file = "../results/WMT18_Results/final/scores_all.tsv"
data = pd.read_csv(results_file, sep="\t")

metrics = list(data.columns)
metrics.remove("HUMAN")
metrics.remove("SYSTEM")
metrics.remove("LP")

correlations = ["Spearman", "Pearson", "Kendall"]

final_correlations = []
final_p_values = []

groups = data.groupby("LP")
for metric in metrics:
    print("Running for Metric", metric)
    for correlation in correlations:
        row = {
            "metric": metric,
            "correlation": correlation,
        }
        p_row = {
            "metric": metric,
            "correlation": correlation,
        }
        for language, group in groups:
            if correlation == "Spearman":
                resp = spearmanr(group[metric], group["HUMAN"])
                corr = resp.correlation
                p_val = resp.pvalue
            elif correlation == "Pearson":
                corr, p_val = pearsonr(group[metric], group["HUMAN"])
            elif correlation == "Kendall":
                corr, p_val = kendalltau(group[metric], group["HUMAN"])
            row[language] = corr
            p_row[language] = p_val
        final_correlations.append(row)
        final_p_values.append(p_row)

final_df = pd.DataFrame(final_correlations)
final_p_df = pd.DataFrame(final_p_values)
print("Saving to file")
final_df.to_csv("../results/WMT18_Results/final/all_correlations.tsv",
                index=False, sep="\t")
final_p_df.to_csv(
    "../results/WMT18_Results/final/all_pvalues.tsv", index=False, sep="\t")
