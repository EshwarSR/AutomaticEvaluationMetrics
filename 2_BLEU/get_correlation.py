from scipy.stats import spearmanr
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


results_file = sys.argv[1]

results = pd.read_csv(results_file, sep="\t")

df = results[['similarity', 'score']]

similarity = results["similarity"].tolist()
score = results["score"].tolist()
# new_score = [x/2 for x in score]

corr = spearmanr(similarity, score)
print("File:", results_file, "Correlation:",
      corr.correlation, "P-value:", corr.pvalue)

# similarity = [2*x for x in similarity]

# corr = df.corr()
# fig = plt.figure()
# plt.matshow(corr, fignum=fig.number)
# plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
# plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

# plt.hist(score, color = 'r')
# plt.hist(similarity, color = 'b')
# plt.show()

# sns.pairplot(df, hue='coolwarm')
sns.pairplot(df, diag_kind='kde', plot_kws=dict(s=75, linewidth=1),
                 diag_kws=dict(shade=True))
plt.show()