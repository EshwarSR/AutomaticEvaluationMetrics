import pandas as pd
import matplotlib.pyplot as plt

odf = pd.read_csv('outs.tsv', delimiter='\t')

# print(len(odf))
# print(odf.columns)
plt.figure(figsize=(16, 9), dpi=100)
plt.grid(True)
plt.hist(odf['ROUGE-L Score'], bins=100)
plt.title('Histogram of similarity scores', \
      fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
plt.xlabel('similarity score (scaled between 0 and 2)')
plt.ylabel('frequency')
# plt.show()
plt.savefig('score_range.png', dpi=300)

plt.figure(figsize=(16, 9), dpi=100)
plt.grid(True)
plt.plot(odf['ROUGE-L Score'] * 2, odf['score'], '.g')
plt.title('Scatter plots of similarity scores vs human scores', \
      fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
plt.xlabel("human score")
plt.ylabel("similarity score")
# plt.show()
plt.savefig('scatter.png')

plt.figure(figsize=(16, 9), dpi=100)
max_score = 2
another = []
for i in range(max_score):
  another.append(odf.loc[(odf['score'] == i)]['ROUGE-L Score'] * 2)
  # print(len(another[i]))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i in range(max_score):
	plt.hist(another[i], bins=50, color=colors[(i + 1) % 2], alpha=0.5, \
	         label='cands-humanscore:{0:1d}'.format(i))
	plt.title('Histogram of scores correspoding to human scores', \
	      fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.xlabel('similarity score (scaled between 0 and 2)')
	plt.ylabel('frequency')
	plt.legend(loc='best')
plt.savefig('each_score.png', dpi=300)
