import os
import pandas as pd
import matplotlib.pyplot as plt

dire = '../results/all_threes/'
fn = os.listdir(dire)
for an in fn:
	ex = an[:-4]
	ou = 'plots/' + ex
	os.mkdir(ou)
	ou += '/'
	inpath = dire + an
	odf = pd.read_csv(inpath, delimiter='\t')
	# print(len(odf))
	# print(odf.columns)
	plt.figure(figsize=(16, 9), dpi=100)
	plt.grid(True)
	plt.hist(odf['similarity score'], bins=100)
	plt.title('Histogram of similarity scores', \
          fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.xlabel('similarity score (scaled between 0 and 2)')
	plt.ylabel('frequency')
	# plt.show()
	plt.savefig(ou + 'score_range.png', dpi=300)

	plt.figure(figsize=(16, 9), dpi=100)
	plt.grid(True)
	plt.plot(odf['similarity score'] * 2, odf['score'], '.g')
	plt.title('Scatter plots of similarity scores vs human scores', \
          fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.xlabel("human score")
	plt.ylabel("similarity score")
	# plt.show()
	plt.savefig(ou + 'scatter.png')
	
	plt.figure(figsize=(16, 9), dpi=100)
	max_score = 2
	another = []
	for i in range(max_score):
	  another.append(odf.loc[(odf['score'] == i)]['similarity score'] * 2)
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
	plt.savefig(ou + 'each_score.png', dpi=300)
	print(inpath)
	