import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import numpy as np

dire = '../results/cnn_final/cnn_all.tsv'
odf = pd.read_csv(dire, sep='\t')
metrics = odf.columns[1:-1].tolist()

for an in metrics:
	print(an[:-4] + ' starting')
	ou = 'plots_cnn_3d/'
	if (os.path.exists(ou) == False):
		os.mkdir(ou)
	exit(7)
##################################################
	# another = []
	# for i in [-1, 1]:
	# 	another.append( * 2 - 1)
	maxxie = max([max(x) for x in another])
	minnie = min([min(x) for x in another])
	rn = maxxie - minnie
	bins_needed = 100
	wd = rn / bins_needed
	# print(bins_needed)
	be = np.array([minnie + i * wd for i in range(bins_needed + 1)])
	# print(len(be))
	fig = plt.figure(figsize=(16, 9), dpi=100)
	ax = fig.add_subplot(111, projection='3d')
	for c, z in zip(['r', 'g', 'b', 'y'], [0, 1, 2, 3]):
		interval = 0.2
		hist, _ = np.histogram(another[z], bins=be)
		xs = (be[:-1] + be[1:])/2
		plt.bar(xs, hist, width=wd, zs=z, zdir='y', color=c, ec=c, alpha=0.8, label='cands-humanscore:{0:1d}'.format(z))

	plt.title('Histogram of scores correspoding to human scores', \
					fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.legend(loc='best')

	ax.set_xlabel('similarity score (scaled between 0 and {0:1d})'.format(max_score - 1))
	ax.set_ylabel('human score')
	ax.set_zlabel('frequency')
	# plt.show()
	plt.savefig(ou + an[:-4] + '.png', dpi=300)
	print(an[:-4] + ' done')
