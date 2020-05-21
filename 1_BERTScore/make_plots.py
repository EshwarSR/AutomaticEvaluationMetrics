import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

odf = pd.read_csv('BERTScore_noidf_no_rescale_3_aes.tsv', delimiter='\t')

max_score = 4
another = []
for i in range(max_score):
	another.append(odf.loc[(odf['score'] == i)]['R score'] * (max_score - 1))
maxxie = max([max(x) for x in another])
minnie = min([min(x) for x in another])

rn = maxxie - minnie

wd = 0.003
bins_needed = math.ceil(rn / wd)
print(bins_needed)

be = np.array([minnie + i * wd for i in range(bins_needed + 1)])
print(len(be))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
i = 0
for c, z in zip(['r', 'g', 'b', 'y'], [0, 1, 2, 3]):
	interval = 0.2
	hist, _ = np.histogram(another[i], bins=be)
	xs = (be[:-1] + be[1:])/2
	# plt.bar(xs, hist, width=wd, alpha=0.3)
	plt.bar(xs, hist, width=wd, zs=z, zdir='y', color=c, ec=c, alpha=0.8, label='cands-humanscore:{0:1d}'.format(i))
	i += 1

plt.title('Histogram of scores correspoding to human scores', \
				fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
plt.legend(loc='best')

ax.set_xlabel('similarity score (scaled between 0 and {0:1d})'.format(max_score - 1))
ax.set_ylabel('Range of scores')
ax.set_zlabel('frequency')
plt.show()

exit(3)


















import pandas as pd
import matplotlib.pyplot as plt

odf = pd.read_csv('outs.tsv', delimiter='\t')

max_score = 4
plt.figure(figsize=(16, 9), dpi=100)
another = []
for i in range(max_score):
	another.append(odf.loc[(odf['score'] == i)]['R score'] * (max_score - 1))
	# print(len(another[i]))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i in range(max_score):
	plt.hist(another[i], bins=50, color=colors[(i + 1) % max_score], alpha=0.3, \
					 label='cands-humanscore:{0:1d}'.format(i))
	plt.title('Histogram of scores correspoding to human scores', \
				fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.xlabel('similarity score (scaled between 0 and {0:1d})'.format(max_score - 1))
	plt.ylabel('frequency')
	plt.legend(loc='best')
plt.savefig('each_score.png', dpi=300)

exit(2)


# print(len(odf))
# print(odf.columns)
plt.figure(figsize=(16, 9), dpi=100)
plt.grid(True)
plt.hist(odf['R score'], bins=100)
plt.title('Histogram of similarity scores', \
			fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
plt.xlabel('similarity score (scaled between 0 and 1)')
plt.ylabel('frequency')
# plt.show()
plt.savefig('score_range.png', dpi=300)

max_score = 4
plt.figure(figsize=(16, 9), dpi=100)
plt.grid(True)
plt.plot(odf['R score'] * (max_score - 1), odf['score'], '.g')
plt.title('Scatter plots of similarity scores vs human scores', \
			fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
plt.xlabel("human score")
plt.ylabel("similarity score")
# plt.show()
plt.savefig('scatter.png')

plt.figure(figsize=(16, 9), dpi=100)
another = []
for i in range(max_score):
	another.append(odf.loc[(odf['score'] == i)]['R score'] * (max_score - 1))
	# print(len(another[i]))
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for i in range(max_score):
	plt.hist(another[i], bins=50, color=colors[(i + 1) % max_score], alpha=1, \
					 label='cands-humanscore:{0:1d}'.format(i))
	plt.title('Histogram of scores correspoding to human scores', \
				fontdict={'fontname': 'DejaVu Sans', 'fontsize': 14})
	plt.xlabel('similarity score (scaled between 0 and {0})'.format(max_score - 1))
	plt.ylabel('frequency')
	plt.legend(loc='best')
plt.savefig('each_score.png', dpi=300)
