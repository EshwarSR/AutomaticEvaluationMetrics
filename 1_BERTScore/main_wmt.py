import pandas as pd
import numpy as np
import os
from bert_score import score

cs = '../data/WMT18Data/system-outputs/newstest2018/cs-en/newstest2018.CUNI-Transformer.5560.cs-en'
rs = '../data/WMT18Data/references/newstest2018-csen-ref.en'
human = '../data/WMT18Data/DA-syslevel.csv'

hdf = pd.read_csv(human, sep=' ')
hdf = hdf.loc[hdf['LP'] == 'cs-en']
hdf = hdf.loc[hdf['SYSTEM'] == 'CUNI-Transformer.5560']
hdf.reset_index(drop=True, inplace=True)

cands = []
fc = open(cs, "r", encoding='utf-8')
while True:
	line = fc.readline() 
	if not line:
		break
	cands.append(line)

refs = []
fr = open(rs, "r", encoding='utf-8')
while True:
	line = fr.readline() 
	if not line:
		break
	refs.append(line)

# broadcasting refs so that len(cands) == len(refs)
assert len(cands) == len(refs)

P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=True, device=None, batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=True)

sys_score = F1.mean()
print(sys_score)
print(hdf['HUMAN'].values[0])

# odf = pd.DataFrame()
# odf['BERT_Score'] = sys_score
# odf['Human_Score'] = hdf['HUMAN'].values[0]

# # print(can1.columns)
# odf = pd.DataFrame()
# odf['cand id'] = can1['Id']
# odf['similarity score'] = pd.DataFrame(F1)
# odf['score'] = can1['Score1']
# odf.reset_index(drop=True, inplace=True)
# # print(odf.columns)

# odf.to_csv("outs.tsv", sep="\t", index=False, header=True)

# odf.to_csv("outs.tsv", sep="\t", index=False, header=True)

# import matplotlib.pyplot as plt
# plt.grid(True)
# plt.hist(F1, bins=20)
# plt.show()
# plt.savefig('histogram_all.png')
# plt.xlabel("Score1")
# plt.ylabel("Pred_Score")
# plt.grid(True)
# plt.plot(odf['score'], odf['similarity score'] * 2, '.g')
# plt.show()
# plt.savefig('res/scatter.png')
# max_score = 2
# _, ax = plt.subplots(ncols=3, nrows=1, constrained_layout=True)
# for i in range(max_score + 1):
#   daf = odf.loc[(odf['score'] == i)]
#   # if (i != max_score):
#   #   ax[i].set_xlim(-0.03, 2.03)
#   ax[i].set_ylim(0, 250)
#   ax[i].hist(daf['similarity score'] * 2, bins = 20)
# plt.show()
# plt.savefig('histograms_ind.png')
# from bert_score import plot_example
# cand = cands[0]
# ref = refs[0][0]
# plot_example(cand, ref, model_type=None, num_layers=None, lang="en", 
#                  rescale_with_baseline=True, fname='cos_sim.png')