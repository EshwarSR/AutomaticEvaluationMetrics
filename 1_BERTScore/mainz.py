import pandas as pd
import numpy as np
import os
from bert_score import score

filename = '../data/asap_sas.tsv'
df = pd.read_csv(filename, delimiter='\t')

can1 = df.loc[(df['EssaySet'] == 3)]
can1.reset_index(drop=True, inplace=True)
cands = list(can1['EssayText'])
print("Candidate Sentences: ", len(cands))
# print(cands[0])

ref1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] == 2)]
ref1.reset_index(drop=True, inplace=True)
refs = list(ref1['EssayText'])
print("Reference Sentences: ", len(refs))
# print(refs[0])

# broadcasting refs so that len(cands) == len(refs)
refs = [refs] * len(cands)
assert len(cands) == len(refs)

print(len(cands), type(cands))
print(len(refs), type(refs))

P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=True, device=None, batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=True)

print(len(F1))

# print(can1.columns)
odf = pd.DataFrame()
odf['cand id'] = can1['Id']
odf['similarity score'] = pd.DataFrame(F1)
odf['score'] = can1['Score1']
odf.reset_index(drop=True, inplace=True)
# print(odf.columns)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)

import matplotlib.pyplot as plt
plt.grid(True)
plt.hist(F1, bins=20)
plt.show()
plt.savefig('histogram_all.png')
plt.xlabel("Score1")
plt.ylabel("Pred_Score")
plt.grid(True)
plt.plot(odf['score'], odf['similarity score'] * 2, '.g')
plt.show()
plt.savefig('res/scatter.png')
max_score = 2
_, ax = plt.subplots(ncols=3, nrows=1, constrained_layout=True)
for i in range(max_score + 1):
  daf = odf.loc[(odf['score'] == i)]
  # if (i != max_score):
  #   ax[i].set_xlim(-0.03, 2.03)
  ax[i].set_ylim(0, 250)
  ax[i].hist(daf['similarity score'] * 2, bins = 20)
plt.show()
plt.savefig('histograms_ind.png')
from bert_score import plot_example
cand = cands[0]
ref = refs[0][0]
plot_example(cand, ref, model_type=None, num_layers=None, lang="en", 
                 rescale_with_baseline=True, fname='cos_sim.png')