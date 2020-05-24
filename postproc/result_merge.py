import os
import pandas as pd
import matplotlib.pyplot as plt

br = '../results/WMT18_Results/'
bdf = pd.read_csv(br + 'BLEU_scores.tsv', sep='\t')
edf = pd.read_csv(br + 'scores_almost.tsv', sep='\t')

bz = bdf.columns.to_list()
ez = edf.columns.to_list()

assert bz[:3] == ez[:3], "First 2 elements must be same in both lists"
assert bz[-1] == ez[-1], "Last element must be same in both lists"

fin = bz[:2]
fin += bz[2:-1]
fin += ez[2:-1]
fin += [bz[-1]]

assert len(edf['SYSTEM']) == len(bdf['SYSTEM']), "Both dfs have different number of systems"

out = []
for i in range(len(edf['SYSTEM'])):
	subdf = bdf.loc[(bdf['LP'] == edf['LP'][i]) & (bdf['SYSTEM'] == edf['SYSTEM'][i])]
	temp = subdf.values.tolist()[0][:-1] + edf.loc[i].to_list()[2:]
	# print(temp)
	out.append(temp)

ndf = pd.DataFrame(out, columns=fin)
ndf.to_csv("scores_all.tsv", sep="\t", index=False, header=True)

#####################################################################

bdf = pd.read_csv(br + 'BLEU_corr.tsv', sep='\t')
edf = pd.read_csv(br + 'corr_almost.tsv', sep='\t')

fdf = pd.DataFrame(columns=edf.columns.to_list())
for col in edf.columns.to_list():
	fin = pd.concat([bdf[col], edf[col]])
	fdf[col] = fin

fdf.reset_index(drop=True, inplace=True)
fdf.to_csv("corr_all.tsv", sep="\t", index=False, header=True)

#####################################################################
