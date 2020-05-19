import pandas as pd
import numpy as np
import os
from rouge.rouge import Rouge

filename = '../data/asap_sas.tsv'
df = pd.read_csv(filename, delimiter='\t')

can1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] != 2)]
can1.reset_index(drop=True, inplace=True)
cands = list(can1['EssayText'])
print("Candidate Sentences: ", len(cands))
# print(cands[0])

ref1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] == 2)]
ref1.reset_index(drop=True, inplace=True)
refs = list(ref1['EssayText'])
print("Reference Sentences: ", len(refs))
# print(refs[0])

rouge = Rouge()
R1 = []
R2 = []
R3 = []
for cand in cands:
	tempr1 = []
	tempr2 = []
	tempr3 = []
	for ref in refs:
		scores = rouge.get_scores(cand, ref)
		tempr1 += [one['rouge-1']['f'] for one in scores]
		tempr2 += [one['rouge-2']['f'] for one in scores]
		tempr3 += [one['rouge-l']['f'] for one in scores]
	R1 += [np.max(tempr1)]
	R2 += [np.max(tempr2)]
	R3 += [np.max(tempr3)]


# print(can1.columns)
odf = pd.DataFrame()
odf['cand id'] = can1['Id']
odf['ROUGE-1 Score'] = pd.DataFrame(R1)
odf['ROUGE-2 Score'] = pd.DataFrame(R2)
odf['ROUGE-L Score'] = pd.DataFrame(R3)
odf['score'] = can1['Score1']
odf.reset_index(drop=True, inplace=True)
# print(odf.columns)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
