import pandas as pd
import numpy as np
import os
from rouge.rouge import Rouge

filename = '../data/ASAP_AES/training_set_rel3.tsv'
ref_file = '../data/ASAP_AES/reference_3_aes.tsv'
df = pd.read_csv(filename, delimiter='\t', encoding='ISO-8859–1')
ref1 = pd.read_csv(ref_file, delimiter='\t')

can1 = df.loc[df['essay_set'] == 3]
can1.reset_index(drop=True, inplace=True)
cands = list(can1['essay'])
print("Candidate Sentences: ", len(cands))
# print(cands[0])

ref1.reset_index(drop=True, inplace=True)
ref = ref1['Reference'][0]

rouge = Rouge()
R1 = []
R2 = []
R3 = []
for cand in cands:
	scores = rouge.get_scores(cand, ref)
	R1 += [one['rouge-1']['f'] for one in scores]
	R2 += [one['rouge-2']['f'] for one in scores]
	R3 += [one['rouge-l']['f'] for one in scores]

# print(can1.columns)
odf = pd.DataFrame()
odf['cand id'] = can1['essay_id']
odf['ROUGE-1 Score'] = pd.DataFrame(R1)
odf['ROUGE-2 Score'] = pd.DataFrame(R2)
odf['ROUGE-L Score'] = pd.DataFrame(R3)
odf['score'] = can1['domain1_score']
odf.reset_index(drop=True, inplace=True)
# print(odf.columns)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
