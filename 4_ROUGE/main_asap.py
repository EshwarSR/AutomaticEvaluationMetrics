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
sc = []
for cand, canid, hscore in zip(cands, can1['essay_id'], can1['domain1_score']):
	scores = rouge.get_scores(cand, ref)[0]
	sc.append([ canid, scores['rouge-1']['f'], scores['rouge-1']['p'], scores['rouge-1']['r'], \
				scores['rouge-2']['f'], scores['rouge-2']['p'], scores['rouge-2']['r'], \
				scores['rouge-l']['f'], scores['rouge-l']['p'], scores['rouge-l']['r'], hscore ])

# print(can1.columns)
odf = pd.DataFrame(sc, columns=['cand id', 'R1f', 'R1p', 'R1r', 'R2f', 'R2p', 'R2r', 'RLf', 'RLp', 'RLr', 'score'])
odf.reset_index(drop=True, inplace=True)
odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
