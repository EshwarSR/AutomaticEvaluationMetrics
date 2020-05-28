import pandas as pd
import numpy as np
import os
from rouge.rouge import Rouge

filename = '../data/CNN_DailyMail.tsv'
df = pd.read_csv(filename, delimiter='\t')

refs = df['reference'].to_list()
cands = df['candidate'].to_list()

rouge = Rouge()
sc = []

scores = rouge.get_scores(cands, refs)


assert len(scores) == len(df), "Something wrong, len(scores) must be equal to len(df)"

for idx in range(len(scores)):
	sc.append([ idx, scores[idx]['rouge-1']['f'], scores[idx]['rouge-1']['p'], scores[idx]['rouge-1']['r'], \
				scores[idx]['rouge-2']['f'], scores[idx]['rouge-2']['p'], scores[idx]['rouge-2']['r'], \
				scores[idx]['rouge-l']['f'], scores[idx]['rouge-l']['p'], scores[idx]['rouge-l']['r'], df['score'][idx] ])

# print(can1.columns)
odf = pd.DataFrame(sc, columns=['cand id', 'R1f', 'R1p', 'R1r', 'R2f', 'R2p', 'R2r', 'RLf', 'RLp', 'RLr', 'score'])
odf.reset_index(drop=True, inplace=True)
odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
