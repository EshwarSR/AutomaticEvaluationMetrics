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
odf['Id'] = can1['Id']
odf['Similarity'] = pd.DataFrame(F1)
odf['Score1'] = can1['Score1']
odf.reset_index(drop=True, inplace=True)
# print(odf.columns)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)