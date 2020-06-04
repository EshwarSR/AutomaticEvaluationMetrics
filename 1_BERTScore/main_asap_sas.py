import pandas as pd
import numpy as np
import os

filename = '../data/ASAP_SAS/candidates_data.tsv'
ref_file = '../data/ASAP_SAS/reference_data.tsv'
can = pd.read_csv(filename, delimiter='\t')
ref = pd.read_csv(ref_file, delimiter='\t')

cands = can['EssayText'].tolist()
print("Candidate Sentences: ", len(cands))

ref.reset_index(drop=True, inplace=True)
refs = ref['EssayText'].tolist()
print("Reference Sentences: ", len(refs))

# broadcasting refs so that len(cands) == len(refs)
refs = [refs] * len(cands)
assert len(cands) == len(refs)

print(len(cands), type(cands))
print(len(refs), type(refs))

from bert_score import score
P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=True, device='cpu', batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=True)

odf = pd.DataFrame()
odf['cand id'] = can['Id']
odf['F1 score'] = pd.DataFrame(F1)
odf['R score'] = pd.DataFrame(R)
odf['P score'] = pd.DataFrame(P)
odf['score'] = can['Score2']
odf.reset_index(drop=True, inplace=True)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
