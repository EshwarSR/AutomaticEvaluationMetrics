import pandas as pd
import numpy as np
import os

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

# broadcasting refs so that len(cands) == len(refs)
refs = [ref] * len(cands)
assert len(cands) == len(refs)

print(len(cands), type(cands))
print(len(refs), type(refs))

from bert_score import score
P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=False, device='cpu', batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=False)

# print(len(F1))

# print(can1.columns)
odf = pd.DataFrame()
odf['cand id'] = can1['essay_id']
odf['similarity score'] = pd.DataFrame(F1)
odf['R score'] = pd.DataFrame(R)
odf['P score'] = pd.DataFrame(P)
odf['score'] = can1['domain1_score']
odf.reset_index(drop=True, inplace=True)
# print(odf.columns)

odf.to_csv("outs.tsv", sep="\t", index=False, header=True)
