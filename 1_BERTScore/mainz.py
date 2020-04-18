import pandas as pd
import numpy as np
import os
from bert_score import score

fn1 = '../data/cand_100.tsv'
fn2 = '../data/reference_data.tsv'
df1 = pd.read_csv(fn1, delimiter='\t')
df2 = pd.read_csv(fn2, delimiter='\t')

cands = df1.values.tolist()
print("Candidate Sentences: ", len(cands))
refs = df2.values.tolist()
print("Reference Sentences: ", len(refs))

# broadcasting refs so that len(cands) == len(refs)
refs = [refs] * len(cands)
assert len(cands) == len(refs)

sol = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=False, device=None, batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=False)

print(len(sol))
