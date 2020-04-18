import pandas as pd
import numpy as np
import os
from bert_score import score
lc = 100
lr = 441

# 1st 100 lines are cands, next 441 lines are refs
file_name = '../data/ess3_small.txt'
df = pd.read_csv(file_name, delimiter='\n', header=None)
print("Total Sentences: ", len(df))

cands = df[:100].values.tolist()
print(len(cands))

refs = df[100:].values.tolist()
refs = [refs]  # * len(cands)
print(len(refs))

sol = score(cands, refs, model_type=None, num_layers=None, verbose=False,
            idf=False, device=None, batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=False)

print(len(sol))
