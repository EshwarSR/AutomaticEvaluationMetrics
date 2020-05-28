import pandas as pd
import numpy as np
import os

filename = '../data/CNN_DailyMail.tsv'
df = pd.read_csv(filename, delimiter='\t')

refs = df['reference'].to_list()
cands = df['candidate'].to_list()

# broadcasting refs so that len(cands) == len(refs)
assert len(cands) == len(refs)

from bert_score import score
P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=True,
            idf=False, device='cpu', batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=False)

odf = pd.DataFrame()
odf['cand id'] = df.index.to_list()
odf['similarity score'] = pd.DataFrame(F1)
odf['R score'] = pd.DataFrame(R)
odf['P score'] = pd.DataFrame(P)
odf['score'] = df['score']
odf.reset_index(drop=True, inplace=True)
odf.to_csv("outs.tsv", sep="\t", index=False, header=True, encoding='utf8')
