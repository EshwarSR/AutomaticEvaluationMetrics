from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import numpy as np
import os
import torch
from bert_score import score

csdir = '../data/WMT18Data/system-outputs/newstest2018/cs-en/'
cses = os.listdir(csdir)
rs = '../data/WMT18Data/references/newstest2018-csen-ref.en'
human = '../data/WMT18Data/DA-syslevel.csv'

refs = []
fr = open(rs, "r", encoding='utf-8')
while True:
	line = fr.readline() 
	if not line:
		break
	refs.append(line)

outlist = []
for cs in cses:
	lp = csdir[-6:-1]
	sys = cs[13:-6]
	hdf = pd.read_csv(human, sep=' ')

	hdf = hdf.loc[hdf['LP'] == lp]
	hdf = hdf.loc[hdf['SYSTEM'] == sys]
	hdf.reset_index(drop=True, inplace=True)

	cands = []
	fc = open(csdir + cs, "r", encoding='utf-8')
	while True:
		line = fc.readline()
		if not line:
			break
		cands.append(line)

	assert len(cands) == len(refs)

	P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=False,
	            idf=True, device=None, batch_size=64, nthreads=4, all_layers=False,
	            lang="en", return_hash=False, rescale_with_baseline=True)

	outlist += [[P.mean().item(), R.mean().item(), F1.mean().item(), hdf['HUMAN'].item()]]

out = pd.DataFrame(outlist, columns = ["P", "R", "F1", "H"])
print(out)

cp = spearmanr(out['P'], out['H'])
cr = spearmanr(out['R'], out['H'])
cf1 = spearmanr(out['F1'], out['H'])
pearson_corrlist = [[cp.correlation, cp.pvalue, cr.correlation, cr.pvalue, cr.correlation, cr.pvalue]]

pearson_corr = pd.DataFrame(pearson_corrlist, columns = ["P", "P-pval", "R", "R-pval", "F1", "F1-pval"])
print(pearson_corr)
