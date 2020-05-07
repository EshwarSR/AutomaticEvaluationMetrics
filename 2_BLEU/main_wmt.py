from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from pandas import DataFrame


from processing import wmt_data_cands, wmt_data_refs
from bleu import BLEU

allcees = '../data/WMT18Data/system-outputs/newstest2018/'
allrefs = '../data/WMT18Data/references/'
human = '../data/WMT18Data/DA-syslevel.csv'

act_dir = []
todo = []
cdirs = os.listdir(allcees)
for csdir in cdirs:
	if csdir[-2:] == 'en':
		todo += [csdir]
		act_dir += [allcees + csdir]

for csdir in act_dir:
	stry = csdir[-5:-3] + csdir[-2:]
	rs = allrefs + 'newstest2018-' + stry + '-ref.en'
	cses = os.listdir(csdir)

	refs = []
	fr = open(rs, "r", encoding='utf-8')
	while True:
		line = fr.readline() 
		if not line:
			break
		refs.append(wmt_data_refs(line))

	outlist = []
	for cs in cses:
		bleu = []
		lp = csdir[-5:]
		sys = cs[13:-6]
		hdf = pd.read_csv(human, sep=' ')

		hdf = hdf.loc[hdf['LP'] == lp]
		hdf = hdf.loc[hdf['SYSTEM'] == sys]
		hdf.reset_index(drop=True, inplace=True)

		cands = []
		fc = open(csdir + '/' + cs, "r", encoding='utf-8')
		while True:
			line = fc.readline()
			if not line:
				break
			cands.append(wmt_data_cands(line))

		assert len(cands) == len(refs)

		for i in range(len(cands)):
			bleu.append(BLEU(refs[i], cands[i], 4))


		outlist.append([sum(bleu)/len(bleu), hdf['HUMAN']])

	out = pd.DataFrame(outlist, columns = ["P", "H"])
	print(out)
	out.to_csv("BLEU-scores-" + stry + ".tsv", sep="\t", index=False, header=True)

	cp = spearmanr(out['P'], out['H'])
	# cr = spearmanr(out['R'], out['H'])
	# cf1 = spearmanr(out['F1'], out['H'])
	spearman_corrlist = [['Spearman Rank Correlation', cp.correlation, cp.pvalue]]

	cp = pearsonr(out['P'], out['H'])
	# cr = pearsonr(out['R'], out['H'])
	# cf1 = pearsonr(out['F1'], out['H'])
	pearson_corrlist = [['Pearson Correlation Coefficient', cp[0], cp[1]]]

	cp = kendalltau(out['P'], out['H'])
	# cr = kendalltau(out['R'], out['H'])
	# cf1 = kendalltau(out['F1'], out['H'])
	kend_corrlist = [['Kendall Rank Correlation', cp[0], cp[1]]]

	fin_list = spearman_corrlist + pearson_corrlist + kend_corrlist

	corrs = pd.DataFrame(fin_list, columns = ["Type of Correlation", "P", "P-pval"])
	print(corrs)
	corrs.to_csv("corr-" + stry + ".tsv", sep="\t", index=False, header=True)
	print(stry + "done")
	print(todo)
