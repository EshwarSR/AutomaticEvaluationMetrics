from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import numpy as np
import os
import torch
from bert_score import score
from time import perf_counter
import tracemalloc

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
        refs.append(line)

    outlist = []
    for cs in cses:
        start = perf_counter()
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
            cands.append(line)

        assert len(cands) == len(refs)

        P, R, F1 = score(cands, refs, model_type=None, num_layers=None, verbose=False,
                         idf=True, device=None, batch_size=64, nthreads=4, all_layers=False,
                         lang="en", return_hash=False, rescale_with_baseline=True)
        outlist.append([P.mean().item(), R.mean().item(), F1.mean().item(), hdf['HUMAN'].item()])
        end = perf_counter()
        print("LP : {0}   SYS: {1}   time taken: {2:5.3f}".format(lp, sys, end - start))

    out = pd.DataFrame(outlist, columns = ["P", "R", "F1", "H"])
    # print(out)
    out.to_csv("scores-" + stry + ".tsv", sep="\t", index=False, header=True)

    cp = spearmanr(out['P'], out['H'])
    cr = spearmanr(out['R'], out['H'])
    cf1 = spearmanr(out['F1'], out['H'])
    spearman_corrlist = [['Spearman Rank Correlation', cp.correlation, cp.pvalue, cr.correlation, cr.pvalue, cf1.correlation, cf1.pvalue]]

    cp = pearsonr(out['P'], out['H'])
    cr = pearsonr(out['R'], out['H'])
    cf1 = pearsonr(out['F1'], out['H'])
    pearson_corrlist = [['Pearson Correlation Coefficient', cp[0], cp[1], cr[0], cr[1], cf1[0], cf1[1]]]

    cp = kendalltau(out['P'], out['H'])
    cr = kendalltau(out['R'], out['H'])
    cf1 = kendalltau(out['F1'], out['H'])
    kend_corrlist = [['Kendall Rank Correlation', cp[0], cp[1], cr[0], cr[1], cf1[0], cf1[1]]]

    fin_list = spearman_corrlist + pearson_corrlist + kend_corrlist

    corrs = pd.DataFrame(fin_list, columns = ["Type of Correlation", "P", "P-pval", "R", "R-pval", "F1", "F1-pval"])
    # print(corrs)
    corrs.to_csv("corr-" + stry + ".tsv", sep="\t", index=False, header=True)
    print(stry + " done.")
    print(todo)
