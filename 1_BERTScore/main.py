from scipy.stats import spearmanr, pearsonr, kendalltau
import sys
import pandas as pd
import numpy as np
import os
import torch
from bert_score import score
from time import perf_counter

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

finlist = [["correlation", "Spearman", "Pearson", "Kendall", "Spearman", "Pearson", "Kendall", "Spearman", "Pearson", "Kendall"]]
outlist = []
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
        outlist.append([lp, sys, P.mean().item(), R.mean().item(), F1.mean().item(), hdf['HUMAN'].item()])
        # P, R, F1 = (torch.rand(1), torch.rand(1), torch.rand(1))
        # outlist.append([lp, sys, P.item(), R.item(), F1.item(), hdf['HUMAN'].item()])
        end = perf_counter()
        print("LP : {0:10}SYS: {1:30s}time taken: {2:5.3f}".format(lp, sys, end - start))
        # print(outlist[-1])
    sz = len(cses)
    pees = [row[2] for row in outlist[-sz:]]
    arrs = [row[3] for row in outlist[-sz:]]
    effs = [row[4] for row in outlist[-sz:]]
    hues = [row[5] for row in outlist[-sz:]]

    lissy = [csdir[-5:]]
    src = spearmanr(pees, hues)
    pcc = pearsonr(pees, hues)
    ktc = kendalltau(pees, hues)
    lissy += [src.correlation, pcc[0], ktc[0]]

    src = spearmanr(arrs, hues)
    pcc = pearsonr(arrs, hues)
    ktc = kendalltau(arrs, hues)
    lissy += [src.correlation, pcc[0], ktc[0]]

    src = spearmanr(effs, hues)
    pcc = pearsonr(effs, hues)
    ktc = kendalltau(effs, hues)
    lissy += [src.correlation, pcc[0], ktc[0]]
    finlist.append(lissy)

out = pd.DataFrame(outlist, columns = ["LP", "SYSTEM", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) F1", "HUMAN"])
# print(out)
out.to_csv("scores.tsv", sep="\t", index=False, header=True)

oyt = pd.DataFrame(finlist, columns = ["metric", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) F1", "BERTScore RoBERTa MNLI (idf) F1", "BERTScore RoBERTa MNLI (idf) F1"])
oyt = oyt.T
# print(oyt)
oyt.to_csv("corr.tsv", sep="\t", index=True, header=False)
