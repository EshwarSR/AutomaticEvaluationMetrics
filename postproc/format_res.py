import os
import pandas as pd
import matplotlib.pyplot as plt

dim = '../results/WMT18_Results/BERTScore_WMT18'
human = '../data/WMT18Data/DA-syslevel.csv'
hdf = pd.read_csv(human, sep=' ')

corrs = []
scores= []
for gor in os.listdir(dim):
	if (gor[:5] == 'corr-'):
		corrs.append(dim + '/' + gor)
	else:
		scores.append(dim + '/' + gor)

fin = [["correlation", "Spearman", "Pearson", "Kendall", "Spearman", "Pearson", "Kendall", "Spearman", "Pearson", "Kendall"]]
for one in corrs:
	df = pd.read_csv(one, sep='\t')
	lp = one[-8:-6] + '-' + one[-6:-4]
	lissy = [lp]

	lissy += df['P'].to_list()
	lissy += df['R'].to_list()
	lissy += df['F1'].to_list()
	fin.append(lissy)
oyt = pd.DataFrame(fin, columns = ["metric", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) F1", "BERTScore RoBERTa MNLI (idf) F1", "BERTScore RoBERTa MNLI (idf) F1"])
oyt = oyt.T
# print(oyt)
# oyt.to_csv("corr.tsv", sep="\t", index=True, header=False)

outlist = []
for one in scores:
	df = pd.read_csv(one, sep='\t')
	df['H'] = df['H'].round(decimals=3)

	lp = one[-8:-6] + '-' + one[-6:-4]
	ndf = hdf.loc[hdf['LP'] == lp]
	ndf.reset_index(drop=True, inplace=True)
	ndf['HUMAN'] = ndf['HUMAN'].round(decimals=3)
	assert len(ndf['HUMAN']) == len(df['H']), "Sizes are not same"

	for i in range(len(ndf['HUMAN'])):
		odf = df.loc[df['H'] == ndf['HUMAN'][i]]
		if (len(odf) != 1):
			print(ndf)
			print("len(odf) = ", len(odf))
			for j in range(len(df['H'])):
				print("{0:25}   {1:25}".format(df['H'][j], ndf['HUMAN'][i]))
			exit(1)
		print([lp, ndf['SYSTEM'][i], odf['P'].item(), odf['R'].item(), odf['F1'].item(), odf['H'].item()])
		outlist.append([lp, ndf['SYSTEM'][i], odf['P'].item(), odf['R'].item(), odf['F1'].item(), odf['H'].item()])

exit(1)
out = pd.DataFrame(outlist, columns = ["LP", "SYSTEM", "BERTScore RoBERTa MNLI (idf) P", "BERTScore RoBERTa MNLI (idf) R", "BERTScore RoBERTa MNLI (idf) F1", "HUMAN"])
print(out)
# out.to_csv("scores.tsv", sep="\t", index=False, header=True)
