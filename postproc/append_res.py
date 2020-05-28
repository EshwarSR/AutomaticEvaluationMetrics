import pandas as pd
import os

res = '../results/CNN_DailyMail_results/'
out = pd.DataFrame(columns=[])
ts = os.listdir(res)
dr = pd.read_csv(res + 'berti_cnn.tsv', sep='\t')
out['cand id'] = dr['cand id']

ks = [ex for ex in ts if ex[:3] == 'CNN']
for an in ks:
	fn = res + an
	df = pd.read_csv(fn, sep='\t')
	out[an[14:-4]] = df['similarity']

out['score'] = dr['score']
out.to_csv("cnn_smses.tsv",  sep="\t", index=False, header=True, encoding='utf8')