import pandas as pd
import os

res = '../results/asap_aes_results/'

out = pd.DataFrame(columns=[])

ts = os.listdir(res)
dr = pd.read_csv(res + ts[0], sep='\t')

out['cand id'] = dr['cand id']
for an in ts:
	fn = res + an
	df = pd.read_csv(fn, sep='\t')
	out[an[:-4]] = df['similarity score']

out['score'] = dr['score']
out.to_csv("asap_aes_scores_ally.tsv",  sep="\t", index=False, header=True, encoding='utf8')