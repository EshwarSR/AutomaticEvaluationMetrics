import pandas as pd
import os
import nltk

dire = '../data/ASAP_AES/training_set_rel3.tsv'
res = '../results/asap_aes_final/asap_aes_scores_all_1726_2.tsv'

df = pd.read_csv(dire, sep='\t' ,encoding='ISO-8859–1')
df = df.loc[df['essay_set'] == 3]

needed_sents = []
total_ids = df['essay_id'].tolist()
for ess_id, ess in zip(df['essay_id'].tolist(), df['essay'].tolist()):
	sents_list = [sent for sent in nltk.sent_tokenize(ess)]
	num_sents = len(sents_list)
	if (num_sents >= 5 and num_sents <= 15):
		needed_sents.append(ess_id)
print(len(needed_sents))

fdf = pd.read_csv(res, sep='\t')
out = []
for one in needed_sents:
	entries = fdf.loc[fdf['cand id'] == one]
	out.append(entries)

new_df = pd.DataFrame()
new_df = pd.concat(out)
print(len(new_df))
new_df.to_csv("asap_aes_scores_all_1088.tsv",  sep="\t", index=False, header=True, encoding='utf8')