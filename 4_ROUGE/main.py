# from rouge.rouge import Rouge

# cand = ['what is going on', 'here we are going together']
# ref = ['here we are going together'] * 2

# rouge = Rouge()
# scores = rouge.get_scores(cand, ref)

# print(scores)
import time
from rouge.rouge import Rouge
import pandas as pd

fn = '../../Official_Repo/IISc-ML-Project/data/asap_sas.tsv'
df = pd.read_csv(fn, delimiter='\t')

can1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] != 2)]
can1.reset_index(drop=True, inplace=True)
cands = list(can1['EssayText'])
print("Candidate Sentences: ", len(cands))
# print(cands[0])

ref1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] == 2)]
ref1.reset_index(drop=True, inplace=True)
refs = list(ref1['EssayText'])
print("Reference Sentences: ", len(refs))
# print(refs[0])

cany = cands
cands = []
for one in cany:
	cands += [one] * len(refs)
refs = refs * len(cany)

assert len(cands) == len(refs), "cands and refs are different lengths"

# print(len(cands), type(cands))
# print(len(refs), type(refs))


rouge = Rouge()
start_time = time.time()
scores = rouge.get_scores(cands[:100], refs[:100])
end_time = time.time()
diff = (end_time - start_time) / 60

# print(scores)
# or
# scores = rouge.get_scores(hyps, refs, avg=True)
# print(len(scores))
print(scores[0]['rouge-l']['f'])
print("--- {0:5.3} minutes ---".format(diff))

# odf = pd.DataFrame()
# odf['cand id'] = can1['Id']
# odf['similarity score'] = pd.DataFrame(scores[:]['rouge-l']['f'])
# odf['score'] = can1['Score1']
# odf.reset_index(drop=True, inplace=True)
# print(odf.columns)
# odf.to_csv("outs.tsv", sep="\t", index=False, header=True)