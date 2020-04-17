import pandas as pd
import numpy as np
import os
from bert_score import score

file_name = 'Data/train.tsv'
df = pd.read_csv(file_name, delimiter='\t')
# print(df) # print(df.head()) # print(df.columns) # print(len(df))
tot = df.loc[(df['EssaySet'] == 3)]
print("Total sentences: ", len(tot))

ref1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] == 2)]
# print(ref1)
ref1.reset_index(drop=True, inplace=True)
arrs = list(ref1['EssayText'])
# print(arrs[0])
print("Reference Sentences: ", len(arrs))

can1 = df.loc[(df['EssaySet'] == 3) & (df['Score1'] != 2)]
can1.reset_index(drop=True, inplace=True)
cees = list(can1['EssayText'])
# print(cees[0])
print("Candidate Sentences: ", len(cees))

# When you are running this cell for the first time,
# it will download the BERT model which will take relatively longer.
# sol = []
# count = 0
# for cand in cees:
#     cand = [cand]
#     refs = [arrs]
#     sol.append(score(cand, refs, lang="en", verbose=True,
#                      rescale_with_baseline=False, return_hash=False))
#     print("P:", sol[count][0], "R:", sol[count][1], "F1:", sol[count][2])
#     count += 1
#     print(count)
# print(len(sol))

refs = [arrs] * len(cees)
cands = cees
sol = score(cands, refs, model_type=None, num_layers=None, verbose=False,
            idf=False, device=None, batch_size=64, nthreads=4, all_layers=False,
            lang="en", return_hash=False, rescale_with_baseline=False)

print(len(sol))
