import pandas as pd
import numpy as np
from pandas import DataFrame


from aes_processing import cand_data, ref_data
from bleu import BLEU


cand_df = pd.read_csv('../data/ASAP_AES/training_set_rel3.tsv', sep='\t', encoding='ISO-8859–1')
ref_df = pd.read_csv('../data/ASAP_AES/reference_3_aes.tsv', sep='\t')


n_gram = 4

cand_id, candidate_corpus, human_scores = cand_data(cand_df)
reference_corpus = ref_data(ref_df)

# print(candidate_corpus[0:4])

# print(BLEU(reference_corpus, reference_corpus[0], 4))

bleu_scores = []

for i in range(len(candidate_corpus)):
    bleu = BLEU(reference_corpus, candidate_corpus[i], n_gram)
    # print(bleu)
    bleu_scores.append(bleu)
print(max(bleu_scores))

# # for i in range(len(bleu)):
# #     print(bleu[i])

filename = "../results/asap_aes_results/BLEU_scores_aes.txt"
with open(filename, 'w') as f:
    f.write("candidate_id\tsimilarity\tscore\n")
    for i in range(len(candidate_corpus)):
        f.write("{0}\t{1}\t{2}\n".format(cand_id[i], bleu_scores[i], human_scores[i]))

