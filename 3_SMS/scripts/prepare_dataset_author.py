import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from wmd import WMD
import numpy as np
import os

# REFERENCE_FILE = "../data/reference_data.tsv"
REFERENCE_FILE = "../data/aes_reference_data.tsv"
# CANDIDATES_FILE = "../data/candidates_data.tsv"
CANDIDATES_FILE = "../data/aes_candidates_data.tsv"
# score_field = "Score1"
score_field = "domain1_score"
# essay_field = "EssayText"
essay_field = "essay"
# id_field = "Id"
id_field = "essay_id"

reference_data = pd.read_csv(REFERENCE_FILE, sep="\t")
reference_data = reference_data.to_dict("records")
reference_data = [reference_data[0][essay_field]]

candidates_data = pd.read_csv(CANDIDATES_FILE, sep="\t")
candidates_data = candidates_data.to_dict("records")
candidates_data = [sample[essay_field] for sample in candidates_data]

reference_data = reference_data * len(candidates_data)

df = pd.DataFrame(list(zip(candidates_data, reference_data)))
print("Writing the file")
df.to_csv("data_as_per_author.tsv", sep="\t", index=False, header=False)
