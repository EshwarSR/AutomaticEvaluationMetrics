import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
import numpy as np
import os

st = time.time()

DATASET_FILE = "../data/CNN_DailyMail.tsv"
print("Reading dataset from", DATASET_FILE)
dataset = pd.read_csv(DATASET_FILE, sep="\t").to_dict("records")

MODEL = sys.argv[1]
METHOD = sys.argv[2]
calculator = EMDMetrics(MODEL)

print("Loaded model", time.time() - st)


results_file_name = DATASET_FILE.rsplit(
    "/", 1)[1] if "/" in DATASET_FILE else DATASET_FILE
results_file_name = results_file_name.split(".")[0]
results_file_name = results_file_name + "_" + MODEL + "_" + METHOD + ".tsv"
results_file_name = "../results/CNN_DailyMail_results/" + results_file_name


final_results = []
for idx, sample in enumerate(dataset):
    s = time.time()
    try:
        cand = sample["candidate"]
        ref = sample["reference"]
        # print("Reference:", ref)
        # print("Candidate:",cand)
        sim, dist = calculator.get_similarity_dist(cand, ref, METHOD)
        sample["similarity"] = sim
        sample["distance"] = dist
        final_results.append(sample)
    except:
        print("ERROR WHILE PROCESSING:", sample)

    print("Time taken for candidate", idx, "is", time.time() - s)

    if idx % 100 == 99:
        final_results_df = pd.DataFrame(final_results)
        final_results_df.to_csv(results_file_name, sep="\t", index=False)


print("Done with similarity")
final_results_df = pd.DataFrame(final_results)
print("Created Dataframe")
print(final_results_df)
final_results_df.to_csv(results_file_name, sep="\t", index=False)
print("Finished writing file")
