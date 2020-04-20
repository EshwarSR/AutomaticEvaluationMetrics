import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from wmd import WMD
import numpy as np
import os

REFERENCE_FILE = "../data/reference_data.tsv"


def calculate_similarity(candidate, next_id, emb):
    s = time.time()
    can_doc = calculator.nlp(candidate["EssayText"])
    similarities = []
    next_id, emb, can_id_list, can_weights = calculator.get_embeddings_ids_weights(
        can_doc, next_id, emb, method)
    nbow = {
        "hypothesis": ("hypothesis", can_id_list, can_weights)
    }

    for id, item in processed_refs.items():
        ref_weights = item["weights"]
        ref_id_list = item["id_list"]
        nbow[id] = (id, ref_id_list, ref_weights)

    calc = WMD(emb, nbow, vocabulary_min=1)
    dist = calc.nearest_neighbors(
        "hypothesis", k=len(processed_refs), early_stop=1)

    for id, score in dist:
        similarity = np.exp(-score)
        similarities.append({
            "candidate_id": candidate["Id"],
            "reference_id": id,
            "similarity": similarity,
            "score": candidate["Score1"]
        })
    print("Time taken for candidate " +
          str(candidate["Id"]) + " is " + str(time.time() - s))

    return similarities


def get_all_ref_emb_weights():
    reference_data = pd.read_csv(REFERENCE_FILE, sep="\t")
    reference_data = reference_data.to_dict("records")
    processed_refs = {}
    next_id = 0
    emb = {}
    for item in reference_data:
        doc = calculator.nlp(item["EssayText"])
        next_id, emb, id_list, weights = calculator.get_embeddings_ids_weights(
            doc, next_id, emb, method)
        processed_refs[item["Id"]] = {
            "id_list": id_list,
            "weights": weights
        }

    return processed_refs, next_id, emb


st = time.time()
CANDIDATES_FILE = sys.argv[1]
model = sys.argv[2]
method = sys.argv[3]
parallel = False
candidates_data = pd.read_csv(CANDIDATES_FILE, sep="\t")
candidates_data = candidates_data.to_dict("records")

calculator = EMDMetrics(model)
print("Loaded model", time.time() - st)

st = time.time()
processed_refs, next_id, emb = get_all_ref_emb_weights()
print("Done preprocessing references", time.time() - st)


results_file_name = CANDIDATES_FILE.rsplit("/", 1)[1]
results_file_name = results_file_name.split(".")[0]
results_file_name = results_file_name + "_" + model + "_" + method + ".tsv"
results_file_name = "../results/" + results_file_name

if os.path.isfile(results_file_name):
    final_results = pd.read_csv(results_file_name, sep="\t")
    max_id = final_results["candidate_id"].max()
    final_results = final_results.to_dict("records")
    for i in range(len(candidates_data)):
        if candidates_data[i]["Id"] == max_id:
            idx = i + 1
            break
    print("File already present.")
    print("Completed till ID:", max_id)
    print("Continuing from ID:", candidates_data[idx]["Id"])

else:
    final_results = []
    idx = 0

if parallel == True:
    with ProcessPoolExecutor(max_workers=2) as executor:
        for resp in executor.map(calculate_similarity, candidates_data):
            final_results.extend(resp)
else:
    # for idx, candidate in enumerate(candidates_data):
    while idx < len(candidates_data):
        candidate = candidates_data[idx]
        try:
            resp = calculate_similarity(candidate, next_id, emb)
            final_results.extend(resp)
        except:
            print("ERROR WHILE PROCESSING:", candidate["Id"])

        if idx % 100 == 0:
            final_results_df = pd.DataFrame(final_results)
            final_results_df.to_csv(results_file_name, sep="\t", index=False)
        idx += 1

print("Done with similarity")
final_results_df = pd.DataFrame(final_results)
print("Created Dataframe")
final_results_df.to_csv(results_file_name, sep="\t", index=False)
print("Finished writing file")