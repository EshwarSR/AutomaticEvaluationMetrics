import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from wmd import WMD
import numpy as np


def calculate_similarity(candidate):
    s = time.time()
    can_doc = calculator.nlp(candidate["EssayText"])
    similarities = []
    for id, item in processed_refs.items():
        next_id = item["next_id"]
        ref_weights = item["weights"]
        ref_id_list = item["id_list"]
        emb = item["emb"]

        next_id, emb, can_id_list, can_weights = calculator.get_embeddings_ids_weights(
            can_doc, next_id, emb, method)

        nbow = {
            "reference": ("reference", ref_id_list, ref_weights),
            "hypothesis": ("reference", can_id_list, can_weights)
        }

        calc = WMD(emb, nbow, vocabulary_min=1)
        dist = calc.nearest_neighbors("reference", k=1, early_stop=1)[
            0][1]
        similarity = np.exp(-dist)

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
    reference_data = pd.read_csv("../data/reference_data.tsv", sep="\t")
    reference_data = reference_data.to_dict("records")
    processed_refs = {}

    for item in reference_data:
        next_id = 0
        emb = {}
        doc = calculator.nlp(item["EssayText"])
        next_id, emb, id_list, weights = calculator.get_embeddings_ids_weights(
            doc, next_id, emb, method)
        processed_refs[item["Id"]] = {
            "next_id": next_id,
            "emb": emb,
            "id_list": id_list,
            "weights": weights
        }

    return processed_refs


st = time.time()
model = sys.argv[1]
method = sys.argv[2]
parallel = False
candidates_data = pd.read_csv("../data/cand_100.tsv", sep="\t")
candidates_data = candidates_data.to_dict("records")

calculator = EMDMetrics(model)
print("Loaded model", time.time() - st)

processed_refs = get_all_ref_emb_weights()
print("Done preprocessing references")

final_results = []
if parallel == True:
    with ProcessPoolExecutor(max_workers=2) as executor:
        for resp in executor.map(calculate_similarity, candidates_data):
            final_results.extend(resp)
else:
    for candidate in candidates_data:
        resp = calculate_similarity(candidate)
        final_results.extend(resp)

print("Done with similarity")
final_results = pd.DataFrame(final_results)
print("Created Dataframe")
final_results.to_csv("../results/cand_100_"+model+"_" +
                     method+".tsv", sep="\t", index=False)
print("Finished writing file")
