import pandas as pd
import time
import sys
import numpy as np
import os
import spacy
from wmd import WMD
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

stop_words = set(stopwords.words('english'))


def get_embeddings_ids_weights(text, next_id, emb):
    ids_list = []
    weights = []
    doc = nlp(text)

    sents_list = [sent.text for sent in doc.sents]
    sent_embeddings = model.encode(sents_list)

    for sent, sent_emb in zip(doc.sents, sent_embeddings):
        count = 0
        for word in sent:
            if word.text.isalpha() and word.text.lower() not in stop_words:
                count += 1
        if count > 0:
            ids_list.append(next_id)
            emb[next_id] = sent_emb
            weights.append(count)
            next_id += 1

    weights = np.array(weights, dtype=np.float32) / sum(weights)

    return next_id, emb, ids_list, weights


def get_emb_nbow(candidate, reference):
    emb = {}
    next_id = 1

    next_id, emb, can_id_list, can_weights = get_embeddings_ids_weights(
        candidate, next_id, emb)
    next_id, emb, ref_id_list, ref_weights = get_embeddings_ids_weights(
        reference, next_id, emb)

    nbow = {
        "reference": ("reference", ref_id_list, ref_weights),
        "hypothesis": ("hypothesis", can_id_list, can_weights)
    }

    return emb, nbow

# Driver function


def get_similarity_dist(candidate, reference):
    emb, nbow = get_emb_nbow(candidate, reference)
    # print("emb:", emb.keys())
    # print("nbow:", nbow)
    calc = WMD(emb, nbow, vocabulary_min=1)
    dist = calc.nearest_neighbors("reference", k=1, early_stop=1)
    # print("Dist:", dist)
    dist = dist[0][1]
    similarity = np.exp(-dist)
    return similarity, dist


def process_dataset(dataset, results_file_name):
    # Dataset should have candidate,reference and score
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


#################
# Main Function #
#################

st = time.time()

# CNN MailDataset
DATASET_FILE = "../data/CNN_DailyMail.tsv"
print("Reading dataset from", DATASET_FILE)
dataset = pd.read_csv(DATASET_FILE, sep="\t").to_dict("records")

nlp = spacy.load("en_core_web_md")
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
print("Loaded model", time.time() - st)

results_file_name = "../results/SentBERT/CNN_DailyMail_SentBERT.tsv"
process_dataset(dataset, results_file_name)


# Essay Dataset
REFERENCE_FILE = "../data/ASAP_AES/aes_reference_data.tsv"
CANDIDATES_FILE = "../data/ASAP_AES/aes_candidates_data.tsv"
print("Reading dataset from", REFERENCE_FILE, CANDIDATES_FILE)

candidates_data = pd.read_csv(CANDIDATES_FILE, sep="\t")
print("Number of candidates:", len(candidates_data))
candidates_data = candidates_data.to_dict("records")

reference_data = pd.read_csv(REFERENCE_FILE, sep="\t")
print("Number of references:", len(reference_data))
reference_data = reference_data.to_dict("records")

score_field = "domain1_score"
essay_field = "essay"
id_field = "essay_id"

processed_dataset = []
for cand in candidates_data:
    for ref in reference_data:
        sample = {
            "candidate": cand[essay_field],
            "candidate_id": cand[id_field],
            "reference": ref[essay_field],
            "reference_id": ref[id_field],
            "score": cand[score_field]
        }
        processed_dataset.append(sample)

results_file_name = "../results/SentBERT/ASAP_AES_SentBERT.tsv"
process_dataset(processed_dataset, results_file_name)
