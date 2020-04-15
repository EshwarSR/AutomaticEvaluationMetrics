import pandas as pd
import sys
import spacy
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

datasets = {
    "asap_aes.xlsx": {
        "id": "essay_id",
        "text": "essay"
    },
    "asap_sas.tsv": {
        "id": "Id",
        "text": "EssayText"
    },
    "test.tsv": {
        "id": "Id",
        "text": "EssayText"
    }
}
st = time.time()
nlp = spacy.load('en_core_web_md')
print("Loaded Spacy Model", time.time() - st)


def dump_object(sample):
    fn_st = time.time()
    id = sample[id_field]
    text = sample[text_field]

    file_name = objs_dir + str(id) + '.spcy'

    doc = nlp(text)
    doc.to_disk(file_name)

    # doc2 = Doc(nlp.vocab).from_disk(file_name)

    print("Time for processing", id, time.time() - fn_st)
    return id


dataset = sys.argv[1]
id_field = datasets[dataset]["id"]
text_field = datasets[dataset]["text"]
objs_dir = "./data/spacy_objs/" + dataset.split(".")[0] + "/"
if not os.path.exists(objs_dir):
    os.makedirs(objs_dir)

if ".xlsx" in dataset:
    data = pd.read_excel("./data/"+dataset)
else:
    data = pd.read_csv("./data/"+dataset, sep="\t")
data = data.to_dict("records")
print("Total data:", len(data))

with ProcessPoolExecutor(max_workers=6) as executor:
    for id in executor.map(dump_object, data):
        if id % 1000 == 0:
            print("Processed", id, "documents")

print("Total time:", time.time() - st)
