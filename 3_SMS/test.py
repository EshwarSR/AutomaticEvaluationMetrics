import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
import numpy as np
import os


def process(text):
    doc = calculator.nlp(text)
    next_id = 0
    emb = {}
    next_id, emb, id_list, weights = calculator.get_embeddings_ids_weights(
        doc, next_id, emb, METHOD)


MODEL = "elmo"
METHOD = "sms"
sample = {'reference': "Archaeologists have found 00 stones shaped by early humans using 'simple techniques' close to the west shore of Lake Turkana in Kenya. The tools are thought to be 000,000 years older than the first Homo species. They are also 000,000 years older than other stone tools found previously. Scientists say it could reshape ideas about how our own species evolved.",
          'candidate': "Archaeologists have discovered ▃ stone flakes and used to help shape the tools-just west of Lake Turkana in Kenya. The stone tools were discovered in sediment along the west coast of Lake Turkana. The discovery could fundamentally change the view of human evolution as-if correct-the date of the tools is ▃,▃ years older than any other others found previously. A chimpanzee's ▃.▃ million years ago. The first thought to have appeared in Africa ▃.▃ million decades ago-the Hand. .", 'score': 0.2}
cand = sample["candidate"]
ref = sample["reference"]
calculator = EMDMetrics(MODEL)

print("processing candidate")
process(cand)
print("processing reference")
process(ref)

print("Getting similarity")
sim, dist = calculator.get_similarity_dist(cand, ref, METHOD)
print("sim:", sim)
print("dist:", dist)
