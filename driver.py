import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys

# methods = ["wms", "sms", "w+sms"]
# models = ["glove", "elmo", "bert"]
# ref = "Hi, I'm Eshwar."
# cand = "Hello, I'm Eshwar."

# for model in models:
#     st = time.time()
#     calculator = EMDMetrics(model)
#     print("Time to load models:", model, time.time() - st)
#     for method in methods:
#         sim = calculator.get_similarity(ref, cand, method)
#         print(model, method, sim)

st = time.time()
filename = sys.argv[1]
model = sys.argv[2]
method = sys.argv[3]
data = pd.read_csv(filename, sep="\t",
                   names=["hypothesis", "reference"])

calculator = EMDMetrics(model)

print("Loaded model", time.time() - st)
for i, row in enumerate(data.to_dict("records")):
    hyp = row["hypothesis"].lower()
    ref = row["reference"].lower()

    sim = calculator.get_similarity(hyp, ref, method)
    print(sim)
