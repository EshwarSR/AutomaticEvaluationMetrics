import time
from emd_metrics import EMDMetrics

methods = ["wms", "sms", "w+sms"]
models = ["glove", "elmo", "bert"]
ref = "Hi, I'm Eshwar."
cand = "Hello, I'm Eshwar."

for model in models:
    st = time.time()
    calculator = EMDMetrics(model)
    print("Time to load models:", model, time.time() - st)
    for method in methods:
        sim = calculator.get_similarity(ref, cand, method)
        print(model, method, sim)
