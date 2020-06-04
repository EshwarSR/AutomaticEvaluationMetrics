import json
import nltk
import pandas as pd

dataset = []
with open("lqual.jsonl") as f:
     for line in f:
        j = json.loads(line.strip())
        if j["input"]["contents"]["system"] == "reference":
                continue
        if len(j["output"]["_responses"]["worker_ids"]) >=2:
            dataset.append(
                {
                    "reference": j["input"]["contents"]["reference"],
                    "candidate": j["input"]["contents"]["text"],
                    "score": j["output"]["overall"]
                }
            )

print("Count:", len(dataset))

df = pd.DataFrame(dataset)
df.to_csv("filtered_dataset.tsv", sep="\t", index=False)