import pandas as pd


data = pd.read_csv("../data/asap_sas.tsv", sep="\t")

# Length of essay 3 data
data = data[data["EssaySet"] == 3]
print("Total data:", len(data))

data.to_csv("../data/candidates_data.tsv", sep="\t", index=False)


reference_data = data[data["Score1"] == 2]
print("Total data:", len(reference_data))

reference_data.to_csv("../data/reference_data.tsv", sep="\t", index=False)

# with open("data/processed_sas.tsv", "w") as f:
#     for hypothesis in data["EssayText"].tolist():
#         for reference in reference_data["EssayText"].tolist():
#             f.write(hypothesis+"\t"+reference+"\n")

# print("Done")
