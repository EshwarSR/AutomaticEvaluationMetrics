import pandas as pd

data = pd.read_csv("../data/asap_sas.tsv", sep="\t")

# Essay 3 data
data = data[data["EssaySet"] == 3]
candidates_data = data[data["Score1"] != 2]

print("Total candidates data:", len(candidates_data))
candidates_data.to_csv("../data/candidates_data.tsv", sep="\t", index=False)

reference_data = data[data["Score1"] == 2]
print("Total reference data:", len(reference_data))
reference_data.to_csv("../data/reference_data.tsv", sep="\t", index=False)
