import pandas as pd

# data = pd.read_csv("../data/asap_sas.tsv", sep="\t")

# # Essay 3 data
# data = data[data["EssaySet"] == 3]
# candidates_data = data[data["Score1"] != 2]

# print("Total candidates data:", len(candidates_data))
# candidates_data.to_csv("../data/candidates_data.tsv", sep="\t", index=False)

# reference_data = data[data["Score1"] == 2]
# print("Total reference data:", len(reference_data))
# reference_data.to_csv("../data/reference_data.tsv", sep="\t", index=False)


data = pd.read_excel("../data/asap_aes.xlsx")

# Essay 3 data
data = data[data["essay_set"] == 3]
candidates_data = data[data["domain1_score"] != 3]

print("Total candidates data:", len(candidates_data))
candidates_data.to_csv("../data/aes_candidates_data.tsv",
                       sep="\t", index=False)

reference_data = data[data["domain1_score"] == 3]
print("Total reference data:", len(reference_data))
reference_data.to_csv("../data/aes_reference_data.tsv", sep="\t", index=False)
