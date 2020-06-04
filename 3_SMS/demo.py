import pandas as pd
import time
from emd_metrics import EMDMetrics
from scipy.stats import spearmanr
from texttable import Texttable
import driver_sms


def get_corr(dataframe):
    # Spearman correlation
    scorr = spearmanr(dataframe["similarity"], dataframe["score"])
    corr = "{0:6.3f}".format(scorr.correlation)
    if (scorr.pvalue >= 0.001):
        pval = "{0:6.3f}".format(scorr.pvalue)
    else:
        pval = "{0:10.3e}".format(scorr.pvalue)
    return corr, pval


def process_dataset(dataset, calculator, method):
    results = []
    for idx, sample in enumerate(dataset):
        cand = sample["candidate"]
        ref = sample["reference"]
        sim, dist = calculator.get_similarity_dist(cand, ref, method)
        sample["similarity"] = sim
        sample["distance"] = dist
        results.append(sample)

    df = pd.DataFrame(results)
    corr, pval = get_corr(df)
    return corr, pval


#################
# Main function #
#################
DATASET_FILE = "../data/demo.tsv"
# print("Reading dataset from", DATASET_FILE)
dataset = pd.read_csv(DATASET_FILE, sep="\t").to_dict("records")

# filter this based on the time taken
MODELS = ["glove", "bert", "elmo", "roberta-large"]
METHODS = ["wms", "sms", "s+wms"]

correlations = [["Model", "Method", "Correlation", "P value"]]
for model in MODELS:
    st = time.time()
    calculator = EMDMetrics(model)
    for method in METHODS:
        corr, pval = process_dataset(dataset, calculator, method)
        correlations.append([model, method, corr, pval])
    # print("Time for Model:", model, time.time()-st)

# Running SMS from SentBERT embeddings
sent_bert_results = driver_sms.process_dataset(dataset, None, False)
corr, pval = get_corr(sent_bert_results)
correlations.append(["SentBERT", "sms", corr, pval])

#print("\nCorrelations from various EMD based metrics\n")
t = Texttable()
t.add_rows(correlations)
print(t.draw())
