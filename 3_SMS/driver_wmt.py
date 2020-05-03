import pandas as pd
import time
from emd_metrics import EMDMetrics
import sys
import glob


# MODELS = ["glove", "elmo", "bert"]
# METHODS = ["wms", "sms", "s+wms"]


def get_scores(CANDIDATES_FILES, REFERENCES_FILE, MODEL, METHOD):

    # print("Reading references from:", REFERENCES_FILE)
    with open(REFERENCES_FILE) as f:
        reference_texts = f.read().strip().split("\n")
    for candidates_file in CANDIDATES_FILES:
        print("Running against the candidates from:", candidates_file)
        with open(candidates_file) as f:
            candidate_texts = f.read().strip().split("\n")
        cand_model = candidates_file.rsplit("/", 1)[1]

        calculator = EMDMetrics(MODEL, ignore_stopwords=False)

        st = time.time()
        method_results = []
        for cand, ref in zip(candidate_texts, reference_texts):
            try:
                sim = calculator.get_similarity(cand, ref, METHOD)
                method_results.append(sim)
            except:
                print("Error for:", cand)
        score = sum(method_results)/len(method_results)
        print("Score:", score)
        print("Done in", time.time() - st)
        print("="*25)


if __name__ == "__main__":
    candidates_folder = sys.argv[1]
    CANDIDATES_FILES = glob.glob(candidates_folder + "/*")
    REFERENCES_FILE = sys.argv[2]
    MODEL = sys.argv[3]
    METHOD = sys.argv[4]

    print("candidates_folder", candidates_folder)
    print("REFERENCES_FILE", REFERENCES_FILE)
    print("MODEL", MODEL)
    print("METHOD", METHOD, "\n\n")

    get_scores(CANDIDATES_FILES, REFERENCES_FILE, MODEL, METHOD)
