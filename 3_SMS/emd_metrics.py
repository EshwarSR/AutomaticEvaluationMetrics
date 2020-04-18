"""Metrics based on Earth Mover's Distance"""

import spacy
from collections import Counter
from wmd import WMD
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from bert_embedding import BertEmbedding
import time
import nltk
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))


class EMDMetrics:
    def __init__(self, model):
        self.model = model
        if self.model == "elmo":
            self.MODEL = ElmoEmbedder()
        elif self.model == "bert":
            self.MODEL = BertEmbedding()
        self.nlp = spacy.load("en_core_web_md")

    def get_sent_embedding(self, word_emb_list):
        word_emb_array = np.array(word_emb_list)
        sent_emb = list(np.mean(word_emb_array, axis=0))
        return sent_emb

    def get_embeddings_ids_weights(self, spacy_doc, next_id, emb, method):
        doc_list = []
        weights = []

        if self.model == "glove":
            sents_list = [sent for sent in nltk.sent_tokenize(spacy_doc.text)]
            # for sent in spacy_doc.sents: # Trying to replicate authors score
            for sent in sents_list:
                sent_list = []
                # for word in sent: # Trying to replicate authors score
                for word in self.nlp(sent):
                    # TODO: Should we consider stop words or no?
                    if word.text.isalpha() and word.text.lower() not in stop_words:
                        sent_list.append(next_id)
                        emb[next_id] = self.nlp.vocab.get_vector(word.text)
                        next_id += 1
                doc_list.append(sent_list)

        elif self.model == "elmo":
            for sent in spacy_doc.sents:
                sent_words_list = []
                sent_ids_list = []
                for word in sent:
                    sent_words_list.append(word.text)
                word_vectors = self.MODEL.embed_sentence(sent_words_list)
                word_vectors = np.average(word_vectors, axis=0)
                for word_idx in range(len(sent_words_list)):
                    sent_ids_list.append(next_id)
                    emb[next_id] = word_vectors[word_idx]
                    next_id += 1
                doc_list.append(sent_ids_list)

        elif self.model == "bert":
            sents = []
            for sent in spacy_doc.sents:
                sents.append(sent.text)
            bert_resp = self.MODEL(sents)
            for words_list, word_vectors in bert_resp:
                sent_ids_list = []
                for word_idx in range(len(words_list)):
                    sent_ids_list.append(next_id)
                    emb[next_id] = word_vectors[word_idx]
                    next_id += 1
                doc_list.append(sent_ids_list)

        if method == "wms":
            # Flattened list
            ids_list = [item for sublist in doc_list for item in sublist]
            weights = [1/(len(ids_list))] * len(ids_list)
            weights = np.array(weights, dtype=np.float32)

        else:
            if method == "sms":
                ids_list = []
                weights = []
            elif method == "w+sms":
                ids_list = [item for sublist in doc_list for item in sublist]
                weights = [1/(len(ids_list))] * len(ids_list)

            for sent in doc_list:
                word_embs = []
                for word_id in sent:
                    word_embs.append(emb[word_id])
                sent_emb = self.get_sent_embedding(word_embs)
                ids_list.append(next_id)
                emb[next_id] = sent_emb
                next_id += 1

                weights.append(len(sent) / len(spacy_doc))

            weights = np.array(weights, dtype=np.float32)

            if method == "w+sms":
                weights = weights / 2

        return next_id, emb, ids_list, weights

    def get_emb_nbow(self, candidate, reference, method):
        can_doc = self.nlp(candidate)
        ref_doc = self.nlp(reference)

        emb = {}
        next_id = 1

        next_id, emb, can_id_list, can_weights = self.get_embeddings_ids_weights(
            can_doc, next_id, emb, method)
        next_id, emb, ref_id_list, ref_weights = self.get_embeddings_ids_weights(
            ref_doc, next_id, emb, method)

        nbow = {
            "reference": ("reference", ref_id_list, ref_weights),
            "hypothesis": ("reference", can_id_list, can_weights)
        }

        return emb, nbow

    # Driver function
    def get_similarity(self, candidate, reference, method):
        emb, nbow = self.get_emb_nbow(candidate, reference, method)
        calc = WMD(emb, nbow, vocabulary_min=1)
        dist = calc.nearest_neighbors("reference", k=1, early_stop=1)[
            0][1]
        similarity = np.exp(-dist)
        return similarity
