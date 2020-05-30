from collections import Counter
from wmd import WMD
import numpy as np
import time
from transformers import AutoModel, AutoTokenizer
import torch

class EMDMetrics:
	def __init__(self, model = 'roberta-large', layers = 17, ignore_stopwords=True):
		self.model = model
		self.num_layers = layers

	def sent_encode(self, one_sent):
		one_sent = one_sent.strip()
		return torch.tensor([self.tok.encode(one_sent)])[0]

	def get_roberta_emb(self, tokenized_sent):
		model.eval()
		with torch.no_grad():
			out = self.MODEL(tokenized_sent)
			emb = out[0]
			# emb = out[-1]
		return emb

	def get_sent_embedding(self, word_emb_list):
		word_emb_array = np.array(word_emb_list)
		sent_emb = list(np.mean(word_emb_array, axis=0))
		return sent_emb

	def get_embeddings_ids_weights(self, doc, next_id, emb, method):
		doc_list = []
		weights  = []
		words    = []

    sents_list = [sent for sent in nltk.sent_tokenize(
            spacy_doc.text)]

		if self.model == "roberta-large":
			for sent in doc:
				encoded_sent = sent_encode(sent)
				emb = get_roberta_emb(encoded_sent)

			for words_list, word_vectors in roberta_rep:
				sent_ids_list = []
				for word_idx in range(len(words_list)):
					include = False
					word = words_list[word_idx]
					if self.ignore_stopwords:
						if word.isalpha() and word.lower() not in stop_words:
							include = True
					else:
						include = True
					if include:
						sent_ids_list.append(next_id)
						emb[next_id] = word_vectors[word_idx]
						next_id += 1
				if len(sent_ids_list) > 0:
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
			elif method == "s+wms":
				ids_list = [item for sublist in doc_list for item in sublist]
				# weights = [1/(len(ids_list))] * len(ids_list)
				weights = []
				for id in ids_list:
					weights.append(ids_list.count(id))

			total_toks = 0
			for sent in doc_list:
				word_embs = []
				for word_id in sent:
					word_embs.append(emb[word_id])
				sent_emb = self.get_sent_embedding(word_embs)
				ids_list.append(next_id)
				emb[next_id] = sent_emb
				next_id += 1

				weights.append(len(sent))
				total_toks += len(sent)

			weights = np.array(weights, dtype=np.float32)  # / total_toks

			if method == "sms" or method == "s+wms":
				weights = weights / (total_toks)
				if method == "s+wms":
					weights = weights / 2

		return next_id, emb, ids_list, weights

	def get_emb_nbow(self, candidate, reference, method):
		can_doc = self.nlp(candidate)
		ref_doc = self.nlp(reference)

		emb = {}
		next_id = 1

		next_id, emb, can_id_list, can_weights = self.get_embeddings_ids_weights(can_doc, next_id, emb, method)
		next_id, emb, ref_id_list, ref_weights = self.get_embeddings_ids_weights(ref_doc, next_id, emb, method)

		nbow = {
			"reference": ("reference", ref_id_list, ref_weights),
			"hypothesis": ("reference", can_id_list, can_weights)
		}

		return emb, nbow

	# Driver function
	def get_similarity(self, candidate, reference, method):
		emb, nbow = self.get_emb_nbow(candidate, reference, method)
		calc = WMD(emb, nbow, vocabulary_min=1)
		dist = calc.nearest_neighbors("reference", k=1, early_stop=1)[0][1]
		similarity = np.exp(-dist)
		return similarity
