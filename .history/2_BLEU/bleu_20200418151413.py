# # BLEU Implementation

import collections
import math
import numpy as np

def get_ngrams(segment, max_order=4):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1

    return ngram_counts

def best_match_length(reference, candidate):
    ref_length_list = []
    for ref in reference:
        ref_length_list.append(len(ref))
    cand_length_list = [len(candidate)]*len(ref_length_list)
    difference = ((np.asarray(ref_length_list) - np.asarray(cand_length_list)))
    if 0 in difference:
        return ref_length_list[np.argmin(difference)]
    else:
        final = []
        final.append(x for x in difference if x<0)
        if len(final)==0:
            return len(candidate)
        else:
            final = np.asarray(final)
            return ref_length_list[np.where(difference = -1*(final[np.argmax(final)]))][0]
            

def modified_precision(reference, candidate, order=4):
    candidate_counts = get_ngrams(candidate, order)
    
    max_counts = {}
    
    for ref in reference:
        ref_counts = get_ngrams(ref, order)
        
        for ngrams in candidate_counts:
            max_counts[ngrams] = max(max_counts.get(ngrams, 0), ref_counts[ngrams])
    
    clipped_counts = {
        ngram: min(count, max_counts[ngram]) for ngram, count in candidate_counts.items()
    }
    numer = sum(clipped_counts.values())
    denom = max(1, sum(candidate_counts.values()))
    
    return numer/denom

def BP(r, c):
    if c>r:
        return 1
    elif c == 0:
        return 0
    else:
        return math.exp(1-(r/c))


def BLEU(reference, candidate, order=4):
    precision = np.zeros((1, order))
    p_log_sum = 0
    
    no_references = len(reference)
    candidate_length = len(candidate)
    for i in range(order):
        precision[0][i] = modified_precision(reference, candidate, i+1)
    
    r = best_match_length(reference, candidate)
    c = candidate_length
    
    bp = BP(r,c)
    
    weight = 1/order
    
    if (np.min(precision)>0):
        for i in range(order):
            p_log_sum += (weight * math.log(precision[0][i]))
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    
    bleu = bp*geo_mean
    
    return bleu
