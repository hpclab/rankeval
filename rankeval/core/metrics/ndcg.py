import math
import numpy as np


"""

        - we sort scores in descending order
q_s_sorted = np.argsort(query_scores, axis=0)[::-1]
top_k_scores = query_scores[q_s_sorted][:cutoff]

IMPL:
[1]burges - exp
[0]jarvelin - flat 

"""


def ndcg(scores, labels, qid_offsets, cutoff=10, no_relevant_results='Yahoo', ties=True, impl="tois2"):
    no_relevant_results_score = 0.0

    if (no_relevant_results == 'Yahoo'):
        no_relevant_results_score = 0.5

    # aggiungere NDCG di HPCKit
    assert len(scores) == labels.size()

    if cutoff == None: topk = -1

    ndcg_values = np.zeros(labels.size())
    ii = 0

    for scores, example in zip(scores, labels.examples()):
        labels = example['y']
        assert len(scores) == len(labels)

        ranked_i = sorted(range(len(scores)), key=lambda k: float(scores[k]), reverse=True)

        ndcg = sum([(2.0 ** labels[i] - 1.0) / (math.log(float(k) + 2.0, 2))
                    for k, i in enumerate(ranked_i[:topk])])
        i_ndcg = sum([(2.0 ** float(l) - 1.0) / (math.log(float(k) + 2.0, 2))
                      for k, l in enumerate(sorted(labels, reverse=True)[:topk])])

        if i_ndcg != 0:
            ndcg_values[ii] = ndcg / i_ndcg
        else:
            ndcg_values[ii] = 0.0
        ii += 1

    ndcg = np.average(ndcg_values)
    return ndcg



def ndcg_perquery(scores, labels, q_index_delim, cutoff = 10, no_relevant_results = 'Yahoo'):

    no_relevant_results_score = 0.0
    if (no_relevant_results == 'Yahoo'):
        no_relevant_results_score = 0.5

    query_scores = scores[q_index_delim[0]:q_index_delim[1]]
    query_labels = scores[q_index_delim[0]:q_index_delim[1]]
    assert len(query_scores) == len(query_labels)

    if cutoff is None:
        topk=-1

    ndcg_values = np.zeros(labels.size())
    ii = 0

    for scores, example in zip(scores, labels.examples()):
        labels = example['y']
        assert len(scores) == len(labels)

        ranked_i = sorted(range(len(scores)), key=lambda k: float(scores[k]), reverse=True)

        ndcg = sum([(2.0 ** labels[i] - 1.0) / (math.log(float(k) + 2.0, 2))
                    for k, i in enumerate(ranked_i[:topk])])
        i_ndcg = sum([(2.0 ** float(l) - 1.0) / (math.log(float(k) + 2.0, 2))
                      for k, l in enumerate(sorted(labels, reverse=True)[:topk])])

        if i_ndcg != 0:
            ndcg_values[ii] = ndcg / i_ndcg
        else:
            ndcg_values[ii] = 0.0
        ii += 1


def ndcg_tied(scores, labels, q_lens, cutoff = 10, no_relevant_results = 'Yahoo'):
    pass

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


#https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
