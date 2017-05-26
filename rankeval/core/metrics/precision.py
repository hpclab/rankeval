import numpy as np
from rankeval.core.metrics.base import query_scores_and_labels_iter


def precision_at_k(scores, labels, qid_offsets, cutoff=10):
    """
    We calculate precision as follows:
        - we sort scores in descending order
        - we take topk
        - we look for scores !=0 (considered relevant)
        - we calculate precision as: number_of_relevant / number_of_items_in_topk (<= cutoff)
        - we return the averaged result

    :param scores: np array with scores for all queries
    :param labels: np array with the labels for all queries
    :param qid_offsets: np array with the qid offsets
    :param cutoff: int, for top k
    :return: float, precision@k
    """
    assert cutoff >= 1
    precision = list()

    for query_scores, query_labels in query_scores_and_labels_iter(scores, labels, qid_offsets):

        q_s_sorted = np.argsort(query_scores, axis=0)[::-1]
        top_k_scores = query_scores[q_s_sorted][:cutoff]
        print top_k_scores
        relevant_scores = float(sum(x > 0 for x in top_k_scores))
        print relevant_scores
        precision.append(relevant_scores/len(top_k_scores))

    return np.mean(precision)


# def average_precision(r):
#
#     r = np.asarray(r) != 0
#     out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
#     if not out:
#         return 0.
#     return np.mean(out)


# https://en.wikipedia.org/wiki/Precision_and_recall

