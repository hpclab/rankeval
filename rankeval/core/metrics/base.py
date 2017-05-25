"""

"""

def query_scores_and_labels_iter(scores, labels, q_lens):
    assert len(scores) == len(labels)
    for i, qid_offset in enumerate(q_lens[:-1]):
        query_scores = scores[qid_offset:q_lens[i+1]]
        query_labels = labels[qid_offset:q_lens[i+1]]
        assert len(query_scores) == len(query_labels)
        yield query_scores, query_labels
