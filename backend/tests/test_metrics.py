from evaluation.metrics import calculate_mrr, calculate_ndcg


def test_mrr_uses_first_relevant_rank():
    assert calculate_mrr({"job_2"}, ["job_1", "job_2"]) == 0.5


def test_ndcg_penalises_missing_postings():
    assert calculate_ndcg({"job_1", "job_9"}, ["job_1"]) < 1.0