import math


def retrieved_job_ids(retrieved_docs: list) -> list[str]:
    """
    Turn retrieved chunks into a ranked list of unique job ids.

    Chunk metadata stores job_id as an integer (205837) while the golden set
    uses the document id format ("job_205837"), so ids are normalised here.
    Several chunks can belong to the same posting; each posting is counted
    once, at the rank of its best (first) chunk.
    """
    ids: list[str] = []
    for doc in retrieved_docs:
        job_id = doc.metadata.get("job_id")
        if job_id is None:
            continue
        doc_id = str(job_id)
        if not doc_id.startswith("job_"):
            doc_id = f"job_{doc_id}"
        if doc_id not in ids:
            ids.append(doc_id)
    return ids


def calculate_mrr(relevant: set[str], ranked_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant posting (0 if none retrieved)."""
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(relevant: set[str], ranked_ids: list[str], k: int = 10) -> float:
    """Calculate nDCG with binary relevance over job ids."""
    relevances = [1 if doc_id in relevant else 0 for doc_id in ranked_ids[:k]]
    dcg = calculate_dcg(relevances, k)

    # Ideal ranking: every relevant posting at the top (capped at k). Using the
    # full relevant set, not just what was retrieved, means missing postings
    # lower the score.
    ideal_relevances = [1] * min(len(relevant), k)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0