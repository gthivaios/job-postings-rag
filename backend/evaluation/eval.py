import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

from app.rag.query import answer_question, fetch_context
from evaluation.metrics import calculate_mrr, calculate_ndcg, retrieved_job_ids
from evaluation.test import TestQuestion, load_tests

REPORT_FILE = Path(__file__).parent / "report.json"


load_dotenv(override=True)

#MODEL = "openrouter/openai/gpt-oss-120b"
MODEL = "openrouter/openai/gpt-4.1-mini"


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance, scored on job ids."""

    mrr: float = Field(description="Reciprocal rank of the first relevant posting retrieved")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    relevant_found: int = Field(description="Number of relevant postings found in top-k results")
    total_relevant: int = Field(description="Total number of relevant postings to find")
    coverage: float = Field(description="Percentage of relevant postings found (recall@k)")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the expected answer description"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the expected answer description? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - everything the expected answer description requires is provided). Only answer 5 if ALL items required by the expected answer description are included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )

def score_retrieval(test: TestQuestion, retrieved_docs: list, k: int = 10) -> RetrievalEval | None:
    """Score already-retrieved chunks, so a full run needs only one RAG call per test."""
    relevant = set(test.relevant_job_ids)
    if not relevant:
        return None
    ranked_ids = retrieved_job_ids(retrieved_docs)[:k]
    found = sum(1 for doc_id in ranked_ids if doc_id in relevant)
    return RetrievalEval(
        mrr=calculate_mrr(relevant, ranked_ids),
        ndcg=calculate_ndcg(relevant, ranked_ids, k),
        relevant_found=found,
        total_relevant=len(relevant),
        coverage=found / len(relevant) * 100,
    )


def mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def run_all(output: Path = REPORT_FILE) -> dict:
    results = []
    for test in load_tests():
        answer_eval, answer, docs = evaluate_answer(test)
        retrieval = score_retrieval(test, docs)
        results.append({
            "id": test.id,
            "question": test.question,
            "answer": answer,
            "mrr": retrieval.mrr if retrieval else None,
            "ndcg": retrieval.ndcg if retrieval else None,
            "coverage": retrieval.coverage if retrieval else None,
            "accuracy": answer_eval.accuracy,
            "completeness": answer_eval.completeness,
            "relevance": answer_eval.relevance,
            "feedback": answer_eval.feedback,
        })
        print(f"{test.id}: accuracy={answer_eval.accuracy}")

    metrics = ["mrr", "ndcg", "coverage", "accuracy", "completeness", "relevance"]
    report = {
        "date": datetime.now(tz=UTC).date().isoformat(),
        "judge_model": MODEL,
        "tests": len(results),
        "summary": {m: mean([r[m] for r in results]) for m in metrics},
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report

def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval | None:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and relevant_job_ids
        k: Number of top postings to score (default 10)

    Returns:
        RetrievalEval with MRR, nDCG and coverage, or None for negative cases
        (no relevant postings), where retrieval metrics are undefined.
    """
    relevant = set(test.relevant_job_ids)
    if not relevant:
        return None

    retrieved_docs = fetch_context(test.question)
    ranked_ids = retrieved_job_ids(retrieved_docs)[:k]

    relevant_found = sum(1 for doc_id in ranked_ids if doc_id in relevant)
    total_relevant = len(relevant)

    return RetrievalEval(
        mrr=calculate_mrr(relevant, ranked_ids),
        ndcg=calculate_ndcg(relevant, ranked_ids, k),
        relevant_found=relevant_found,
        total_relevant=total_relevant,
        coverage=relevant_found / total_relevant * 100,
    )


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and expected answer description

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    generated_answer, retrieved_docs = answer_question(test.question)

    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers from a job postings assistant. Evaluate the generated answer against the expected answer description. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Expected Answer (a description of what a correct answer must contain, not a model answer to match word for word):
{test.expected}

Please evaluate the generated answer on three dimensions:
1. Accuracy: Is it factually consistent with the expected answer description? If the description says no matching jobs exist, an answer that presents jobs as matches is wrong. Only give 5/5 scores for perfect answers.
2. Completeness: Does it cover every posting, company and fact the expected answer description requires? Items the description marks as optional ("may be mentioned", "may be flagged") are not required.
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)
    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs


def evaluate_all_retrieval():
    """Evaluate all retrieval tests, skipping negative cases (no relevant postings)."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        if result is None:
            continue
        yield test, result, progress


def evaluate_all_answers():
    """Evaluate the answers to all tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_answer(test)[0]
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test."""
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    test = tests[test_number]

    print(f"\n{'=' * 80}")
    print(f"Test #{test_number} ({test.id})")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Relevant job ids: {test.relevant_job_ids}")
    print(f"Expected: {test.expected}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    if retrieval_result is None:
        print("Negative case (no relevant postings) - retrieval metrics skipped")
    else:
        print(f"MRR: {retrieval_result.mrr:.4f}")
        print(f"nDCG: {retrieval_result.ndcg:.4f}")
        print(
            f"Relevant Found: {retrieval_result.relevant_found}/{retrieval_result.total_relevant}"
        )
        print(f"Coverage: {retrieval_result.coverage:.1f}%")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nRetrieved job ids: {retrieved_job_ids(retrieved_docs)}")
    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--all", action="store_true", help="Run every test and write report.json")
    parser.add_argument("--test", type=int, help="Run one test by row number")
    args = parser.parse_args()
    if args.all:
        report = run_all()
        print(json.dumps(report["summary"], indent=2))
    elif args.test is not None:
        run_cli_evaluation(args.test)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()