import json
from pathlib import Path

from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "golden_set.jsonl")


class TestQuestion(BaseModel):
    """A golden-set question with the relevant postings and expected answer."""

    id: str = Field(description="Unique identifier of the test case")
    question: str = Field(description="The question to ask the RAG system")
    relevant_job_ids: list[str] = Field(
        default_factory=list,
        description="Document ids (e.g. 'job_205837') that should be retrieved; empty for negative cases",
    )
    expected: str = Field(description="Description of what a correct answer should contain")


def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # tolerate blank lines / trailing newline
            tests.append(TestQuestion(**json.loads(line)))
    return tests