import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REPORT_FILE = Path(__file__).resolve().parents[1] / "evaluation" / "report.json"
from app.rag.query import answer_question
from app.schemas import ChatRequest, ChatResponse, Source

app = FastAPI(title="Job Postings RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = [m.model_dump() for m in req.history]
    try:
        answer, chunks = answer_question(req.question, history)
    except Exception as exc:
        raise HTTPException(502, f"The model call failed: {exc}") from exc
    sources = [
        Source(
            job_id=c.metadata.get("job_id"),
            portal=c.metadata.get("portal"),
            company=c.metadata.get("company"),
            location=c.metadata.get("location"),
            country=c.metadata.get("country"),
            date=c.metadata.get("date"),
            text=c.page_content,
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)

@app.get("/eval/report")
def eval_report():
    if not REPORT_FILE.exists():
        raise HTTPException(404, "No report yet. Run: uv run python -m evaluation.eval --all")
    return json.loads(REPORT_FILE.read_text(encoding="utf-8"))