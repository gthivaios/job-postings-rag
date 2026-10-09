import os
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

load_dotenv(override=True)

# MODEL = "openai/gpt-4.1-nano"
# MODEL = "groq/openai/gpt-oss-120b"
MODEL = "openrouter/openai/gpt-oss-120b"
#MODEL = "openrouter/openai/gpt-4.1-mini"
BACKEND_DIR = Path(__file__).resolve().parents[2]   # ...\job-postings-rag\backend
DB_NAME = os.getenv("CHROMA_PATH", str(BACKEND_DIR / "job_postings_db"))
collection_name = os.getenv("CHROMA_COLLECTION", "docs")

chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_collection(collection_name)
# IMPORTANT: this must match the collection_name you used when building the
# vectorstore in your notebook (Chroma.from_documents(..., collection_name="docs")).
# LangChain's Chroma wrapper defaults to its own collection name if you don't
# pass collection_name explicitly there -- if this script can't find your data,
# that mismatch is the first thing to check.

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
wait = wait_exponential(multiplier=1, min=10, max=240)

#chroma = PersistentClient(path=DB_NAME)
#collection = chroma.get_or_create_collection(collection_name)

# Build the list of countries actually present in the corpus once, at import
# time, rather than hardcoding them -- keeps this working automatically as
# your dataset grows across more countries.
def _get_known_countries():
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["country"] for m in all_meta if m.get("country")})


KNOWN_COUNTRIES = _get_known_countries()


def extract_country_filter(question: str) -> str | None:
    """Detect a country mention in the question and return the exact metadata
    value to filter on, or None if no known country is mentioned."""
    q_lower = question.lower()
    for country in KNOWN_COUNTRIES:
        if country.lower() in q_lower:
            return country
    return None

RETRIEVAL_K = 20
FINAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, precise assistant helping users explore and understand
a database of European job postings.
You are chatting with a user about jobs, employers, requirements, and hiring
trends found in the postings.
Your answer will be evaluated for accuracy, relevance and completeness, so make
sure it only answers the question and fully answers it.
If you don't know the answer, say so -- do not invent details (salary, benefits,
deadlines) that aren't present in the context below.
The context consists of excerpts from job postings; some are full postings,
others are partial chunks of a longer posting. Chunks sharing the same job_id
belong to the same posting.

For context, here are specific extracts from the job postings database that
might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant,
and complete. When you reference a specific posting, mention the job title and
company/location if available, so the user can tell which posting you mean.
"""


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


@retry(wait=wait)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a job postings database.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=MODEL, messages=messages, response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from job posting {chunk.metadata.get('job_id', 'unknown')} "
        f"({chunk.metadata.get('company', 'unknown company')}, "
        f"{chunk.metadata.get('location', 'unknown location')}):\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def rewrite_query(question, history=None):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the job postings database."""
    history = history or []
    message = f"""
You are in a conversation with a user, answering questions about a database of
European job postings (job titles, descriptions, requirements, companies,
locations, and benefits).
You are about to look up information in the job postings database to answer
the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the
job postings database.
It should be a VERY short specific question most likely to surface relevant
postings. Focus on the question details (e.g. job title, skills, location,
industry, seniority) rather than conversational phrasing.
IMPORTANT: Respond ONLY with the precise search query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question, where=None):
    query = embeddings.embed_query(question)
    results = collection.query(
        query_embeddings=[query], n_results=RETRIEVAL_K, where=where
    )
    chunks = []
    for result in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append(Result(page_content=result[0], metadata=result[1]))
    return chunks


def fetch_context(original_question):
    # detect a country mention up front, before rewriting -- the rewrite step
    # may compress the question in a way that drops the location, so we pull
    # it from the original wording
    country = extract_country_filter(original_question)
    where = {"country": country} if country else None
    if country:
        print(f"Filtering results to country: {country}")

    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question, where=where)
    chunks2 = fetch_context_unranked(rewritten_question, where=where)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]


@retry(wait=wait)
def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    history = history or []
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks
