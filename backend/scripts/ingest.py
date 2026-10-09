"""Build the Chroma collection from data/documents.jsonl.

Run from the backend/ folder:
    uv run python -m scripts.ingest            # build (skips if it already exists)
    uv run python -m scripts.ingest --rebuild  # delete and rebuild
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from chromadb import PersistentClient
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

load_dotenv(override=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

BACKEND_DIR = Path(__file__).parent.parent

# Same environment variables (and defaults) as app/rag/query.py, so the API
# always reads the collection this script builds.
DB_NAME = os.getenv("CHROMA_PATH", str(BACKEND_DIR / "job_postings_db"))
collection_name = os.getenv("CHROMA_COLLECTION", "docs")

JOB_POSTINGS_PATH = Path(os.getenv("DOCUMENTS_PATH", BACKEND_DIR / "data" / "documents.jsonl"))

# Chroma rejects a single add() above its max batch size (a few thousand
# records), so embeddings are computed and written in batches.
BATCH_SIZE = 500

# all-mpnet-base-v2 truncates silently past 384 tokens, so we target ~330
# tokens/chunk (buffer for the title/company/location prefix added to every
# chunk) with a bit of overlap so content spanning a chunk boundary isn't lost.
MAX_CHUNK_TOKENS = 330
CHUNK_OVERLAP_TOKENS = 40

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Recursively split text on the first separator that yields chunks
    within max_tokens, falling back to progressively finer separators."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text, seps):
        if count_tokens(text) <= max_tokens:
            return [text]
        if not seps:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [
                tokenizer.decode(ids[i : i + max_tokens])
                for i in range(0, len(ids), max_tokens - overlap)
            ]

        sep, rest_seps = seps[0], seps[1:]
        pieces = [p for p in text.split(sep) if p.strip()]
        if len(pieces) == 1:
            return _split(text, rest_seps)

        chunks, current = [], ""
        for piece in pieces:
            candidate = (current + sep + piece) if current else piece
            if count_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if count_tokens(piece) > max_tokens:
                    chunks.extend(_split(piece, rest_seps))
                    current = ""
                else:
                    current = piece
        if current:
            chunks.append(current)

        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for prev, cur in zip(chunks, chunks[1:]):
                prev_ids = tokenizer.encode(prev, add_special_tokens=False)
                tail = tokenizer.decode(prev_ids[-overlap:])
                overlapped.append(tail + " " + cur)
            chunks = overlapped

        return chunks

    return _split(text, separators)


def sanitize_metadata(meta: dict) -> dict:
    """Chroma's metadata store only accepts str/int/float/bool -- no None,
    no nested dicts/lists. Some postings have missing fields (None) which
    crashes collection.add() otherwise."""
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ""  # keep the key present (consistent schema for filtering), just empty
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)  # coerce anything unexpected (list/dict) to a string
    return clean


def fetch_documents():
    """Load job postings from JSONL."""
    documents = []
    if not JOB_POSTINGS_PATH.exists():
        raise SystemExit(f"Documents file not found: {JOB_POSTINGS_PATH}")
    with open(JOB_POSTINGS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            documents.append(json.loads(line))
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_document(record: dict) -> list[dict]:
    meta = sanitize_metadata(record.get("metadata", {}))
    title = meta.get("title") or record["text"].split("\n", 1)[0].replace("Title: ", "")
    company = meta.get("company", "")
    location = meta.get("location", "")

    prefix_parts = [f"Title: {title}"]
    if company:
        prefix_parts.append(f"Company: {company}")
    if location:
        prefix_parts.append(f"Location: {location}")
    prefix = "\n".join(prefix_parts) + "\n\n"

    body = record["text"]
    # the raw text already starts with "Title: ..." as its own line -- strip
    # it so it isn't duplicated (the prefix above re-adds it to every chunk)
    if body.startswith("Title:") and "\n" in body:
        body = body.split("\n", 1)[1]

    budget = MAX_CHUNK_TOKENS - count_tokens(prefix)
    raw_chunks = split_text(body, max_tokens=budget, overlap=CHUNK_OVERLAP_TOKENS)

    out = []
    for i, chunk_text in enumerate(raw_chunks):
        full_text = prefix + chunk_text.strip()
        out.append(
            {
                "id": f"{record['id']}_chunk{i}",
                "text": full_text,
                "metadata": {
                    # job_id stays the integer from the source metadata
                    # (e.g. 205003); doc_id is the document id ("job_205003").
                    **meta,
                    "doc_id": record["id"],
                    "chunk_index": i,
                    "chunk_count": len(raw_chunks),
                    "token_count": count_tokens(full_text),
                },
            }
        )
    return out


def create_chunks(documents):
    chunks = []
    for record in tqdm(documents, desc="Chunking"):
        chunks.extend(chunk_document(record))
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_embeddings(chunks, force_rebuild: bool = False):
    """
    Build the Chroma collection, or load the existing one if it already has
    data -- avoids re-running the (slow, CPU-bound) embedding step every time
    this script is executed.

    Pass force_rebuild=True to always delete and recreate (e.g. after
    changing chunking parameters or updating the source data).
    """
    chroma = PersistentClient(path=DB_NAME)
    existing_collections = [c.name for c in chroma.list_collections()]

    if collection_name in existing_collections and not force_rebuild:
        collection = chroma.get_collection(collection_name)
        if collection.count() > 0:
            print(
                f"Found existing vectorstore with {collection.count()} documents "
                f"-- loading, not recreating. Pass force_rebuild=True to rebuild."
            )
            return collection
        # collection exists but is empty (e.g. a failed previous run) -- clean up and rebuild
        chroma.delete_collection(collection_name)
    elif collection_name in existing_collections and force_rebuild:
        chroma.delete_collection(collection_name)

    print("Creating new vectorstore...")
    collection = chroma.get_or_create_collection(collection_name)

    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings.embed_documents(texts),
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
        )
    print(f"Vectorstore created with {collection.count()} documents")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Chroma collection of job postings")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the collection")
    args = parser.parse_args()

    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks, force_rebuild=args.rebuild)
    print("Ingestion complete")