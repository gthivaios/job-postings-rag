from app.rag.query import collection, fetch_context

QUESTION = "nurse jobs in Athens"
EXPECTED = [205661, 205810]  # from the golden set

print("Chunks in collection:", collection.count())

found = collection.get(where={"job_id": {"$in": EXPECTED}}, include=["metadatas"])
print("Expected postings present:", sorted({m["job_id"] for m in found["metadatas"]}))

print("\nRetrieved for:", QUESTION)
for c in fetch_context(QUESTION):
    title = c.page_content.split("\n", 1)[0]
    print(f"  {c.metadata.get('job_id')} | {c.metadata.get('location')} | {title}")