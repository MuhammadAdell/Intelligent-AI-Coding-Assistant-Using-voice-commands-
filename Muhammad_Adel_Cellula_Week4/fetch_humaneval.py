import json
import urllib.request
from chromadb import PersistentClient
from build_kb import COLLECTION, DB_DIR, embeddings, splitter

BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "openai%2Fopenai_humaneval"
CONFIG = "openai_humaneval"


def fetch_rows():
    rows = []
    offset = 0
    while True:
        url = f"{BASE}?dataset={DATASET}&config={CONFIG}&split=test&offset={offset}&length=100"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        batch = [item["row"] for item in data["rows"]]
        rows.extend(batch)
        if not batch or len(rows) >= data["num_rows_total"]:
            break
        offset += len(batch)
    return rows


def problem_text(row):
    return (
        f"Problem {row['task_id']}, function {row['entry_point']}\n"
        f"Prompt:\n{row['prompt'].rstrip()}\n"
        f"Reference solution:\n{row['prompt'].rstrip()}\n"
        f"{row['canonical_solution'].rstrip()}"
    )


def store(rows):
    client = PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    total = 0
    for row in rows:
        chunks = splitter.split_text(problem_text(row))
        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"{row['task_id'].replace('/', '_')}_chunk_{i}"],
                embeddings=embeddings.embed_documents([chunk]),
                documents=[chunk],
                metadatas=[{
                    "source": "openai_humaneval",
                    "problem": row["task_id"],
                    "entry_point": row["entry_point"],
                    "chunk": i,
                }],
            )
            total += 1
    return total


if __name__ == "__main__":
    rows = fetch_rows()
    chunks = store(rows)
    print(f"Stored {len(rows)} problems as {chunks} chunks in {DB_DIR}/chroma.sqlite3")
