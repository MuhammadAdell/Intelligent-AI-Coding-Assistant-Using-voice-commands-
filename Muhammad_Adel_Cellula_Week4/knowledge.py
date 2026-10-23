import threading
from chromadb import PersistentClient
from build_kb import COLLECTION, DB_DIR, embeddings, splitter
from llm import get_llm
from prompts import EVALUATOR_PROMPT

TOP_K = 3

_lock = threading.Lock()
_client = None
_collection = None


def get_collection():
    global _client, _collection
    with _lock:
        if _collection is None:
            _client = PersistentClient(path=str(DB_DIR))
            _collection = _client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return _collection


def retrieve(query, k=TOP_K):
    vector = embeddings.embed_query(query)
    hits = get_collection().query(
        query_embeddings=[vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return hits["documents"][0], hits["metadatas"][0], hits["distances"][0]


def evaluate_relevance(query, context):
    llm = get_llm(temperature=0)
    prompt = EVALUATOR_PROMPT.format(context=context, query=query)
    reply = llm.invoke(prompt).content.strip()
    return "not relevant" not in reply.lower()


def add_solution(text, source="user_feedback"):
    collection = get_collection()
    chunks = splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        collection.upsert(
            ids=[f"feedback_{source}_{i}"],
            embeddings=embeddings.embed_documents([chunk]),
            documents=[chunk],
            metadatas=[{"source": source, "chunk": i}],
        )
    return len(chunks)
