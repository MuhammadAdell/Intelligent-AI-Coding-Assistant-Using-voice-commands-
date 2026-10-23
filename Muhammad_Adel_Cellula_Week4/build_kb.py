from pathlib import Path
from chromadb import PersistentClient
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
COLLECTION = "coding_knowledge"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    separators=["\n## ", "\n```", "\n\n", "\n", ". ", " "],
    chunk_size=1000,
    chunk_overlap=150,
)


def load_documents():
    docs = []
    for file in sorted(DATA_DIR.glob("*.md")):
        docs.append((file.name, file.read_text(encoding="utf-8")))
    return docs


def build():
    client = PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    total = 0
    for name, text in load_documents():
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"{name[:-3]}_{i}"],  # file name without .md
                embeddings=embeddings.embed_documents([chunk]),
                documents=[chunk],
                metadatas=[{"source": name, "chunk": i}],
            )
            total += 1
        print(f"{name}: {len(chunks)} chunks")
    print(f"Stored {total} chunks in {DB_DIR}/")


if __name__ == "__main__":
    build()
