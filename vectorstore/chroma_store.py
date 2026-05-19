import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH
from typing import List

class ChromaVectorStore:
    def __init__(self):
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb_parquet",
                persist_directory=CHROMA_DB_PATH,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="rag_collection"
        )

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str]):
        """Add embeddings to vector store."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
        )

    def query(self, query_embedding: List[float], n_results: int = 5) -> dict:
        """Query vector store."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def persist(self):
        """Persist vector store."""
        self.client.persist()