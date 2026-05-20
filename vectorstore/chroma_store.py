from typing import List

from langchain_chroma import Chroma
from langsmith import traceable

from config import CHROMA_DB_PATH
from utils.tracing import indexed_documents_input, retrieval_inputs


class ChromaVectorStore:
    def __init__(self):
        self.store = Chroma(
            collection_name="rag_collection",
            persist_directory=CHROMA_DB_PATH,
        )

    @traceable(
        name="Add Chroma Documents",
        run_type="retriever",
        process_inputs=indexed_documents_input,
        process_outputs=lambda output: {"status": "upserted", "path": CHROMA_DB_PATH},
    )
    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str]):
        """Add embeddings to vector store."""
        self.store._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
        )

    @traceable(
        name="Query Chroma Documents",
        run_type="retriever",
        process_inputs=retrieval_inputs,
    )
    def query(self, query_embedding: List[float], n_results: int = 5) -> dict:
        """Query vector store."""
        return self.store._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    @traceable(
        name="Persist Chroma",
        run_type="tool",
        process_outputs=lambda output: {"path": CHROMA_DB_PATH},
    )
    def persist(self):
        """Persist vector store."""
        persist = getattr(self.store, "persist", None)
        if persist is not None:
            persist()
