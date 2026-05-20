import faiss
import numpy as np
from config import FAISS_INDEX_PATH
from typing import List, Tuple

from langsmith import traceable

from utils.tracing import indexed_documents_input, retrieval_inputs, retrieval_outputs


class FAISSRetriever:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    @traceable(
        name="Add FAISS Documents",
        run_type="retriever",
        process_inputs=indexed_documents_input,
        process_outputs=lambda output: {"status": "added"},
    )
    def add(self, embeddings: List[List[float]], documents: List[str]):
        """Add embeddings to FAISS index."""
        embeddings_array = np.array(embeddings).astype("float32")
        self.index.add(embeddings_array)
        self.documents.extend(documents)

    @traceable(
        name="Retrieve FAISS Documents",
        run_type="retriever",
        process_inputs=retrieval_inputs,
        process_outputs=retrieval_outputs,
    )
    def retrieve(self, query_embedding: List[float], k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top k documents."""
        query_array = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_array, k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs, distances[0].tolist()

    @traceable(
        name="Save FAISS Index",
        run_type="tool",
        process_outputs=lambda output: {"path": FAISS_INDEX_PATH},
    )
    def save(self):
        """Save FAISS index."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)

    @traceable(
        name="Load FAISS Index",
        run_type="tool",
        process_outputs=lambda output: {"path": FAISS_INDEX_PATH},
    )
    def load(self):
        """Load FAISS index."""
        self.index = faiss.read_index(FAISS_INDEX_PATH)
