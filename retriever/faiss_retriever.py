import faiss
import numpy as np
from config import FAISS_INDEX_PATH
from typing import List, Tuple

class FAISSRetriever:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []

    def add(self, embeddings: List[List[float]], documents: List[str]):
        """Add embeddings to FAISS index."""
        embeddings_array = np.array(embeddings).astype("float32")
        self.index.add(embeddings_array)
        self.documents.extend(documents)

    def retrieve(self, query_embedding: List[float], k: int = 5) -> Tuple[List[str], List[float]]:
        """Retrieve top k documents."""
        query_array = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_array, k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        return retrieved_docs, distances[0].tolist()

    def save(self):
        """Save FAISS index."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)

    def load(self):
        """Load FAISS index."""
        self.index = faiss.read_index(FAISS_INDEX_PATH)