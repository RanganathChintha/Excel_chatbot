from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDER_MODEL


class Embedder:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDER_MODEL}", flush=True)
        self.model = SentenceTransformer(EMBEDDER_MODEL)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype("float32").tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        total = len(texts)
        print(f"Embedding {total} chunks locally...", flush=True)
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return embeddings.astype("float32").tolist()
