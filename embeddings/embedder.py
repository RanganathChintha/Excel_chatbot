from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY, EMBEDDER_MODEL
from typing import List

class Embedder:
    def __init__(self):
        self.client = InferenceClient(api_key=HUGGINGFACE_API_KEY)
        self.model = EMBEDDER_MODEL

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.feature_extraction(
            text=text,
            model=self.model,
        )
        return response

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings