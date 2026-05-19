from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

class LLMModel:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def generate(self, context: str, query: str) -> str:
        """Generate response using LLM model with retrieved context."""
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        message = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return message.content[0].text