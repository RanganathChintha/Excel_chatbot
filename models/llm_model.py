from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL


class LLMModel:
    def __init__(self):
        self.client = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0,
            max_tokens=1024,
            timeout=60,
        )

    def generate(self, context: str, query: str) -> str:
        """Generate response using LLM model with retrieved context."""
        prompt = f"""
You are a strict file-grounded assistant.

Rules:
- Answer only using the provided file context.
- Do not use outside knowledge.
- Do not guess, infer beyond the context, or fill missing details.
- If the context does not contain the answer, say: "I don't know based on the provided file."
- Keep the answer concise and mention the relevant values from the context when possible.

File context:
{context}

Question:
{query}

Answer:
""".strip()
        message = self.client.invoke(prompt)
        return message.content
