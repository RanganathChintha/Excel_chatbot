from langchain_groq import ChatGroq
from langsmith import traceable

from config import GROQ_API_KEY, SCOUT_MODEL
from utils.tracing import string_output, text_input


class ScoutModel:
    def __init__(self):
        self.client = ChatGroq(
            model=SCOUT_MODEL,
            api_key=GROQ_API_KEY,
            max_tokens=512,
            timeout=60,
        )

    @traceable(
        name="Scout Preprocess Chunk",
        run_type="llm",
        metadata={"model": SCOUT_MODEL},
        process_inputs=text_input,
        process_outputs=string_output,
    )
    def process(self, text: str) -> str:
        """Process text using Scout model."""
        prompt = (
            "Clean and preserve the useful information from this spreadsheet "
            "chunk for retrieval. Keep names, numbers, dates, and relationships. "
            "Do not add facts.\n\n"
            f"{text}"
        )
        message = self.client.invoke(prompt)
        return message.content
