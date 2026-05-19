from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

class TextSplitter:
    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)