from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from langsmith import traceable

from utils.tracing import strings_output, text_input


class TextSplitter:
    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @traceable(
        name="Split Text",
        run_type="tool",
        process_inputs=text_input,
        process_outputs=strings_output,
    )
    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.splitter.split_text(text)
