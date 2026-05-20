from pathlib import Path
from typing import Any, Dict, Iterable, List


def preview_text(text: str, limit: int = 500) -> str:
    """Keep LangSmith payloads readable for large spreadsheet chunks."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"


def summarize_texts(texts: Iterable[str]) -> List[Dict[str, Any]]:
    return [
        {
            "index": index,
            "characters": len(text),
            "preview": preview_text(text, 200),
        }
        for index, text in enumerate(texts)
    ]


def summarize_embeddings(embeddings: List[List[float]]) -> Dict[str, int]:
    return {
        "count": len(embeddings),
        "dimension": len(embeddings[0]) if embeddings else 0,
    }


def ingest_inputs(inputs: dict) -> dict:
    file_path = inputs.get("excel_file_path") or inputs.get("file_path", "")
    path = Path(file_path) if file_path else None
    return {
        "file_path": file_path,
        "file_name": path.name if path else "",
    }


def text_input(inputs: dict) -> dict:
    text = inputs.get("text", "")
    return {
        "characters": len(text),
        "preview": preview_text(text),
    }


def texts_input(inputs: dict) -> dict:
    texts = inputs.get("texts", [])
    return {
        "count": len(texts),
        "texts": summarize_texts(texts[:5]),
    }


def indexed_documents_input(inputs: dict) -> dict:
    embeddings = inputs.get("embeddings", [])
    documents = inputs.get("documents", [])
    ids = inputs.get("ids", [])
    return {
        **summarize_embeddings(embeddings),
        "ids": ids[:5],
        "document_count": len(documents),
        "documents": summarize_texts(documents[:5]),
    }


def retrieval_inputs(inputs: dict) -> dict:
    return {
        "k": inputs.get("k") or inputs.get("n_results", 5),
        "query_embedding_dimension": len(inputs.get("query_embedding", [])),
    }


def retrieval_outputs(outputs: Any) -> dict:
    documents, distances = outputs
    return {
        "documents": summarize_texts(documents),
        "distances": distances,
    }


def string_output(outputs: str) -> dict:
    return {
        "characters": len(outputs),
        "preview": preview_text(outputs),
    }


def strings_output(outputs: List[str]) -> dict:
    return {
        "count": len(outputs),
        "chunks": summarize_texts(outputs[:5]),
    }


def embeddings_output(outputs: List[List[float]]) -> dict:
    return summarize_embeddings(outputs)


def records_output(outputs: List[Dict[str, Any]]) -> dict:
    return {
        "records": len(outputs),
        "columns": list(outputs[0].keys()) if outputs else [],
    }


def llm_inputs(inputs: dict) -> dict:
    context = inputs.get("context", "")
    query = inputs.get("query", "")
    return {
        "query": query,
        "context_characters": len(context),
        "context_preview": preview_text(context),
    }
