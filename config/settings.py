import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "excel-chatbot")

SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDER_MODEL = "sentence-transformers/all-minilm-l6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./faiss_index"
