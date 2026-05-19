# Excel Chatbot

A command-line RAG chatbot for asking questions about spreadsheet files. The app loads a spreadsheet, splits it into chunks, creates local embeddings, stores them in Chroma and FAISS, then answers questions using Groq through LangChain.

The answer model is prompted to stay strict: it should answer only from the retrieved file context and say `I don't know based on the provided file.` when the file does not contain the answer.

## Features

- Supports `.csv`, `.tsv`, `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, and `.ods` files.
- Reads all sheets from workbook formats.
- Uses local `sentence-transformers/all-minilm-l6-v2` embeddings.
- Stores chunks in Chroma and FAISS.
- Runs an interactive question loop.
- Uses `ChatGroq` for answer generation.

## Project Structure

```text
chain/              RAG ingestion and query pipeline
config/             Environment and model settings
embeddings/         Local embedding model wrapper
loaders/            Spreadsheet loaders
models/             Groq Scout and answer model wrappers
retriever/          FAISS retriever
sample_files/       Example input files
text_processing/    Text splitter
vectorstore/        Chroma vector store wrapper
main.py             CLI entry point
```

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv excel_bot
.\excel_bot\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HUGGINGFACE_API_KEY=your_huggingface_api_key
```

`GROQ_API_KEY` is required for the LLM. `HUGGINGFACE_API_KEY` is currently loaded by settings, but embeddings run locally through `sentence-transformers`.

## Usage

Run the app:

```powershell
python main.py
```

When prompted, enter a spreadsheet path:

```text
Enter spreadsheet path [data.xlsx]: sample_files/breast-cancer.csv
```

Then ask questions in the loop:

```text
Ask a question (or type 'exit' to quit): What are the columns in this file?
```

Type `exit`, `quit`, or `q` to stop.

## Configuration

Main settings live in `config/settings.py`:

```python
SCOUT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDER_MODEL = "sentence-transformers/all-minilm-l6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHROMA_DB_PATH = "./chroma_db"
FAISS_INDEX_PATH = "./faiss_index"
```

Scout preprocessing is disabled by default in `RAGChain` to keep ingestion faster and avoid large Groq requests. Raw file chunks are embedded directly.

## Notes

- The first run can take longer because the local embedding model may need to download and initialize.
- Generated indexes are saved to `chroma_db/` and `faiss_index`.
- If the model cannot answer from the retrieved spreadsheet context, it should refuse instead of guessing.
