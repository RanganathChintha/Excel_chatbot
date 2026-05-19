from chain import RAGChain
from models import LLMModel
import sys

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    # Initialize RAG Chain
    print("Starting Excel chatbot...", flush=True)
    rag_chain = RAGChain()
    llm_model = LLMModel()

    # Ingest spreadsheet file
    excel_file = input("Enter spreadsheet path [data.xlsx]: ").strip() or "sample_files/breast-cancer.csv"
    chunks_processed = rag_chain.ingest(excel_file)
    print(f"Processed {chunks_processed} chunks", flush=True)

    # Query loop
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.", flush=True)
            break
        if not query:
            continue

        response = rag_chain.query(query, llm_model)
        print(f"\nResponse: {response}", flush=True)

if __name__ == "__main__":
    main()
