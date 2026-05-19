from chain import RAGChain
from models import LLMModel

def main():
    # Initialize RAG Chain
    rag_chain = RAGChain()
    llm_model = LLMModel()

    # Ingest Excel File
    excel_file = "data.xlsx"
    chunks_processed = rag_chain.ingest(excel_file)
    print(f"Processed {chunks_processed} chunks")

    # Query
    query = "What is the main information?"
    response = rag_chain.query(query, llm_model)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()