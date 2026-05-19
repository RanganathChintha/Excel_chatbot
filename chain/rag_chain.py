from loaders import ExcelLoader
from models import ScoutModel, LLMModel
from text_processing import TextSplitter
from embeddings import Embedder
from vectorstore import ChromaVectorStore
from retriever import FAISSRetriever
from typing import List

class RAGChain:
    def __init__(self):
        self.excel_loader = ExcelLoader()
        self.scout_model = ScoutModel()
        self.text_splitter = TextSplitter()
        self.embedder = Embedder()
        self.vector_store = ChromaVectorStore()
        self.retriever = FAISSRetriever()

    def ingest(self, excel_file_path: str):
        """
        Chain: excel_loader -> scout_model -> text_splitter -> 
               embedder -> vector_store -> retriever
        """
        # Step 1: Load Excel
        text = self.excel_loader.load_as_text(excel_file_path)

        # Step 2: Process with Scout Model
        processed_text = self.scout_model.process(text)

        # Step 3: Split Text
        chunks = self.text_splitter.split(processed_text)

        # Step 4: Generate Embeddings
        embeddings = self.embedder.embed_batch(chunks)

        # Step 5: Add to Vector Store (Chroma)
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.vector_store.add(chunk_ids, embeddings, chunks)

        # Step 6: Add to Retriever (FAISS)
        self.retriever.add(embeddings, chunks)

        # Persist
        self.vector_store.persist()
        self.retriever.save()

        return len(chunks)

    def query(self, query_text: str, llm_model: LLMModel) -> str:
        """
        Chain: query -> embedder -> retriever -> llm
        """
        # Step 1: Embed Query
        query_embedding = self.embedder.embed(query_text)

        # Step 2: Retrieve Documents
        retrieved_docs, distances = self.retriever.retrieve(query_embedding, k=5)

        # Step 3: Generate Response with LLM
        context = "\n".join(retrieved_docs)
        response = llm_model.generate(context, query_text)

        return response