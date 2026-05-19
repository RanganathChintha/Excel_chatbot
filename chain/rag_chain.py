from loaders import ExcelLoader
from models import ScoutModel, LLMModel
from text_processing import TextSplitter
from embeddings import Embedder
from vectorstore import ChromaVectorStore
from retriever import FAISSRetriever
class RAGChain:
    def __init__(self, use_scout_processing: bool = False):
        print("Initializing RAG chain...", flush=True)
        self.excel_loader = ExcelLoader()
        self.use_scout_processing = use_scout_processing
        self.scout_model = ScoutModel() if use_scout_processing else None
        self.text_splitter = TextSplitter()
        self.embedder = Embedder()
        self.vector_store = ChromaVectorStore()
        self.retriever = FAISSRetriever()
        print("RAG chain initialized.", flush=True)

    def ingest(self, excel_file_path: str):
        """
        Chain: excel_loader -> text_splitter -> scout_model ->
               embedder -> vector_store -> retriever
        """
        # Step 1: Load Excel
        print(f"Loading Excel file: {excel_file_path}", flush=True)
        text = self.excel_loader.load_as_text(excel_file_path)
        print(f"Loaded {len(text)} characters.", flush=True)

        # Step 2: Split Text before Scout to stay within model request limits.
        print("Splitting text into chunks...", flush=True)
        raw_chunks = self.text_splitter.split(text)
        print(f"Created {len(raw_chunks)} chunks.", flush=True)

        # Step 3: Process each chunk with Scout Model
        if self.use_scout_processing:
            chunks = []
            for index, chunk in enumerate(raw_chunks, start=1):
                print(f"Processing chunk {index}/{len(raw_chunks)} with Scout...", flush=True)
                try:
                    chunks.append(self.scout_model.process(chunk))
                except Exception as exc:
                    print(f"Scout failed for chunk {index}; using raw chunk. {exc}", flush=True)
                    chunks.append(chunk)
        else:
            print("Skipping Scout preprocessing during ingest.", flush=True)
            chunks = raw_chunks

        # Step 4: Generate Embeddings
        print("Generating embeddings...", flush=True)
        embeddings = self.embedder.embed_batch(chunks)

        # Step 5: Add to Vector Store (Chroma)
        print("Saving chunks to Chroma...", flush=True)
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.vector_store.add(chunk_ids, embeddings, chunks)

        # Step 6: Add to Retriever (FAISS)
        print("Saving chunks to FAISS...", flush=True)
        self.retriever.add(embeddings, chunks)

        # Persist
        print("Persisting indexes...", flush=True)
        self.vector_store.persist()
        self.retriever.save()
        print("Ingest complete.", flush=True)

        return len(chunks)

    def query(self, query_text: str, llm_model: LLMModel) -> str:
        """
        Chain: query -> embedder -> retriever -> llm
        """
        # Step 1: Embed Query
        print("Embedding query...", flush=True)
        query_embedding = self.embedder.embed(query_text)

        # Step 2: Retrieve Documents
        print("Retrieving relevant chunks...", flush=True)
        retrieved_docs, distances = self.retriever.retrieve(query_embedding, k=5)

        # Step 3: Generate Response with LLM
        print("Generating answer...", flush=True)
        context = "\n".join(retrieved_docs)
        response = llm_model.generate(context, query_text)

        return response
