import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from properties.vector_config_loader import load_config as vector_config

import logging

from pathlib import Path

class VectorDB:

    def __init__(self, collection_name: str = None,
                  embedding_model: str = None,
                  db_type: str = "chromadb"):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        # Load configuration
        config = vector_config()

        self.load_into_vector = config.get(db_type, {}).get("load_into_vector", False)
        self.chunk_type = config.get(db_type, {}).get("chunk_type", "text-splitter")

        self.collection_name = collection_name or config.get(db_type, {}).get(
            "collection_name", "rag_documents"
        )
        self.embedding_model_name = embedding_model or config.get(db_type, {}).get(
            "embedding_function", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=config.get(db_type, {}).get(
            "persist_directory", "./chroma_db"
        ))

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_documents(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
        """
        Add documents to the vector database in chunks.

        Args:
            text: The text document to be chunked
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        
        chunks = []
        if self.chunk_type == "simple-word":
            start = 0
            text_length = len(text)

            while start < text_length:
                end = min(start + chunk_size, text_length)
                chunk = text[start:end]
                chunks.append(chunk)
                start += chunk_size - chunk_overlap

            return chunks
        elif self.chunk_type == "text-splitter":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            return chunks
        elif self.chunk_type == "semantic-splitting":
            raise NotImplementedError("Semantic splitting is not implemented yet.")
        else:
            raise ValueError(f"Unknown chunk type: {self.chunk_type}")
        
    
    def load_documents() -> list[str]:
        """
        Load documents for demonstration.

        Returns:
            List of sample documents
        """
        results = []
        documents_path = Path(__file__).resolve().parent.parent / "raw_data"

        for files in documents_path.glob("*.txt"):
            with open(files, 'r', encoding='utf-8') as file:
                results.append(file.read())
        
        return results


    def add_document(self, doc_list: list[str]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            doc_list: List of text documents paths which to be added to the vector DB
        """

        if self.load_into_vector is False:
            print("Loading into vector DB is disabled in the configuration.")
            return
        
        if doc_list is None or len(doc_list) == 0:
            logging.info("starting to load documents form raw_data...")
            doc_list = self.load_documents()
        
        for doc in doc_list:
            chunks = self.chunk_documents(text=doc, chunk_size=500, chunk_overlap=50)
            # self.embedding_model.