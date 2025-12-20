import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from properties.vector_config_loader import load_config as vector_config

import logging

from pathlib import Path

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

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

        self.load_into_vector = config.get("load_into_vector", False)
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
        logging.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        logging.info(f"Vector database initialized with collection: {self.collection_name}")

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
        
    
    def load_documents(self) -> list[str]:
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


    def add_documents(self, doc_list: list[str] | None = None) -> None:
        """
        Add documents to the vector database.
        
        Args:
            doc_list: List of text documents paths which to be added to the vector DB
        """

        if self.load_into_vector is False:
            logging.info("Loading into vector DB is disabled in the configuration.")
            return
        
        if doc_list is None or len(doc_list) == 0:
            logging.info("starting to load documents form raw_data...")
            doc_list = self.load_documents()
        
      
        doc_id = "doc_id_"
        counter = 0
        for i, doc in enumerate(doc_list):
            logging.info("Processing document...\n", i)
            chunks = self.chunk_documents(text=doc, chunk_size=500, chunk_overlap=50)
            for chunk in chunks:
                embeddings =self.embedding_model.encode([chunk])
                self.collection.add(embeddings=embeddings,
                                    metadatas={"doc": chunk},
                                    ids=doc_id + str(counter),
                                    documents=[chunk])
                counter += 1

    def get_all_documents(self) -> None:
        """Print all documents in the vector database."""

        results = self.collection.get()
        logging.info(f"Total documents in vector DB: {self.collection.count()}")
        logging.info(results)
        logging.info("done uploading, now printing all the documents in the vector db")

        for i, meta in enumerate(results["metadatas"]):
            logging.info(f"ID: {results['ids'][i]}")
            logging.info("Chunk text:", meta["doc"])
            logging.info("-" * 40)


    def delete_all_data(self) -> None:
        """Delete all documents in the vector database."""
        self.collection.delete(ids=self.collection.get().get("ids", []))
        logging.info("All data deleted from the vector database.")

        # or alternative way to delete all data
        # self.collection.delete(ids=self.collection.get()["ids"])
        # logging.info(self.collection.count())

    
    def similarity_search(self, query: str, n_results: int = 5) -> dict[str, any]:
        """
        Perform a similarity search in the vector database.

        Args:
            query: The query text
            n_results: Number of similar results to retrieve
        """

        query_embedding = self.embedding_model.encode([query])
        logging.info("Performing similarity search...")
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        return results