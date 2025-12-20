import logging
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(level=logging.INFO)

from src.vector_db import VectorDB

class RAGAssignment:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        self.vector_db = VectorDB()

        # self.llm = ChatGroq(model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        #                     temperature = 0.7,
        #                     api_key = os.getenv("GROQ_API_KEY"))
        
        self.llm = self._initialize_llm()

        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the question using the context below:
            Context:
            {context}
            
            Question: 
            {question}
            """
        )
        self.llm_chain = self.prompt_template | self.llm | StrOutputParser()

    
    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.3
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.3
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.3,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )
        
    def process(self, user_query: str) -> str:
        search_result = self.vector_db.similarity_search(query=user_query)
        logging.info("llm call begins...")
        return self.llm_chain.invoke({"context": search_result, "question": user_query})

def main():
    """Main function to demonstrate the RAG assistant."""
    assignment = RAGAssignment()

    # loads based on the yaml config field = load_into_vector
    assignment.vector_db.add_documents()

    while True:
        print("\n" + "=" * 50)
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        print("\n" + "=" * 50)
        if user_query.lower() == 'exit' or user_query.lower() == 'quit':
            logging.info("Exiting the program.")
            break
        result = assignment.process(user_query)
        print("-" * 50)
        print(f"\nResponse: \n{result}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()