import logging
import os
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)

from src.vector_db import VectorDB

class RAGAssignment:
    def __init__(self):
        self.vector_db = VectorDB()

        self.llm = ChatGroq(model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                            temperature = 0.7,
                            api_key = os.getenv("GROQ_API_KEY"))
        
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
        
    def process(self, user_query: str) -> str:
        search_result = self.vector_db.similarity_search(query=user_query)
        logging.info("llm call begins...")
        return self.llm_chain.invoke({"context": search_result, "question": user_query})

def main():
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