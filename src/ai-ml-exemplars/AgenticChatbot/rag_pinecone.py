"""
Pinecone RAG Implementation for Agentic Chatbot
"""
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from pinecone import Pinecone
import time

load_dotenv()

# Initialize components
embeddings = OpenAIEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0.2)

class PineconeRAG:
    def __init__(self, index_name: str = "rag-text-embedding"):
        self.index_name = index_name
        self.vector_store = None
        self.initialize_vector_store()

    def initialize_vector_store(self):
        """Initialize the Pinecone vector store"""
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

            # Check if index exists, create if not
            if self.index_name not in pc.list_indexes().names():
                pc.create_index(name=self.index_name,
                                dimension=3072,
                                metric='cosine',
                                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}})

                # Wait for index to be ready
                while not pc.describe_index(self.index_name).index.status['ready']:
                    time.sleep(1)

            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=embeddings,
                namespace="default"
            )
            print(f"Successfully initialized Pinecone vector store: {self.index_name}")

        except Exception as e:
            print(f"Error initializing Pinecone vector store: {e}")
            self.vector_store = None

    def add_documents(self, documents: list):
        """Add documents to the vector store"""
        if not self.vector_store:
            raise Exception("Vector store not initialized")

        try:
            # Add documents with metadata
            ids = self.vector_store.add_documents(documents)
            print(f"Successfully added {len(ids)} documents to Pinecone")
            return ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def query(self, query_text: str, k: int = 4):
        """Query the vector store for relevant documents"""
        if not self.vector_store:
            raise Exception("Vector store not initialized")

        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query_text, k=k)
            return results
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []

    def get_relevant_context(self, query: str):
        """Get relevant context for a query"""
        docs = self.query(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

# Example usage
def example_usage():
    # Initialize RAG system
    rag = PineconeRAG()

    # Sample documents to add
    sample_docs = [
        Document(
            page_content="The capital of France is Paris. Paris is located in the north-central part of France.",
            metadata={"source": "wikipedia", "topic": "geography"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            metadata={"source": "tech_blog", "topic": "AI"}
        ),
        Document(
            page_content="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            metadata={"source": "travel_guide", "topic": "tourism"}
        )
    ]

    # Add documents (uncomment to run)
    # rag.add_documents(sample_docs)

    # Query example
    query = "What is the capital of France?"
    context = rag.get_relevant_context(query)
    print(f"Query: {query}")
    print(f"Context: {context}")

if __name__ == "__main__":
    example_usage()