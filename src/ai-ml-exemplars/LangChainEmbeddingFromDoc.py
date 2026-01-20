'''
Created on 8/28/2025 at 6:05 PM
By yuvaraj
Module Name: LangChainEmbeddingFromDoc
'''
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai

load_dotenv()


def main():
    # Set your API keys for OpenAI
    openai.api_key = os.environ['OPENAI_API_KEY']
    print(os.getenv('OPENAI_API_KEY'))

    #Embedding model name
    EMBEDDING_MODEL="text-embedding-3-large"

    # Initialize OpenAI Embeddings using LangChain
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # Specify which embedding model

    # Load and split document
    loader = TextLoader("G://github//ai-ml-learnings-exemplars//src//resources//datasets//Yuvaraj-Durairaj-Resume.txt")  # Load a text file
    documents = loader.load()

    # # Use a TextSplitter to split the documents into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    # split_documents = text_splitter.split_documents(documents)

    # Load the whole document
    full_text = "/n/n".join(d.page_content for d in documents)

    # Connect to the Pinecone index using LangChain's Pinecone wrapper
    # Add the splitDocuments into Pinecone
    # pinecone_index_name = "sagemaker-guide-embeddings"
    pinecone_index_name = "rag-text-embedding"
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
    # vectorstore.add_documents(documents=split_documents)
    ids = vectorstore.add_texts(texts=[full_text], metadatas=[{"source": documents[0].metadata.get("source","Yuvaraj-Durairaj-Resume")}])

    print(f"Upserted the document with {len(ids)} ")
    print("Embeddings from single text file created and inserted in Pinecone Vector Database successfully!")


if __name__ == '__main__':
    main()
