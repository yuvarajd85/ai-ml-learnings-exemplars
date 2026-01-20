'''
Created on 1/17/26 at 9:40 AM
By yuvarajdurairaj
Module Name build_index
'''
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import openai

CORPUS_DIR = "rag_corpus/Business_logic"
PERSITS_DIR = "chroma_kpi"
openai.api_key = os.environ['OPENAI_API_KEY']
# Embedding model name
EMBEDDING_MODEL = "text-embedding-3-large"

def main():
    # Initialize OpenAI Embeddings using LangChain
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)  # Specify which embedding model

    loader = DirectoryLoader(
        CORPUS_DIR,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(documents=docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSITS_DIR
    )

    vectordb.persist()



if __name__ == '__main__':
    main()
