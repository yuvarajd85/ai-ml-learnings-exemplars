import os
import getpass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

# Choose ONE embeddings option:

# Option 1: OpenAI embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Option 2: HuggingFace embeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CORPUS_DIR = "rag_corpus/Business_logic"
PERSIST_DIR = "chroma_kpi"

# Load markdown docs from the KPI corpus folder.
loader = DirectoryLoader(
    CORPUS_DIR,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
docs = loader.load()

# Split into overlapping chunks to improve retrieval quality.
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Build and persist the vector index for later retrieval.
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
vectordb.persist()
print(f"✅ Indexed {len(chunks)} chunks into {PERSIST_DIR}")
