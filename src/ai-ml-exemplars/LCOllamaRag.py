'''
Created on 3/2/26 at 12:25 PM
By yuvarajdurairaj
Module Name LCOllamaRag
'''
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# For a lightweight local vector store:
from langchain_community.vectorstores import FAISS

def main():
    # 1) Sample corpus
    docs = [
        Document(page_content="Polars is a fast DataFrame library built in Rust with lazy execution support."),
        Document(page_content="Pandas is the most common Python DataFrame library; great ecosystem, slower on huge data."),
        Document(page_content="RAG retrieves relevant documents and feeds them to an LLM to reduce hallucinations."),
        Document(page_content="Gradient boosting (e.g., XGBoost/LightGBM) often wins on tabular data."),
    ]

    # 2) Embeddings via Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",  # common local embedding model name
        # base_url="http://localhost:11434",
    )

    # 3) Build vector index + retriever
    vs = FAISS.from_documents(docs, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    # 4) Prompt
    prompt = ChatPromptTemplate.from_template(template="""
You are a precise assistant.

INSTRUCTIONS:
- Answer using ONLY the provided context.
- If the context does not contain the answer, say "I don't know."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )

    # 5) LLM
    llm = ChatOllama(model="llama3.2", temperature=0.2)

    # 6) Chain: retrieve -> format context -> ask model
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    question = "When should I prefer Polars over Pandas, and why?"
    answer = chain.invoke(question)
    print(answer)

if __name__ == "__main__":
    main()
