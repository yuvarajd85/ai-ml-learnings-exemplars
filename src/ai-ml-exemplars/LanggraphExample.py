'''
Created on 9/14/2025 at 2:15 AM
By yuvaraj
Module Name: LanggraphExample
'''
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# 1. Setup LLM + retriever
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.load_local("my_index", embeddings)
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 2. Define nodes
def retrieve(state):
    docs = retriever.get_relevant_documents(state["question"])
    return {"docs": docs}

def generate(state):
    answer = qa_chain.run(state["question"])
    return {"draft_answer": answer}

def verify(state):
    # simple heuristic check (can be replaced with evaluator LLM)
    if any(doc.page_content in state["draft_answer"] for doc in state["docs"]):
        return {"verified": True, "final_answer": state["draft_answer"]}
    return {"verified": False}

def refine(state):
    return {"final_answer": f"Refined: {state['draft_answer']} (needs checking)"}

# 3. Build graph
graph = StateGraph()

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
graph.add_node("refine", refine)

graph.set_entry_point("retrieve")

# Edges: define routing
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "verify")
graph.add_conditional_edges(
    "verify",
    lambda state: "refine" if not state["verified"] else None,
    {"refine": "refine"}  # refine branch
)

# 4. Compile
executor = graph.compile()

# 5. Run
result = executor.invoke({"question": "What is the capital of France?"})
print(result["final_answer"])

