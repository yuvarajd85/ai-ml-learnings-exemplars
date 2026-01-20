# <u style="text-align:center"> LangChain and LangGraph Workshop </u>

**Generative AI** 

GenAI refers to AI systems (typically powered by LLMs) that can generate text, images, code, etc., based on inputs.

## Agentic AI System

- Langchain &rarr; Sequential 
- LangGraph &rarr; Routing, Branching and Decision based
- FlowGraph

## Langchain
- It's a framework to build applications powered by LLM's like GPT
- It helps connect LLM's with data sources, API's, tools and business systems.
- Enables multi-step workflows instead of single question-answer prompts
- Supports memory, reasoning and tool usage to create smarter AI applications
- Commonly used to build chatbots, copilots and automation solutions. 

## LangGraph
- Graph &rarr; Nodes &rarr; Edges &rarr; Checkpoints &rarr; State

Memory in LangGraph 
- State Persistence
- Types of memory
- Message passing with memory
- Benefits

## Tools Setup

- A small function that performs a concrete action (fetch, compute, query).
- Called inside a node
- Keep tools pure and testable: deterministic inputs &rarr; raise errors on failure. 

```python
def my_node(state):
    data = my_tool(state['query']) # call tool
    state['result'] = data # persist in shared memory
    return state
```

`faiss_index` &rarr; ?

