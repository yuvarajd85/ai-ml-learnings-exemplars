<div>
<h1 style="text-align: center; text-decoration: underline;">Building Agentic AI Systems</h1>
</div>

----
## <u>Introduction Section</u>

System Prompt, context, Specs, Role, Task - in prompts 

Drop, Alibi - internal positional encoding

search_arxiv -? 

ArXiv Search Tool that retrieves information from top research papers from arxiv.org relevant to the query

[artificial.ai](https://artificial.ai)

from typing import Literal &rarr; Literal["tools","__end__"] &rarr; Python what is this ?

`|` operator in LangChain, LangGraph and CrewAi &rarr; what is this ?

Memory Management for Lanchain &rarr; using Redis Cache & Memcache 

[LEANN](https://github.com/yichuan-w/LEANN) &rarr; RAG for Laptop

Rubric &rarr; what is this ?


----

## <u>React Agent using Langchain and Langgraph</u>

----

## <u>CrewAI</u>

[CrewAI](https://docs.crewai.com/)

Disable Telemetry when using in production

os.envrion["CREWAI_DISABLE_TELEMETRY"] = 'true'

os.envrion["OTEL_SDK_DISABLED"] = 'true'

[OpenAI Agent Builder](https://platform.openai.com/agent-builder)

----

## <u>LangGraph</u>

Nodes, Edges and State

Convert the RDBMS DB to JSON format - convert the table records as collection 
of dictionary and construct a metadata schema that explains the table schema, 
table relationship with other tables, datastore location etc. 

Hence, this could be a better format for RAG solution

---

## **Session 1 – Introduction to Building Agentic AI Systems**

---

### **1. Concept Overview: ReAct Agent Architecture (LangChain / LangGraph)**

#### **Concept**

The **ReAct framework** (Reason + Act) allows an AI agent to reason about context before performing an action such as calling a tool or retrieving data.
Each cycle alternates between:

* **Reasoning:** internal planning and thought generation
* **Acting:** invoking a function, tool, or API call
* **Observation:** interpreting the tool’s output
* **Iteration:** refining reasoning based on results

#### **Architecture (LangChain / LangGraph)**

* **LangChain:** Previously used `create_react_agent()`; now moving toward `create_agent()` under the unified `langchain.agents` namespace (as of v1.0 – Nov 2025).
* **LangGraph:** Provides a graph-based view of the ReAct process—each reasoning ↔ action ↔ observation step forms a node and edge in an execution graph.

#### **Best Practices**

* Keep production dependencies stable for **1–2 months after a major release**.
* Use the **official migration guide** when transitioning from older ReAct implementations.
* Combine LangChain for logic orchestration and LangGraph for visual debugging.

#### **Key Takeaway**

ReAct unifies decision-making and execution, forming the backbone of modern **agentic AI** systems that can autonomously plan and act within a defined environment.

---

### **2. Demo 1 – Building a Tool-Use ReAct Agent (System with LangChain / LangGraph)**

#### **Objective**

Demonstrate how to **build an agent capable of reasoning, choosing tools, and executing tasks** in a structured environment.

#### **Architecture**

1. Define **custom tools** (e.g., API connectors, data retrievers).
2. Register tools within LangChain.
3. Create a **ReAct Agent** that alternates between planning → tool call → reflection.
4. Monitor and visualize via LangGraph’s state flow.

#### **Environment**

Run the hands-on notebook on **Google Colab** for easy setup and GPU access.
The demo notebook (“Get it from HERE”) mirrors real-world workflows.

#### **Key Takeaways**

* Understand tool registration and invocation flow.
* Learn how to interpret the ReAct reasoning trace.
* Observe the value of graph visualization for debugging agent decisions.

---

### **3. Agentic AI Systems with CrewAI**

#### **Concept**

**CrewAI** extends the ReAct idea to **multi-agent collaboration**. Each agent has a defined role—researcher, planner, executor—and they coordinate to achieve a shared goal.

#### **Architecture**

* Agents share memory and communicate asynchronously.
* A **controller** (or “crew manager”) orchestrates task assignment.
* Each agent executes a ReAct-style loop within its own context.
* Outputs cascade between agents, forming a cooperative reasoning network.

#### **Use Cases**

* Complex pipelines: data gathering → analysis → report generation.
* Distributed planning and execution across specialized domains.
* Dynamic task allocation for scalability.

#### **Key Takeaway**

CrewAI transforms isolated reasoning loops into a **collaborative ecosystem**—ideal for workflows where multiple reasoning chains need to converge.

---

### **4. Demo 2 – Tool-Use ReAct System with CrewAI**

#### **Objective**

Show how multiple agents can collaborate on tool usage and task completion using CrewAI’s orchestration layer.

#### **Implementation Flow**

1. Define individual agents with specific roles and tools.
2. Implement shared state management to track context across agents.
3. Run a task workflow (e.g., research → analysis → summarization).
4. Inspect CrewAI’s built-in ReAct flows and tool-calling logic.

#### **Environment**

Google Colab recommended for execution and notebook interactivity.

#### **Key Takeaways**

* Learn how agents communicate and share context.
* Observe task handoffs in a multi-agent setup.
* Understand the advantages of CrewAI for orchestrated problem solving.

---

### **5. Database Schema – Shared Context for Agents**

#### **Purpose**

The **database schema** acts as a real-world sandbox for structured data reasoning.
It supports Text2SQL agents by giving them relational context.

#### **Components**

* **Tables:** Users, Orders, Transactions (typical relational setup).
* **Relationships:** Foreign keys link business entities.
* **Metadata:** Used by the agent to map natural-language queries to SQL commands.

#### **Key Takeaway**

A well-structured schema enables agents to reason logically over data and build queries without manual SQL knowledge.

---

### **6. Demo 3 – Text2SQL Agent in LangGraph**

#### **Objective**

Develop a LangGraph-based agent that translates natural language into SQL queries autonomously.

#### **Workflow**

1. **Interpret Query:** Parse user intent from text.
2. **Schema Understanding:** Identify target tables and columns.
3. **SQL Generation:** Use reasoning chains to construct valid queries.
4. **Execution + Interpretation:** Run SQL on the database and format results for humans.

#### **Outcomes**

* Build a self-contained data assistant using LangGraph.
* Understand how semantic reasoning bridges text and structured data.
* Visualize the reasoning process through LangGraph’s graph view.

#### **Key Takeaway**

Text2SQL agents demonstrate how reasoning frameworks can extend beyond language tasks into **data engineering and analytics automation**.

---

### **7. Notes for the Future / Best Practices**

#### **Stability and Versioning**

* **Freeze dependencies** in production.
* Delay upgrades until APIs stabilize (1–2 months post-release).
* Review migration guides before refactoring agents.

#### **Experimentation Environment**

* Use **Google Colab notebooks** for sandbox testing.
* Keep prototype agents isolated from production pipelines.

#### **Integration Guidance**

* Pair ReAct (LangChain) for logic and LangGraph for visualization.
* Use CrewAI for multi-agent extensions.

#### **Key Takeaway**

Robust Agentic AI systems emerge from **careful version management, clear architecture, and disciplined experimentation**.

---

### **8. Summary of Learning Outcomes**

By the end of Session 1, participants will be able to:

* Explain the ReAct architecture and its role in agentic AI.
* Build tool-use agents using LangChain and LangGraph.
* Orchestrate multi-agent systems with CrewAI.
* Implement reasoning-driven Text2SQL solutions.
* Apply best practices for stability and production migration.


-----
## <u> Day-2 Workshop Notes </u>

### Key Design Patterns 

- Reflection Pattern 
- Tool Use Pattern
- Planning Pattern
- Multi-Agent Pattern
  - Single Agent 
  - Network Agent 
  - Supervisor
  - Supervisor as tools
  - Hierarchical
  - Custom

- langsmith
  - API Key is required
- agentevals 


MCP Server
- [FastMCP](https://gofastmcp.com/getting-started/welcome)

%%writefile

!nohup python &rarr; ? 

----
 
## **Session 2 – Building Effective Agentic AI Systems**

Instructor: *Dipanjan (DJ) Sarkar*
Focus: Practical architectures, engineering techniques, and hands-on demos for building **robust, observable, and memory-aware Agentic AI systems**.

---

### **1. Single-Agent vs. Multi-Agent Architecture**

#### **Conceptual Overview**

Agentic AI systems can operate as either **single-agent** or **multi-agent** architectures depending on the scope and complexity of the tasks.
Understanding their trade-offs is crucial for designing scalable and efficient systems.

#### **Single-Agent Architecture**

A **single agent** independently handles all stages of reasoning, planning, and execution.
It’s ideal for simple, linear tasks that don’t require specialization or collaboration.

**Key Characteristics:**

* Centralized control and memory.
* Simplified debugging and tracing.
* Limited scalability for complex workflows.
* Easier to deploy and maintain.

**Example Use Cases:**

* Chatbots performing task-specific functions.
* Text summarization, Q&A, or code explanation.
* Single-tool orchestration (e.g., calling an API or database).

#### **Multi-Agent Architecture**

In **multi-agent systems**, several autonomous agents collaborate—each optimized for a specialized role.
Agents communicate through shared memory, messaging protocols, or task queues.

**Key Characteristics:**

* Distributed control and specialization.
* Agents can act in parallel for efficiency.
* Requires coordination logic or a “controller agent.”
* Higher complexity but greater flexibility.

**Example Use Cases:**

* Research agent + reasoning agent + summarizer agent pipeline.
* Multi-departmental problem solving (e.g., finance + operations + marketing).
* Agentic orchestration in RAG (Retrieval-Augmented Generation) systems.

#### **Best Practices**

* Use **single-agent systems** for MVPs and prototypes.
* Adopt **multi-agent systems** for complex, long-running, or interdisciplinary workflows.
* Ensure robust communication protocols between agents (LangGraph or CrewAI recommended).

**Hands-On Practice:**
Notebook provided in session – *“Single-Agent vs Multi-Agent Architecture”* (Google Colab).
Includes step-by-step implementation and visualization of both architectures.

---

### **2. Agent Observability with LangSmith**

#### **Concept Overview**

Agent observability is the process of **tracking, monitoring, and understanding** how an AI agent behaves internally—across reasoning, tool calls, and state transitions.

LangSmith provides an advanced observability platform for LangChain and LangGraph-based agents.

#### **Key Features of LangSmith**

1. **Trace Visualization:**
   Observe the reasoning and action sequence of agents.
   Each step (prompt → thought → tool → result) is logged.
2. **Error Tracking:**
   Detect anomalies such as hallucinations, incorrect tool usage, or infinite loops.
3. **Prompt Evaluation:**
   Compare performance across prompt versions and model configurations.
4. **Run Metadata:**
   Capture version, environment, and context data for reproducibility.
5. **Collaboration:**
   Teams can review traces and diagnose performance issues together.

#### **Importance**

* Provides visibility into **decision chains** that are otherwise opaque.
* Reduces debugging time and improves interpretability.
* Enables **continuous evaluation and improvement** of agent logic.

#### **Best Practices**

* Always integrate LangSmith or equivalent observability tools in production pipelines.
* Use tagging and metadata logging for better experiment tracking.
* Combine with **LangGraph visualizations** for end-to-end trace analysis.

**Hands-On Practice:**
Notebook – *“Agent Observability with LangSmith”* (Google Colab).
Demonstrates live monitoring of agent reasoning, tool usage, and context flow.

---

### **3. Multi-Department MCP Architecture for AI Agents**

#### **Concept Overview**

MCP (**Multi-Context/Control Plane**) architecture allows multiple departments or domains (e.g., finance, HR, operations) to run their **own specialized agents** under a shared governance framework.

It represents **multi-agent systems at organizational scale**.

#### **Architecture Design**

* **Departmental Agents:** Each department has a specialized agent (e.g., “FinanceGPT”, “HRBot”).
* **Central Control Plane (MCP):** Coordinates requests, manages access, and enforces policies.
* **Shared Context Layer:** Maintains unified memory, authentication, and context exchange.
* **APIs and Tools:** Agents use department-specific tools while adhering to shared governance.

#### **Advantages**

* **Scalability:** Each domain can evolve independently.
* **Security:** Control plane enforces data governance and API-level permissions.
* **Interoperability:** Agents can share insights while maintaining domain boundaries.
* **Adaptability:** Departments can integrate new models or tools without breaking global consistency.

#### **Implementation Notes**

* Use **LangGraph or CrewAI** to represent the agent network.
* Centralize **prompt templates, logging, and versioning** under MCP.
* Establish **memory boundaries** per department for data privacy.

**Hands-On Practice:**
Notebook – *“Multi-Department MCP Architecture for AI Agents”* (Google Colab).
Shows multi-agent collaboration across virtual departments using control-plane logic.

---

### **4. Memory Context Engineering for AI Agents**

#### **Concept Overview**

Memory context engineering is about **designing how agents remember, forget, and use past information** to make better decisions.

It is essential for maintaining continuity across sessions, enabling personalization, and optimizing performance in multi-turn reasoning.

#### **Types of Memory**

1. **Short-Term Memory:**
   Temporary context such as the last few user queries or actions.
   Stored in memory buffers or context windows.
2. **Long-Term Memory:**
   Persistent memory across sessions—can include summaries, user profiles, or vector-embedded documents.
3. **Episodic Memory:**
   Contextual understanding of sequences and dependencies (similar to human episodic recall).
4. **Semantic Memory:**
   Abstract knowledge stored in embeddings, useful for reasoning and generalization.

#### **Engineering Techniques**

* Use **vector databases** (e.g., Chroma, Pinecone, FAISS) to store semantic memories.
* Implement **memory summarization** to condense long dialogues.
* Control **retrieval scope** dynamically to maintain focus and efficiency.
* Integrate **context windows** with adaptive truncation for long conversations.

#### **Challenges**

* Memory bloat: Retaining too much irrelevant context.
* Hallucination: Misusing stale or unrelated past information.
* Latency: Increased retrieval overhead in long-term memory systems.

#### **Best Practices**

* Define **clear retention policies** (what to forget, when, and why).
* Maintain **context embeddings** for cross-session reasoning.
* Use **hierarchical memory** (short-term + long-term combined).
* Regularly fine-tune summarization and retrieval strategies.

**Hands-On Practice:**
Notebook – *“Memory Context Engineering for AI Agents”* (Google Colab).
Demonstrates how to design adaptive memory strategies and measure context efficiency.

---

### **5. Best Practices for Building Effective Agentic AI Systems**

1. **Start Small:** Prototype with a single-agent ReAct system before scaling.
2. **Instrument Everything:** Integrate observability early.
3. **Design for Modularity:** Each agent should have a clear role and isolated context.
4. **Guardrails:** Implement governance, logging, and validation layers.
5. **Memory Hygiene:** Avoid uncontrolled growth of context; regularly prune or summarize.
6. **Experimentation Environment:** Always sandbox in **Google Colab or staging environments** before productionizing.

---










