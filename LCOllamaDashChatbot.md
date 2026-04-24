# LCOllamaDashChatbot — Component Architecture

C4 Component-level request-flow view of the Dash + LangGraph + Ollama chatbot.

## Diagram

```mermaid
flowchart TD
  User([User / Browser])

  subgraph Dash["Dash Web UI (localhost)"]
    direction TB
    Input[/"Textarea + Send Button"/]
    CB1["on_send_or_clear callback\nbuilds LangChain messages"]
    Store[(chat-store\ndcc.Store)]
    CB2["update_chat_window callback\nrenders bubbles"]
    Window["Chat Window"]
  end

  subgraph Graph["LangGraph Agent Graph"]
    direction TB
    AgentNode["agent_node\nLLM decides: respond or call tool"]
    Route{should_continue}
    ToolNode["tool_node\nexecutes tool calls"]
  end

  Ollama["ChatOllama / llama3.2\nlocalhost:11434"]
  WriterAPI["Writer API\napi.writer.com\n(external)"]

  User -->|"click Send"| Input
  Input -->|"n_clicks trigger"| CB1
  CB1 -->|"agent_app.invoke(lc_messages)"| AgentNode
  AgentNode -->|"chat completion"| Ollama
  Ollama -->|"AIMessage"| AgentNode
  AgentNode --> Route
  Route -->|"tool_calls present → tools"| ToolNode
  ToolNode -->|"POST search query"| WriterAPI
  WriterAPI -->|"search result text"| ToolNode
  ToolNode -->|"ToolMessage"| AgentNode
  Route -->|"no tool_calls → END"| CB1
  CB1 -->|"append AIMessage to history"| Store
  Store -->|"data change triggers"| CB2
  CB2 -->|"render chat bubbles"| Window
  Window -->|"display response"| User
```

## Notes

- **Scope**: Full request path from user input through the LangGraph ReAct loop to rendered response.
- **Deliberate omissions**: Error path (`[ERROR] …` fallback in the callback) and the Clear button flow are not shown; they bypass the agent entirely and reset `chat-store` to `[]`.
- **Assumptions**: Ollama runs locally at `localhost:11434`; `WRITER_API_KEY` env var is set. The tool loop can cycle multiple times (tools → agent → tools…) before reaching END — the diagram shows one cycle for clarity.
- **State**: `chat-store` accumulates the full conversation history as plain dicts; the callback reconstructs `HumanMessage` / `AIMessage` objects on every turn before invoking the graph.
```