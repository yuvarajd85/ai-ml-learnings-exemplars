'''
Created on 8/28/2025 at 10:39 PM
By yuvaraj
Module Name: claudechatbot
'''
import os
import uuid
import streamlit as st

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()


def main():

    # --------- Sidebar: config ---------
    st.set_page_config(page_title="Bedrock + LangChain (Haiku) Chat", page_icon="🤖")

    st.sidebar.title("⚙️ Settings")

    # Common Haiku model IDs (pick one you have access to)
    DEFAULT_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"  # Claude 3.5 Haiku
    ALT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Claude 3 Haiku (earlier)

    model_id = st.sidebar.selectbox(
        "Model ID",
        [DEFAULT_MODEL, ALT_MODEL],
        index=0,
    )

    region_name = st.sidebar.text_input("AWS Region", value=os.getenv("AWS_REGION", "us-east-1"))

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    max_tokens = st.sidebar.slider("Max output tokens", 128, 4096, 512, 64)

    system_prompt = st.sidebar.text_area(
        "System prompt",
        value="You are a concise, helpful assistant. Keep answers under 8 sentences when possible."
    )

    # Session management
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    col1, col2 = st.sidebar.columns(2)
    new_clicked = col1.button("New Session")
    clear_clicked = col2.button("Clear Chat")

    if new_clicked:
        session_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = session_id
        # re-init UI messages for new session
        st.session_state.setdefault("ui_histories", {})
        st.session_state["ui_histories"][session_id] = []

    # --------- Initialize LLM (Bedrock via LangChain) ---------
    @st.cache_resource(show_spinner=False)
    def get_llm(model_id: str, region: str, temperature: float, max_tokens: int):
        return ChatBedrockConverse(
            model_id=model_id,
            region_name=region,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    llm = get_llm(model_id, region_name, temperature, max_tokens)

    # --------- Memory per session (LangChain) ---------
    # We'll keep a dict of session_id -> InMemoryChatMessageHistory inside Streamlit's session_state
    if "lc_histories" not in st.session_state:
        st.session_state["lc_histories"] = {}

    def get_history(sess_id: str) -> InMemoryChatMessageHistory:
        if sess_id not in st.session_state["lc_histories"]:
            st.session_state["lc_histories"][sess_id] = InMemoryChatMessageHistory()
        return st.session_state["lc_histories"][sess_id]

    # UI chat logs per session (for Streamlit rendering)
    if "ui_histories" not in st.session_state:
        st.session_state["ui_histories"] = {}
    if session_id not in st.session_state["ui_histories"]:
        st.session_state["ui_histories"][session_id] = []

    if clear_clicked:
        # Clear BOTH LangChain memory + UI history for current session
        st.session_state["lc_histories"][session_id] = InMemoryChatMessageHistory()
        st.session_state["ui_histories"][session_id] = []

    # --------- Build the chain with memory ---------
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    chat_with_memory = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
        history_factory_config={"session_id": session_id},
    )

    # --------- App header ---------
    st.title("🤖 LangChain + Bedrock (Claude Haiku)")
    st.caption(f"Session: `{session_id}`  •  Model: `{model_id}`  •  Region: `{region_name}`")

    # --------- Render previous messages ---------
    for role, content in st.session_state["ui_histories"][session_id]:
        with st.chat_message(role):
            st.markdown(content)

    # --------- Chat input ---------
    user_input = st.chat_input("Ask something…")

    # --------- Handle a new turn ---------
    if user_input:
        # Show the user's message
        st.session_state["ui_histories"][session_id].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream the model response using LangChain's streaming
        with st.chat_message("assistant"):
            # Placeholder for streaming text
            output_placeholder = st.empty()
            streamed_text = ""

            # When using RunnableWithMessageHistory, pass config with session_id
            config = {"configurable": {"session_id": session_id}}

            # Stream tokens
            for chunk in chat_with_memory.stream(
                    {"system_prompt": system_prompt, "input": user_input},
                    config=config
            ):
                # chunk here is typically an AIMessageChunk; extract the text safely
                piece = ""
                if hasattr(chunk, "content"):
                    # content can be str or a list of parts; normalize to str
                    if isinstance(chunk.content, str):
                        piece = chunk.content
                    else:
                        try:
                            piece = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in chunk.content
                            )
                        except Exception:
                            piece = str(chunk.content)
                else:
                    piece = str(chunk)

                streamed_text += piece
                output_placeholder.markdown(streamed_text)

            # Persist assistant message in Streamlit UI history
            st.session_state["ui_histories"][session_id].append(("assistant", streamed_text))


if __name__ == '__main__':
    main()
