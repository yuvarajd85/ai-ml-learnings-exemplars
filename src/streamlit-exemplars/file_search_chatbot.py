'''
Created on 3/24/2025 at 1:59 AM
By yuvaraj
Module Name: file_search_chatbot
'''
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("File Search Chatbot")

    root_dir = st.text_input("Enter the root directory to search:", value=os.getcwd())

    # Chat interface
    with st.chat_message("assistant"):
        st.markdown("Hi! I can help you search for files. Try asking things like 'Find all .pdf files' or 'Search for report'.")

    user_input = st.chat_input("Type your file search query...")

    if user_input:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Search logic
        query = user_input.strip()
        files = search_files(query, root_dir)

        if files:
            response = f"Found {len(files)} file(s):\n\n" + "\n".join(files[:10])  # limit to 10 results
            if len(files) > 10:
                response += f"\n\n...and {len(files) - 10} more."
        else:
            response = "No matching files found."

        # Save bot response
        st.session_state.messages.append({"role": "assistant", "content": response})

    print(st.session_state.messages)

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def search_files(query, directory):
    result = []
    print(f"Query: {query}")

    for folder_name, sub_folders, filenames in os.walk(directory):
        for filename in filenames:
            if query.lower() in filename.lower():
                full_path = os.path.join(folder_name, filename)
                result.append(full_path)

    return result


if __name__ == '__main__':
    main()
