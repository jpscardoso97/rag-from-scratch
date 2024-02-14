import streamlit as st

from inference import Inference

inference = Inference()

def run():
    # Set the page tab title
    st.set_page_config(page_title="RAG-from-scratch", page_icon="👻", layout="wide")

    # Button to clear the chat history
    if st.button("Clear chat history"):
        st.session_state.messages = []

    st.title("Welcome!")
    st.write("This is a RAG application built from scratch on top of GPT-3.5 Turbo")

    # Toggle button to enable disable RAG
    rag_enabled = st.checkbox("Enable RAG", value=True)

    # Initialize the session state variables to store the conversations and chat history
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Set the page title
    st.header("RAG Chatbot (built from scratch)")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = inference.generate_response(use_rag=rag_enabled, messages=st.session_state.messages)
        
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I didn't understand that."})
            st.chat_message("assistant").write("Sorry, I didn't understand that.")

if __name__ == "__main__":
    run()