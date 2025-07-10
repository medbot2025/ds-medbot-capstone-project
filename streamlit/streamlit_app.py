import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
import os
import chatbot_helper as ch
from config import LLM_MODEL


model = LLM_MODEL
st.title(f"Chatbot using {model} + medical RAG")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("Hi, I'm a medical chatbot, how can I help you?"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    ch.log_message_to_csv("user", prompt)  # Log user message
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ch.chatbot_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    ch.log_message_to_csv("assistant", response)  
    # Log bot message

# Reset conversation button    
st.button('Reset Chat', on_click= ch.reset_conversation)

# Download chat history
if os.path.exists("chat_history.csv"):
    with open("chat_history.csv", "rb") as f:
        st.download_button(
            label="📥 Download Chat History (CSV)",
            data=f,
            file_name="chat_history.csv",
            mime="text/csv"
        )