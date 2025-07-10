# Chatbot with memory
# https://medium.com/@saurabhzodex/memory-enhanced-rag-chatbot-with-langchain-integrating-chat-history-for-context-aware-845100184c4f



################################
## 1. RAG pipeline on medical database + LLM prompting
################################
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_core.runnables import chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from modeling.llm_config import set_model
import csv
import os


################################
## Add Logging to CSV
################################

def log_message_to_csv(role, message, file_path="chat_history.csv"):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Role", "Message"])  # Write header only once
        writer.writerow([role, message])


################################

# Hyperparameters

# model
model="llama3-8b-8192"  
#model_name="deepseek-r1-distill-llama-70b"

# threshold for similarity computation between user query and database vectors
similarity_threshold = 0.5

# max number of retrieved chunks (could be less depending on similarity scores and threshold)
k = 6    


################################

load_dotenv()
warnings.filterwarnings("ignore")

llm = set_model('llama3') 

# add req.file 
# Define LLM
#llm = ChatGroq(
#    model=model,   
#    temperature=0,
#    max_tokens=None,
#    timeout=None,
#    max_retries=2
#)


# Load embedded chunks from Vector Database 
def retrieve_from_vector_db(vector_db_path):
    """
    this function splits out a retriever object from a local vector database
    """
    # instantiate embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2'
    )
    vectorstore = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )
    return vectorstore

# Load the vectorstore
vectorstore = retrieve_from_vector_db("vector_databases/vector_db_medical")



# This helper function extracts the content field from a message object. 
# It is used to simplify the extraction of text from model outputs.
def get_msg_content(msg):
    print(msg.content) 
    return msg.content


# This system prompt instructs the model to reformulate the user’s question into a standalone question, 
# ensuring it can be understood without referencing the chat history.
contextualize_system_prompt = (
"""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question which can be understood \
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
)


contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])


contextualize_chain = (
    contextualize_prompt
    | llm
    | get_msg_content
)


qa_system_prompt = (
    "You are an assistant for any medical concerns. Answer the user's questions or queries based on the below context. " 
    "If the context doesn't contain any relevant information to the question or if the context is empty, "
    "do NOT make something up and just say 'Sorry, I don't know. Can you rephrase your concern?':"
    "\n\n"
    "###"
    "{context}"
    "###"
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


qa_chain = (
    qa_prompt
    | llm
    | get_msg_content
)


# Define the overall chain the uses both the retrieved documents and the chat history to answer the question
@chain
def history_aware_qa(input):
    # Rephrase the question if needed
    if input.get('chat_history'):
        question = contextualize_chain.invoke(input)
    else:
        question = input['input']


    # Return docs and relevance scores in the range [0, 1]. 0 is dissimilar, 1 is most similar.
    # could be empty, depending on the threshold
    similarity_results = vectorstore.similarity_search_with_relevance_scores(question, k=k, score_threshold=similarity_threshold)
    if similarity_results:
        context, scores = zip(*similarity_results)
        print(context)
    else:   # no chunks similar enough
        context = ()

    # Get the final answer
    return qa_chain.invoke({
        **input,
        "context": context
    })



chat_history_for_chain = InMemoryChatMessageHistory()
qa_with_history = RunnableWithMessageHistory(
    history_aware_qa,
    lambda _: chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)


def chatbot_response(user_input):
    result = qa_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "123"}},
    )
    return f"{result}"



################################
## 2. Streamlit web app
################################

import streamlit as st


st.title(f"Chatbot using {'model'} + medical RAG")


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
    log_message_to_csv("user", prompt)  # Log user message
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chatbot_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    log_message_to_csv("assistant", response)  
    # Log bot message


def reset_conversation():
    st.session_state.messages = []
    
st.button('Reset Chat', on_click=reset_conversation)

# Download chat history
if os.path.exists("chat_history.csv"):
    with open("chat_history.csv", "rb") as f:
        st.download_button(
            label="📥 Download Chat History (CSV)",
            data=f,
            file_name="chat_history.csv",
            mime="text/csv"
        )