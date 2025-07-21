

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
import os
import csv
import streamlit as st
from langchain_core.runnables import chain
from modeling.llm_config import set_model
from modeling.retrieval_pipeline import retrieve_from_vector_db
from config import LLM_MODEL, VECTOR_DB_PATH, SIMILARITY_THRESHOLD, K

# Define the model to be used for the chatbot
model = LLM_MODEL # 'llama3'

# Set the model using the Ollama
llm = set_model(model) 

# threshold for similarity computation between user query and database vectors
similarity_threshold = SIMILARITY_THRESHOLD # 0.5

# max number of retrieved chunks (could be less depending on similarity scores and threshold)
k = K  # 6

# Retrieve the vector store from the vector databas
vectorstore = retrieve_from_vector_db(VECTOR_DB_PATH)[1] 

def log_message_to_csv(role, message, file_path="chat_history.csv"):# we need to add time stamp
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Role", "Message"])  # Write header only once
        writer.writerow([role, message])

def get_msg_content(msg):
    return msg.content

def reset_conversation():
    st.session_state.messages = []

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
        #print(context)
        context = ()

    # Get the final answer
    return qa_chain.invoke({
        **input,
        "context": context
    })

def chatbot_response(user_input):
    result = qa_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "123"}},
    )
    return f"{result}"

contextualize_system_prompt = (
"""Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question which can be understood \
without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
)

# Create a prompt template for contextualizing the question
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

# This helper function extracts the content field from a message object.
contextualize_chain = (
    contextualize_prompt
    | llm
    | get_msg_content
)

# Define the system prompt for the question-answering chain
# It instructs the model to answer medical questions based on the provided context.
qa_system_prompt = (
    "You are an assistant for any medical concerns. Answer the user's questions or queries based on the below context. " 
    "If the context doesn't contain any relevant information to the question or if the context is empty, "
    "do NOT make something up. The only thing you are allowed to say is 'Sorry, I don't know. Can you rephrase your medical concern?'"
    "Do not be talkative with non-medical enquiries, just say: 'Sorry, I don't know. Can you rephrase your medical concern?':"
    "\n\n"
    "###"
    "{context}"
    "###"
)

# Create a prompt template for the question-answering chain
# It includes the system prompt, chat history, and user input.
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

# Define the question-answering chain
# It combines the prompt, LLM, and a function to extract the message content.
qa_chain = (
    qa_prompt
    | llm
    | get_msg_content
)

# Initialize chat history for the chain
# This will store the chat messages during the interaction.
chat_history_for_chain = InMemoryChatMessageHistory()
qa_with_history = RunnableWithMessageHistory(
    history_aware_qa,
    lambda _: chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)



