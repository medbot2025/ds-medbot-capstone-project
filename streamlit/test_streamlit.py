from dotenv import load_dotenv
import warnings
from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import streamlit as st




################################
## 1. RAG pipeline on medical database + LLM prompting
################################

load_dotenv()

warnings.filterwarnings("ignore")

# Define LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# Load embedded chunks from Vector Database and make retrieve object
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
    retriever = vectorstore.as_retriever()
    return retriever, vectorstore


# Load the retriever and index
retriever, vectorstore = retrieve_from_vector_db("../vector_databases/vector_db_med_quad_answers")


def connect_chains(retriever):
    """
    this function connects stuff_documents_chain with retrieval_chain
    """
    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    )
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_documents_chain
    )
    return retrieval_chain

retrieval_chain = connect_chains(retriever)


def medbot(user_query):

    output = retrieval_chain.invoke(
    {"input": user_query}
    )
    return output['answer']




################################
## 2. Streamlit web app
################################

st.title("Chatbot using Llama model + medical RAG")


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
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    response = medbot(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


def reset_conversation():
    st.session_state.messages = []
    
st.button('Reset Chat', on_click=reset_conversation)







