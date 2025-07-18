# modeling/retrieval_pipeline.py
"""
This module provides functions to create a retrieval pipeline using LangChain.
It includes loading CSV data, splitting documents, creating an embedding vector database,
and connecting chains for retrieval.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import CSVLoader
from modeling.llm_config import set_model
from config import LLM_MODEL, DB_NAME, CHUNK_SIZE, CHUNK_OVERLAP, SENTENCE_TRANSFORMER_MODEL_NAME, SENTENCE_TRANSFORMER_MODEL_PATH

llm_model_name = LLM_MODEL  # LLM model name, can be changed as needed from config.py
db_name = DB_NAME  # Vector database name, can be changed as needed from config.py
# CHUNK_SIZE and CHUNK_OVERLAP are defined in config.py

# Define the Sentence Transformer model name and path
st_model_name = SENTENCE_TRANSFORMER_MODEL_NAME
st_model_path = f'{SENTENCE_TRANSFORMER_MODEL_PATH}/{st_model_name}'  # Local path



def load_csv_data(csv_path):
    """
    Load text data from a CSV file.
    """
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits documents into chunks of given size and overlap
    """
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    # Just to add id for etch chunks to map it later 
    for i, chunk in enumerate(chunks):
         chunk.metadata.update({
        "id": f"chunk_{i}",
    })
    return chunks


def create_embedding_vector_db(chunks, db_name):
    """
    This function uses the open-source embedding model HuggingFaceEmbeddings 
    to create embeddings and store those in a vector database called FAISS, 
    which allows for efficient similarity search
    """
    check_sentence_transformer_model()
    embedding = HuggingFaceEmbeddings(
        model_name= st_model_path,  # Local path
        #model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}  
    )
    
    # create the vector store 
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )
    
    # save vector database locally
    vectorstore.save_local(f"./vector_databases/vector_db_{db_name}")
    return vectorstore


def retrieve_from_vector_db(vector_db_path):
    """
    This function splits out a retriever object from a local vector database
    """
    check_sentence_transformer_model()
    embeddings = HuggingFaceEmbeddings(
        model_name=st_model_path,  # Local path
        #model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True} 
    )
 
    medical_vectorstore = FAISS.load_local(
        folder_path=vector_db_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )
    retriever = medical_vectorstore.as_retriever()
    return retriever ,medical_vectorstore


def connect_chains(retriever):
    """
   This function connects stuff_documents_chain with retrieval_chain
    """
    llm = set_model(llm_model_name)
    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=hub.pull("langchain-ai/retrieval-qa-chat")
    )
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=stuff_documents_chain
    )
    return retrieval_chain


# This block ensures the model is downloaded only once
def check_sentence_transformer_model():
    """
    Check if the sentence transformer model exists locally, if not, download it.
    """
    if not os.path.exists(st_model_path):
       st_model = SentenceTransformer(st_model_name)
       print(f"Downloading model {st_model_name}...")
       st_model.save(st_model_path)
       print(f"Model {st_model_name} downloaded and saved to {st_model_path}")
