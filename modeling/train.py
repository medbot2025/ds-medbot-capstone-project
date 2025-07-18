#modeling/train.py

from pathlib import Path
from retrieval_pipeline import load_csv_data, split_documents, create_embedding_vector_db
from config import CLEANED_DATA_PATH

#loading medical documents from csv file
print(f"The trained dataset path: {CLEANED_DATA_PATH}")
medical_docs = load_csv_data(csv_path = CLEANED_DATA_PATH)
print(f"Number of loaded pages: {len(medical_docs)}")

# Splitting documents into chunks
medical_chunks = split_documents(medical_docs)

# Show number of chunks created
print(f"Number of chunks created: {len(medical_chunks)}") #"\n",f"Type of the chunks : {type(medical_chunks)}","\n\n" ,medical_chunks)

# Creating a vector database from the chunks 
print("Creating vector database... Please wait...This might take a while...")   
create_embedding_vector_db(chunks=medical_chunks, db_name="medical")
print("Vector database created successfully!")

