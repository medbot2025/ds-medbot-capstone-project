# config.py

LLM_MODEL = 'llama3' # Before change please check out ollama.com for available models.
VECTOR_DB_PATH = '../modeling/vector_databases/vector_db_medical'
SIMILARITY_THRESHOLD = 0.5  # threshold for similarity computation between user query and database vectors
K = 6  # max number of retrieved chunks (could be less depending on similarity scores and threshold)
CLEANED_DATA_PATH = '../data/medquad_cleaned.csv'  # path to the cleaned medical data
DB_NAME = 'medical'  # name of the vector database
CHUNK_SIZE = 200  # size of each chunk in characters
CHUNK_OVERLAP = 50  # overlap size between chunks in characters
SENTENCE_TRANSFORMER_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # name of the sentence transformer
SENTENCE_TRANSFORMER_MODEL_PATH = '../models'  # local path to save the sentence transformer model
#CHAT_HISTORY_FILE = './chat_history/chat_history.csv' not implemented yet