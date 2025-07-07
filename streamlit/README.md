# Basic RAG pipeline + LLM prompting with streamlit application

The file MedBot_with_Memory.py consists of two parts:
1. Respond to user queries with LLM + RAG given an already created medical vector database.
2. Streamlit application on top of that.

<br>

**Install streamlit**<br>
One can use the requirements.txt file, I added streamlit there. Or directly using pip, as streamlit is a python package.

<br>

**Groq API key**<br>
You need a groq api key to access the llama model. 
We already used it in the rag project. I just copied the .env file containing this key from the rag project to the main folder here.

<br>

**Vector database**<br>
It is assumed that the vector database has already been created. One will have to adjust the path to the database in MedBot_with_Memory.py<br>
(It's the code line `retriever = retrieve_from_vector_db("vector_databases/vector_db_med_quad_answers`"))

<br>

**Run web app**<br>
Go to streamlit folder.<br>
Terminal command:   `streamlit run MedBot_with_Memory.py`

<br>
<br>


Now a browser page with the chatbot interface should be visible.          
