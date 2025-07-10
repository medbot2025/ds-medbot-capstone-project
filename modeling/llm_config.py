# modeling/llm_config.py
"""This module provides configuration for the Large Language Model (LLM) used in the application.
It allows for setting and getting the current model, and initializes the LLM with a default model.
The LLM is defined using the Ollama library, which provides a simple interface to interact with various LLMs.
"""


# This assumes you have Ollama installed and running. 
# You can change the model name to any other model available on Ollama. https://ollama.com/
# If you haven not downloaded the model yet, it will automatically download it for you.
# For example, you can use "llama3", "deepseek-r1", or "medllama2".
# Make sure to adjust the base_url if your Ollama server is running on a different port or host.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_ollama import ChatOllama
from ollama_files.ollama_helper import run_ollama
from config import LLM_MODEL

# Internal variable to track the current model name
_current_model = LLM_MODEL  

# Check if the Ollama service is running and the model is ready.
def ollama_check(model):
   if run_ollama(model):
       print("Model is ready, your code can continue!")
   else:
       print("Model could not be prepared!")

# Function to set the current LLM model
def set_model(model_name):
    """Set the current LLM model."""
    global llm, _current_model
    llm = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434", # default Ollama endpoint
        temperature=0
    )
    _current_model = model_name
    ollama_check(model_name)
    return llm

# Function to get the current LLM model name
def get_model():
    """Get the current LLM model name."""
    return _current_model


