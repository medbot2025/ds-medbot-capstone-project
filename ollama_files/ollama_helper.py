# ollama_helper.py
import subprocess
import requests
import time
import sys
#import os 

def check_ollama_service():
    """Check if Ollama service is running"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        return response.status_code == 200
    except:
        return False

def start_ollama_service():
    """Start Ollama service"""
    try:
        # For Windows
        if sys.platform == "win32":
            subprocess.Popen(['ollama', 'serve'], shell=True)
        # For Mac/Linux
        else:
            subprocess.Popen(['ollama', 'serve'])
        
        # Wait for service to start
        print("Starting Ollama service...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_ollama_service():
                print("Ollama service started successfully!")
                return True
        return False
    except Exception as e:
        print(f"Error starting Ollama service: {e}")
        return False

def list_local_models():
    """List locally installed models"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models.get('models', [])]
        return []
    except:
        return []

def check_model_exists(model_name):
    """Check if model exists locally"""
    local_models = list_local_models()
    # Model name might include tag (e.g., llama3:latest)
    model_base = model_name.split(':')[0]
    
    for local_model in local_models:
        if local_model.startswith(model_base):
            return True
    return False

def pull_model(model_name):
    """Pull/download a model"""
    print(f"Downloading {model_name} model...")
    try:
        # Run ollama pull command and wait for it to finish
        process = subprocess.run(
            ['ollama', 'pull', model_name],
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            print(process.stdout)
            print(f"{model_name} model downloaded successfully!")
            return True
        else:
            print(f"Model download error: {process.stderr}")
            return False
    except Exception as e:
        print(f"Error occurred during model download: {e}")
        return False

def test_model(model_name):
    """Test if model is working"""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': 'Hello',
                'stream': False
            }
        )
        return response.status_code == 200
    except:
        return False

def run_ollama(model_name):
    """
    Main function: Run Ollama model
    
    Args:
        model_name (str): Name of the model to run (e.g., 'llama3', 'mistral', 'codellama')
    
    Returns:
        bool: True if model is ready, False otherwise
    """
    
    print(f"Preparing Ollama model: {model_name}")
    
    # 1. Check if Ollama service is running
    if not check_ollama_service():
        print("Ollama service is not running, starting...")
        if not start_ollama_service():
            print("ERROR: Could not start Ollama service!")
            print("Please ensure Ollama is installed: https://ollama.ai")
            return False
    
    # 2. Check if model exists
    if check_model_exists(model_name):
        print(f"{model_name} model is already installed.")
    else:
        print(f"{model_name} model not found.")
        # 3. Download model if it doesn't exist
        if not pull_model(model_name):
            print(f"ERROR: Could not download {model_name} model!")
            return False
    
    # 4. Test if model is working
    print(f"Testing {model_name} model...")
    if test_model(model_name):
        print(f"{model_name} model is ready!")
        return True
    else:
        print(f"ERROR: Could not run {model_name} model!")
        return False

# Additional helper functions

def get_available_models():
    """List all available models in Ollama"""
    models = list_local_models()
    if models:
        print("Installed models:")
        for model in models:
            print(f"  - {model}")
    else:
        print("No models installed yet.")
    return models

def remove_model(model_name):
    """Remove a model"""
    try:
        process = subprocess.run(
            ['ollama', 'rm', model_name],
            capture_output=True,
            text=True
        )
        if process.returncode == 0:
            print(f"{model_name} model removed successfully.")
            return True
        else:
            print(f"Model removal error: {process.stderr}")
            return False
    except Exception as e:
        print(f"Error during model removal: {e}")
        return False