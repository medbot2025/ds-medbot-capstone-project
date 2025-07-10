# Medical Chatbot version 0.1

## Installation (Tested on macOS)

You can find the installation steps for the first version below.  
These steps have been tested only on macOS and may not work as expected on other operating systems.


## Step A: Create a Virtual Environment

### **`macOS`** type the following commands : 

- Install the virtual environment and the required packages by following commands:

    ```bash
    pyenv local 3.11.3
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```


## Step 1: Installation

### 1.1: Install Homebrew

` Note: If you want to install the desktop (GUI) version of Ollama, please skip to Step 1.2`

If Homebrew is not already installed on your system, please install it first.

You can check if Homebrew is installed by running the following command in your terminal:

```bash
brew --version
```
You can install Homebrew by running the following command in your terminal: For more info please see : [Homebrew Website](https://brew.sh/)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 1.2: Install Ollama

If Ollama is not already installed on your system, you can install it using the `install_ollama.sh`script provided in this repository. Alternatively, you can download and install the Desktop (GUI) version from the [Ollama Website](https://ollama.com/).
**For stability reasons, downloading and installing the Desktop (GUI) version of the Ollama is recommended.**

If you want to install via script, make sure it has execution permissions:

```bash
cd ollama_files
chmod +x install_ollama.sh
```
Then, run the script:
```bash
./install_ollama.sh
```

### Step 1.3: Configure and Run Training

Open the `config.py` file and adjust the necessary parameters such as:

- `LLM_MODEL` – the local model you want to use
- `CHUNK_SIZE` – the size of each text chunk
- `CHUNK_OVERLAP` – the overlap between consecutive chunks

Please make sure that the dataset is located in the `data` folder and the file is named `medquad_cleaned.csv`.
Or change the `CLEANED_DATA_PATH` in `config.py`.

After making your changes, run the training script:

```bash
cd modeling
python train.py
```

## Step 2: Starting the MedChatbot

### 2.1: Start Ollama Server
Make sure Ollama is runnig in your computer. To check it please go to `http://localhost:11434` in your browser. You should see something like `Ollama is running`.<br>

### To run the Ollama:

### 2.1.1: If you downloaded the desktop (GUI) version from the website:
 Simply launch it like any other desktop application. 
### 2.1.2: If you installed it via Homebrew:
 Start the server by running the following command in your NEW terminal (it would run as a service):
```bash
ollama serve
```

### Step 2.1: Launch the Chatbot UI

Navigate to the `streamlit` folder and run the `streamlit_app.py` file using Streamlit:

```bash
cd streamlit
streamlit run streamlit_app.py
```



**Note:** If the LLM model has not been downloaded before, it may take some time to download on first run.  
Please check the terminal output for progress.

## Important Info for Restarting the Application

- To restart the chatbot, it is enough to follow the instructions in **Step 2**.

- If for any reason, you need to re-run the training process (i.e., recreate the vector database),  
you only need to follow **Step 1.3**.

- When you specify a model name in the `LLM_MODEL` field in `config.py`,  
  the corresponding model will be automatically downloaded to your machine (if not already available). Please make sure that the model is available on ollama.com

- If you make any changes to the `config.py` file, especially the `LLM_MODEL`,  
  please re-run **Step 2.1** to apply the updated configuration.

## Notes

- The sentence-transformer currently does not work offline.  
  Offline support will be implemented and updated in a future version.
- Data cleaning and exploratory data analysis (EDA) script is available in `cleaning.py`,  
  but it is not yet automated.  
  Currently, the chatbot uses a pre-cleaned dataset (`medquad_cleaned.csv`).  
  This process will be automated in a future version.
- To maintain the code structure, please make all changes only via the `config.py` file.  
  If you have any additional parameters you want to add, please let us know.


## Troubleshooting

- To check if the local LLM is running properly, you may need to inspect the retrieved chunks.  
  When you ask questions to the chatbot, monitor the terminal where you launched Streamlit to see the chunks being retrieved in response.

- Most error messages can be seen directly in the terminal where you run `streamlit_app.py`.