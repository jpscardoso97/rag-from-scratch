# rag-from-scratch
This project is part of the AIPI-590(LLMs) course from Duke University's MEng in Artificial Intelligence.

## Project Description
This is an implementation of RAG without the use of any framework such as LangChain. The project is implemented in Python is two-folded:
1. Creates a pipeline to extract data from the Duke MEng in AI website, create chunks of data, generate embedding using a small language model and store them in a SQLite database. 
2. Implements a chatbot interface based in streamlit. That will expose an inference process that generates the embedding of the user input and retrieves the most similar chunk of data from the database using cosine similarity. This additional context will be added to a prompt to GPT-3.5 Turbo to generate a response.  

## Running the project
1. Install the requirements
```bash
pip install -r requirements.txt
```
2. Run the setup pipeline
```bash 
python setup.py
```
3. Run the chatbot interface
```bash
streamlit run app.py
```

Note: to use the chatbot interface you will need to have a GPT-3.5 Turbo API key. You can get one [here](https://beta.openai.com/signup/)
