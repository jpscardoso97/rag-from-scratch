import json
import numpy as np
import openai
import os
import pandas as pd
import sqlite3

from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_ = load_dotenv(find_dotenv())
openai.api_key  = os.getenv('OPENAI_API_KEY')

class Inference:

    def __init__(self):
        self.db = sqlite3.connect('data/db.sqlite')
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def generate_response(self, use_rag, messages):
        ctx_messages = messages[:-1]
        prompt = messages[-1]['content']

        if use_rag:
            rag_output = self.search_rag(prompt)
            if rag_output:
                rag_ctx = f"\nPlease use the following information (only if it's relevant to answer the question, otherwise ignore it): \n{rag_output}"
                prompt += rag_ctx

        ctx_messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=ctx_messages,
        temperature=1,
        max_tokens=3350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    
        return response.choices[0].message["content"]
    
    def search_rag(self, user_prompt):
        emb_input = self.embedding_model.encode(user_prompt)

        # Get all rows from embeddings table
        df = pd.read_sql_query("SELECT * from embeddings", self.db)
        df['embedding'] = df['embedding'].apply(lambda x: json.loads(x))

        # Find the most similar embedding
        cosine_sim = cosine_similarity(np.array([emb_input]), np.vstack(df['embedding'].apply(lambda x: np.array(x))))
        most_similar = df.iloc[np.argmax(cosine_sim)]
        
        best_result = most_similar['original']

        return best_result
        