import faiss
import pickle
import numpy as np
import openai
import os
import streamlit as st

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_FILE = os.getenv("INDEX_FILE_PATH")
CHUNKS_FILE = os.getenv("CHUNKS_FILE_PATH")

@st.cache_resource
def load_vector_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_embedding(text: str):
    response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL)
    return response['data'][0]['embedding']

def retrieve_relevant_schema(question: str, index, schema_chunks, top_k=3):
    vec = np.array([get_embedding(question)], dtype='float32')
    distances, indices = index.search(vec, top_k)
    return [schema_chunks[i] for i in indices[0]]
