import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import openai


index = faiss.read_index('index/faiss.index')
model = SentenceTransformer('all-MiniLM-L6-v2')


query = "which classes include machine learning"
query_embedding = model.encode([query])  

k = 3
D, I = index.search(query_embedding, k)

print(f"Closest chunk indexes: {I}")
print(f"Distances: {D}")

with open('data/chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)  

top_k_indices = I[0]  
relevant_chunks = [chunks[i] for i in top_k_indices]


prompt = "Read the text below and answer the question:\n\n"
for idx, chunk in enumerate(relevant_chunks, 1):
    prompt += f"{idx}) {chunk}\n\n"
prompt += f"Q: {query}\n\nA:"



openai.api_key = "API_KEYİNİZİ_BURAYA"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.7,
)

answer = response['choices'][0]['message']['content']
print("Model cevabı:\n", answer)