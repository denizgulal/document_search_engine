import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Embedder modeli
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Soru-cevap modeli
model_id = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

# Veri yükle
with open("index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Soru al
query = input("Question: ")

# Chunkları embed et (önceden embed'lenmişse bunu dışarıda yap, burada demo için tekrar hesaplıyorum)
embeddings = embed_model.encode(chunks, convert_to_tensor=False)
query_embedding = embed_model.encode(query)

# Benzerlik skorları ve en iyi 3 chunk
similarities = cosine_similarity([query_embedding], embeddings)[0]
top_k = similarities.argsort()[-1:][::-1]
print("Closest chunk indexes:", top_k)

def chunk_to_smaller_pieces(text, max_len=80):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = []
    for i in range(0, len(token_ids), max_len):
        piece_ids = token_ids[i:i+max_len]
        piece_text = tokenizer.decode(piece_ids)
        pieces.append(piece_text)
    return pieces


max_context_tokens = 300
context_chunks = []

for idx in top_k:
    small_chunks = chunk_to_smaller_pieces(chunks[idx], max_len=80)
    for chunk in small_chunks:
        # Toplam token sayısını encode ile kontrol et
        trial_context = "\n\n".join(context_chunks + [chunk])
        trial_len = len(tokenizer.encode(trial_context, add_special_tokens=False))
        if trial_len <= max_context_tokens:
            context_chunks.append(chunk)
        else:
            break

context = "\n\n".join(context_chunks)

prompt = f"""
You are an assistant. Answer the question based on the context below.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

#print("Prompt token length:", len(tokenizer.tokenize(prompt)))

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🔍 Answer:\n", answer.split("Answer:")[-1].strip())
