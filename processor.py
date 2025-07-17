import fitz
doc = fitz.open("data/pdfs/descriptions.pdf") 
big_text = ""
for page in doc: 
    text = page.get_text()
    big_text = big_text + text



def sliding_window_chunking(big_text, window_size=200, step=75):
    words = big_text.split()
    chunks = [' '.join(words[i:i+window_size]) for i in range(0, len(words), step)]
    return chunks

sliding_chunks = sliding_window_chunking(big_text, window_size=400, step=80)


from sentence_transformers import SentenceTransformer
sentences = sliding_chunks

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

