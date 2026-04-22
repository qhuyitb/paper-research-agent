import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_chunking(text, threshold=0.75):
    sentences = nltk.sent_tokenize(text)  
    if len(sentences) <= 1:
        return [text]

    embeddings = model.encode(sentences, show_progress_bar=False)

    chunks = []
    current_chunk = [sentences[0]]  

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])

        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def preprocess_papers(input_path="data/raw/papers.json",
                       save_path="data/processed/papers_processed.json"):
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    all_chunks = []

    for paper in tqdm(papers, desc="Semantic chunking"):
        full_text = f"{paper['title']}. {paper['abstract']}"
        chunks = semantic_chunking(full_text, threshold=0.75)

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            all_chunks.append({
                "chunk_id": f"{paper['id']}_chunk{i}",
                "text": chunk,
                "metadata": {
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "year": paper["year"],
                    "categories": paper["categories"],
                    "url": paper["url"],
                }
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Đã tạo {len(all_chunks)} chunks từ {len(papers)} papers")
    print(f"Lưu vào: {save_path}")
    return all_chunks

if __name__ == "__main__":
    chunks = preprocess_papers()

    print("\nPreview 3 chunks đầu tiên:")
    for c in chunks[:3]:
        print(f"\nchunk_id : {c['chunk_id']}")
        print(f"text     : {c['text'][:150]}...")