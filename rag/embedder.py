import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_chunks(input_path="data/processed/papers_processed.json",
                  save_path="data/processed/embeddings.npy",
                  batch_size=64):

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"Tổng số chunks: {len(texts)}")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    np.save(save_path, all_embeddings)
    print(f"Đã embed {len(texts)} chunks → shape: {all_embeddings.shape}")
    print(f"Lưu vào: {save_path}")

    return chunks, all_embeddings

if __name__ == "__main__":
    chunks, embeddings = embed_chunks()
    print(f" vector shape: {embeddings.shape}")
    print(f"Mỗi vector có {embeddings.shape[1]} chiều")