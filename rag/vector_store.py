from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import json
import os
from tqdm import tqdm


COLLECTION_NAME = "arxiv_papers"
VECTOR_SIZE = 384  # Kích thước vector embedding từ model SentenceTransformer 'all-MiniLM-L6-v2'

def get_client():
    return QdrantClient(host="localhost", port=6333)

def create_collection(client):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = VectorParams(size=VECTOR_SIZE,
                                     distance=Distance.COSINE)
    )
    print(f"Tạo collection '{COLLECTION_NAME}' thành công")
    
def index_chunks(chunks_path = "data/processed/papers_processed.json", 
                 embeddings_path = "data/processed/embeddings.npy",
                 batch_size = 100):
    client = get_client()
    create_collection(client)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(embeddings_path)

    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        points = [
            PointStruct(
                id=idx + i,
                vector=batch_embeddings[idx].tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    **chunk["metadata"]  # title, authors, year, url
                }
            )
            for idx, chunk in enumerate(batch_chunks)
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print(f"Index xong {len(chunks)} chunks vào Qdrant")
    print(f"Xem tại: http://localhost:6333/dashboard")


if __name__ == "__main__":
    index_chunks()
