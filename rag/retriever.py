from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np          
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()


COLLECTION_NAME = "arxiv_papers"
model =SentenceTransformer('all-MiniLM-L6-v2')
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 



def get_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key )
    else:
        return QdrantClient(host="localhost", port=6333)
    
def search(query: str, top_k: int = 5) -> list:
    client = get_client()
    query_vector = model.encode(query).tolist()

    # Rerank
    candidates = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k * 4,  
        with_payload=True
    )

    if not candidates:
        return []

    pairs = [
        (query, r.payload.get("title", "") + ". " + r.payload.get("text", ""))
        for r in candidates
    ]
    rerank_scores = reranker.predict(pairs)

    # Gộp score mới vào, sort lại
    reranked = sorted(
        zip(candidates, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    formatted = []
    for r, rerank_score in reranked:
        formatted.append({
            "score": round(float(rerank_score), 4),  
            "text": r.payload.get("text", ""),
            "title": r.payload.get("title", ""),
            "authors": r.payload.get("authors", ""),
            "year": r.payload.get("year", ""),
            "url": r.payload.get("url", ""),
            "categories": r.payload.get("categories", ""),
        })

    return formatted


def format_context(results: list[dict]) -> str:
    """
    Gộp kết quả search thành 1 đoạn context
    để đưa vào prompt cho LLM
    """
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[{i+1}] {r['title']} ({r['year']})\n"
            f"Authors: {r['authors']}\n"
            f"URL: {r['url']}\n"
            f"Content: {r['text']}\n"
            f"Relevance score: {r['score']}"
        )
    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    # Test RAG 
    query = "attention mechanism"
    print(f"Query: '{query}'")
    results = search(query, top_k=5)
    for r in results:
        print(f"{r['score']} | {r['title'][:60]}")