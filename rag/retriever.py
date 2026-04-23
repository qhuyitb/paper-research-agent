from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np          
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "arxiv_papers"
model =SentenceTransformer('all-MiniLM-L6-v2')


def get_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key )
    else:
        return QdrantClient(host="localhost", port=6333)
def search(query: str, top_k : int = 5) -> list:
    client  = get_client()
    query_vector = model.encode(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True 
    )

    # format results
    formatted = []
    for r in results:
        formatted.append({
            "score": round(r.score, 4),
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
    context = format_context(results)
    print(context)
    
    print(f"Tìm được {len(results)} kết quả")
    print(f"Top paper: {results[0]['title']}")