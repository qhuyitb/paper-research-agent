import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import nltk

nltk.download('punkt', quiet = True)
nltk.download('punkt_tab', quiet = True)

model = SentenceTransformer('all-MiniLM-L6-v2')


def cosine_similarity(vec1, vec2):
    """Tính cosine similarity giữa hai vector."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def sematic_chunking(text, threshold = 0.75):
    sentences  = nltk.sentence_tokenizers(text)
    if(len(sentences)<= 1):
        return [text]
    embeddings = model.encode(sentences, show_progress_bar=False)
    chunks = []
    current_chunk = sentences[0]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim < threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
