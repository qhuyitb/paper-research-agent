from rag.retriever import search, format_context
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RETRIEVAL TOOL
def rag_search(query: str, top_k: int = 5) -> list[dict]:
    """Tool 1: Tìm papers liên quan trong Qdrant
    Agent gọi tool này khi cần tìm thông tin về 1 topic"""
    
    print(f"[RAG search] query: '{query}'")
    result = search(query, top_k= top_k)
    context =   format_context(result)
    return {
        "query": query,
        "results": result, # list dict để agent xử lý
        "context": context, # string để đưa vào prompt
        "num_found": len(result)
    }

# REASONING TOOL
def extract_keypoints(text: str, title: str = "") -> list[dict]:
    """
    Tool 2: Trích xuất các điểm quan trọng từ 1 paper
    KHÔNG gọi DB — chỉ dùng LLM phân tích text đã có
    """
    
    print(f"[Extract keypoints] title: '{title}'")
    prompt = f"""Analyze this AI/ML paper and extract key information.

Paper: {title}
Content: {text}

Return a structured analysis with:
1. Main contribution (what's new/novel) 
2. Method (how they did it)  
3. Results (key numbers/improvements)
4. Limitations (what doesn't work well)

Be concise, 1-2 sentences per section."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3  
    )

    return {
        "title": title,
        "analysis": response.choices[0].message.content
    }


def compare_papers(summary1: str, title1: str,
                   summary2: str, title2: str) -> dict:
    """
    Tool 3: So sánh 2 papers
    KHÔNG gọi DB — nhận summaries đã có từ extract_keypoints
    """
    
    print(f"[Compare papers] '{title1}' vs '{title2}'")
    prompt = f"""Compare these two AI/ML papers:

Paper 1: {title1}
{summary1}

Paper 2: {title2}  
{summary2}

Create a comparison covering:
1. Core approach (how they differ fundamentally)
2. Strengths of each
3. Weaknesses of each
4. When to use which

Format as a clear comparison table then a summary paragraph."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return {
        "title1": title1,
        "title2": title2,
        "comparison": response.choices[0].message.content
    }