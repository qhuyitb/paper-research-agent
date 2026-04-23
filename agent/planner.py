from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PLANNER_SYSTEM_PROMPT = """You are a research assistant that plans how to answer questions about AI/ML papers.

Given a user query, create a step-by-step plan using these available tools:
- rag_search(query): Search for relevant papers in the database
- extract_keypoints(text, title): Extract key contributions, methods, limitations from a paper
- compare_papers(summary1, title1, summary2, title2): Compare two papers side by side

Rules:
1. For simple concept questions => just 1-2 steps (search + answer)
2. For comparison questions → search both topics separately, then compare
3. For limitation questions → search + extract keypoints focusing on limitations
4. Always end with a "synthesize" step

Return ONLY a JSON array of steps, no explanation.

Examples:
Query: "What is attention mechanism?"
Plan: ["rag_search: attention mechanism", "synthesize"]

Query: "Compare Transformer and RNN"  
Plan: ["rag_search: Transformer architecture", "rag_search: RNN recurrent neural network", "extract_keypoints: Transformer", "extract_keypoints: RNN", "compare_papers: Transformer vs RNN", "synthesize"]

Query: "What are limitations of BERT?"
Plan: ["rag_search: BERT limitations", "extract_keypoints: BERT", "synthesize"]"""

def is_new_topic(query: str, chat_history: list) -> bool:
    """
    Dùng LLM kiểm tra câu hỏi có liên quan đến history không
    Nếu topic mới => reset history
    """
    if not chat_history:
        return True

    # Lấy 4 tin nhắn gần nhất
    recent = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        recent += f"{role}: {msg['content'][:150]}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Determine if the new question is related to the previous conversation or starts a completely new topic.
Reply with only: RELATED or NEW_TOPIC"""
            },
            {
                "role": "user",
                "content": f"Previous conversation:\n{recent}\n\nNew question: {query}"
            }
        ],
        temperature=0
    )

    result = response.choices[0].message.content.strip()
    return result == "NEW_TOPIC"


def create_plan(query: str, chat_history: list = []) -> list[str]:
    
    # Nếu topic mới => không dùng history
    if is_new_topic(query, chat_history):
        print("[Planner] Topic mới => reset history context")
        history_text = ""
    else:
        history_text = ""
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:200]}\n"

    user_content = f"""Previous conversation:
{history_text}

Current query: {query}""" if history_text else query

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    try:
        plan = json.loads(raw)
        print(f"[Planner] Plan tạo xong: {plan}")
        return plan
    except json.JSONDecodeError:
        print("Parse plan thất bại => dùng plan mặc định")
        return [f"rag_search: {query}", "synthesize"]
    
if __name__ == "__main__":
    # Test planner
    queries = [
        "What is attention mechanism?",
        "Compare Transformer and RNN",
        "What are limitations of BERT?"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        plan = create_plan(q)
        print(f"Plan: {plan}")