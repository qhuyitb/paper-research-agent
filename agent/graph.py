from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.planner import create_plan
from agent.tools import rag_search, extract_keypoints, compare_papers
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Node 1: planner
def planner_node(state: AgentState) -> AgentState:
    """
    Nhận query => tạo plan => lưu vào state
    Chạy 1 lần duy nhất ở đầu
    """
    
    print("[Node: Planner] Tạo plan...")
    plan = create_plan(
        query=state["query"],
        chat_history=state.get("chat_history", [])
    )
    return {
        **state,
        "plan": plan,
        "current_step": 0,
        "intermediate_results": [],
        "context": [],
        "selected_papers": [],
        "citations": []
    }

# Node 2: executor
def executor_node(state: AgentState) -> AgentState:
    """
    Thực hiện từng step trong plan
    Mỗi lần gọi => thực hiện 1 step => tăng current_step
    LangGraph sẽ loop lại node này cho đến hết plan
    """
    
    step_idx = state["current_step"]
    if not state["plan"] or step_idx >= len(state["plan"]):
        return {
            **state,
            "current_step": step_idx + 1,
            "final_answer": "Tôi chỉ trả lời câu hỏi về AI/ML research papers.\n\nVí dụ:\n- *What is attention mechanism?*\n- *Compare Transformer and RNN?*"
        }
    step = state["plan"][step_idx]
    print(f"Executor node — Step {step_idx + 1}/{len(state['plan'])}: '{step}'")

    result = None
    new_context = list(state.get("context", []))
    new_papers = list(state.get("selected_papers", []))

    if step.startswith("rag_search:"):
        # Retrieve: tìm papers
        query = step.replace("rag_search:", "").strip()
        result = rag_search(query, top_k=5)

        # Lưu papers tìm được vào selected_papers
        for r in result["results"]:
            if r not in new_papers:
                new_papers.append(r)

        new_context.append({
            "step": step,
            "data": result
        })

    elif step.startswith("extract_keypoints:"):
        topic = step.replace("extract_keypoints:", "").strip()

        relevant = [p for p in new_papers
                    if topic.lower() in p.get("title", "").lower()
                    or topic.lower() in p.get("text", "").lower()]

        if not relevant:
            # Fallback: search lại 
            fallback = rag_search(topic, top_k=3)
            relevant = fallback["results"]

        paper = relevant[0]
        result = extract_keypoints(
            text=paper.get("text", ""),
            title=paper.get("title", "")
        )

    elif step.startswith("compare_papers:"):
        # Reasoning: so sánh 2 papers
        # Lấy 2 kết quả extract_keypoints trước đó
        extractions = [r for r in state.get("intermediate_results", [])
                       if "analysis" in r]

        if len(extractions) >= 2:
            result = compare_papers(
                summary1=extractions[-2]["analysis"],
                title1=extractions[-2]["title"],
                summary2=extractions[-1]["analysis"],
                title2=extractions[-1]["title"]
            )
        else:
            result = {"comparison": "Not enough papers to compare."}

    elif step == "synthesize":
        # Bước cuối — tổng hợp tất cả => final_answer
        # Xử lý ở synthesize_node riêng
        result = {"status": "ready_to_synthesize"}

    return {
        **state,
        "current_step": step_idx + 1,
        "context": new_context,
        "selected_papers": new_papers,
        "intermediate_results": [result] if result else []
    }
        
# Node 3: synthesizer
def synthesize_node(state: AgentState) -> AgentState:
    """
    Tổng hợp tất cả kết quả => final_answer
    Chạy sau khi executor xong tất cả steps
    """

    print(f"NODE: Synthesizer — Tổng hợp final answer.")

    # Gộp tất cả intermediate results thành context
    all_results = state.get("intermediate_results", [])
    context_text = ""

    for r in all_results:
        if "context" in r:
            context_text += f"\nSearch Results:\n{r['context']}\n"
        if "analysis" in r:
            context_text += f"\nPaper Analysis ({r.get('title','')}):\n{r['analysis']}\n"
        if "comparison" in r:
            context_text += f"\nComparison:\n{r['comparison']}\n"
    if not context_text.strip():
        return {
            **state,
            "final_answer": "Tôi không tìm thấy thông tin về chủ đề này trong database papers AI/ML.",
            "citations": []
        }
    # Gộp chat history vào prompt
    history_text = ""
    for msg in state.get("chat_history", [])[-6:]:  
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content'][:300]}\n"
    prompt = f"""You are an AI research assistant. Answer the user's question based ONLY on the research context below.

{"Previous conversation:" + history_text if history_text else ""}

Current Question: {state['query']}

Research Context:
{context_text}

Instructions:
- Respond in the same language as the user's question
- Give a clear, well-structured answer
- Cite specific papers when relevant (use paper titles)
- If comparing papers, use a table format
- End with a "References" section listing papers used"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    final_answer = response.choices[0].message.content

    # Tạo citations từ selected_papers
    final_answer = response.choices[0].message.content

    citations = []
    seen_titles = set()
    for p in state.get("selected_papers", []):
        title = p.get("title", "")
        if title and title[:30].lower() in final_answer.lower():
            if title not in seen_titles:
                seen_titles.add(title)
                citations.append({
                    "title": title,
                    "authors": p.get("authors", ""),
                    "year": p.get("year", ""),
                    "url": p.get("url", "")
                })

    print(f"Final answer generated ({len(final_answer)} chars)")

    return {
        **state,
        "final_answer": final_answer,
        "citations": citations
    }


# Routing logic
def should_continue(state: AgentState) -> str:
    """
    Quyết định bước tiếp theo:
    - Còn step trong plan => tiếp tục execute
    - Step hiện tại là synthesize => chuyển sang synthesize_node
    - Hết plan => END
    """
    
    current_step = state["current_step"]
    plan = state["plan"]

    if current_step >= len(plan):
        return "end"

    next_step = plan[current_step]
    if next_step == "synthesize":
        return "synthesize"

    return "continue"

# Build graph
def build_graph():
    """
    Xây dựng LangGraph

    Flow:
    START => planner => executor => (loop hoặc synthesize) => END
    """
    graph = StateGraph(AgentState)

    # Thêm nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("synthesizer", synthesize_node)

    # Edges
    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")

    # Conditional edge: executor => loop lại hoặc synthesize hoặc end
    graph.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue": "executor",    # loop lại executor
            "synthesize": "synthesizer",
            "end": END
        }
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()

# Run agent
def run_agent(query: str, chat_history: list = []) -> dict:
    app = build_graph()

    initial_state = AgentState(
        query=query,
        messages=[],
        plan=[],
        current_step=0,
        context=[],
        selected_papers=[],
        intermediate_results=[],
        final_answer="",
        citations=[],
        chat_history=chat_history
    )

    final_state = app.invoke(initial_state)

    return {
        "query": query,
        "plan": final_state["plan"],
        "answer": final_state["final_answer"],
        "citations": final_state["citations"]
    }


if __name__ == "__main__":
    # Test 3 cases 
    test_queries = [
        "What is attention mechanism?",
        "Compare Transformer and RNN",
        "What are limitations of BERT?"
    ]

    for query in test_queries:
        print(f"QUERY: {query}")
        result = run_agent(query)
        print(f"\nPlan: {result['plan']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result['citations'][:3]:
            print(f"  - {c['title']} ({c['year']})")