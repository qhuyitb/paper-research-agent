import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import build_graph
from agent.state import AgentState

st.set_page_config(
    page_title="Paper Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Paper Research Agent")
st.caption("Chatbot hỗ trợ nghiên cứu các paper AI/ML từ ArXiv")

# Session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_app" not in st.session_state:
    st.session_state.agent_app = build_graph()
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Nếu là message của assistant thì hiện citations
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📚 Citations"):
                for c in msg["citations"]:
                    st.markdown(
                        f"- [{c['title']}]({c['url']}) "
                        f"— {c['authors'][:50]}... ({c['year']})"
                    )

# Chat input 
if query := st.chat_input("Hỏi về paper AI/ML... (vd: What is attention mechanism?)"):

    # Hiện câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Chạy agent + hiện trace
    with st.chat_message("assistant"):
        with st.status("🤔 Agent đang xử lý...", expanded=True) as status:
            st.write("📋 **Đang tạo plan...**")

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
                chat_history=st.session_state.chat_history
            )

            final_state = None

            for event in st.session_state.agent_app.stream(initial_state):
                node_name = list(event.keys())[0]
                node_state = event[node_name]
                final_state = node_state  

                if node_name == "planner":
                    plan = node_state.get("plan", [])
                    st.write("📋 **Plan tạo xong:**")
                    for i, step in enumerate(plan):
                        st.write(f"  {i+1}. `{step}`")

                elif node_name == "executor":
                    current_step = node_state.get("current_step", 0)
                    plan = node_state.get("plan", [])
                    if current_step > 0 and current_step <= len(plan):
                        executed_step = plan[current_step - 1]
                        if "rag_search" in executed_step:
                            st.write(f"🔍 **{executed_step}**")
                        elif "extract_keypoints" in executed_step:
                            st.write(f"🧠 **{executed_step}**")
                        elif "compare_papers" in executed_step:
                            st.write(f"⚖️ **{executed_step}**")

                elif node_name == "synthesizer":
                    st.write("✍️ **Đang tổng hợp câu trả lời...**")

            status.update(label="Hoàn thành!", state="complete", expanded=False)

        # Lấy answer từ final_state đã stream xong
        answer = final_state.get("final_answer", "") if final_state else ""
        citations = final_state.get("citations", []) if final_state else []
        plan = final_state.get("plan", []) if final_state else []

        # Hiện answer
        st.markdown(answer)

        # Hiện citations
        if citations:
            with st.expander(f"📚 Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(
                        f"- [{c['title']}]({c['url']}) "
                        f"— {c['authors'][:50]}... ({c['year']})"
                    )
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer[:300] 
        })
        # Lưu vào history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "plan": plan
        })