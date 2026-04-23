from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    # Câu hỏi gốc của user
    query: str
    
    # Lịch sử chat
    messages: Annotated[list, operator.add]
    
    # Plan do Planner tạo ra
    plan: list[str]
     
    # Bước hiện tại trong plan
    current_step: int  
    
    # Kết quả RAG search(raw)
    context: list[dict]  
    
    # Papers đã tìm được
    selected_papers: list[dict]
    
    # Kết quả trung gian của từng step
    intermediate_results: Annotated[list, operator.add]

    # Câu trả lời cuối cùng
    final_answer: str

    # Citations để hiện trên UI
    citations: list[dict]
    
    chat_history: list[dict]