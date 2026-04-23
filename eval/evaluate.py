"""
eval/evaluate.py
Đánh giá 3 tầng: Retrieval, Agent (Plan), Answer Quality
"""

import sys
import os
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retriever import search
from agent.graph import run_agent
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


TEST_CASES = [
    {
        "query": "What is a Convolutional Neural Network (CNN)?",
        "expected_keywords": ["convolution", "filter", "feature", "pooling", "image"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "How does CNN work for image classification?",
        "expected_keywords": ["cnn", "kernel", "feature map", "stride", "pooling"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "What is a Recurrent Neural Network (RNN)?",
        "expected_keywords": ["rnn", "sequence", "hidden state", "time step", "recurrent"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "What is the vanishing gradient problem in RNN?",
        "expected_keywords": ["vanishing", "gradient", "long term", "dependency", "rnn"],
        "expected_plan_tools": ["rag_search", "extract_keypoints"],
        "query_type": "limitation",
    },
    {
        "query": "What is LSTM and how does it work?",
        "expected_keywords": ["lstm", "cell state", "gate", "forget", "memory"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "Difference between LSTM and RNN",
        "expected_keywords": ["lstm", "rnn", "long term", "memory", "gate"],
        "expected_plan_tools": ["rag_search", "compare_papers"],
        "query_type": "comparison",
    },
    {
        "query": "What is attention mechanism in deep learning?",
        "expected_keywords": ["attention", "query", "key", "value", "transformer"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "Explain Transformer architecture",
        "expected_keywords": ["transformer", "self attention", "encoder", "decoder", "multi head"],
        "expected_plan_tools": ["rag_search"],
        "query_type": "concept",
    },
    {
        "query": "What are limitations of BERT?",
        "expected_keywords": ["bert", "pretrain", "mask", "context", "limitation"],
        "expected_plan_tools": ["rag_search", "extract_keypoints"],
        "query_type": "limitation",
    },
    
    
]


# Retrieval Eval
def evaluate_retrieval(test_case: dict, top_k: int = 5) -> dict:
    query = test_case["query"]
    expected_keywords = test_case["expected_keywords"]

    results = search(query, top_k=top_k)

    if not results:
        return {
            "query": query,
            "num_results": 0,
            "avg_score": 0.0,
            "keyword_hit_rate": 0.0,
            "keyword_hits": "0/0",
            "status": "[FAIL]",
        }

    all_text = " ".join(
        [(r.get("title", "") + " " + r.get("text", "")).lower() for r in results]
    )

    hits = sum(1 for kw in expected_keywords if kw.lower() in all_text)
    keyword_hit_rate = hits / len(expected_keywords)
    avg_score = sum(r["score"] for r in results) / len(results)

    status = "[PASS]" if keyword_hit_rate >= 0.4 else "[FAIL]"

    return {
        "query": query,
        "num_results": len(results),
        "avg_score": round(avg_score, 4),
        "keyword_hit_rate": round(keyword_hit_rate, 4),
        "keyword_hits": f"{hits}/{len(expected_keywords)}",
        "top_paper": results[0]["title"] if results else "",
        "status": status,
    }


# Plan Eval
def evaluate_plan(plan: list, test_case: dict) -> dict:
    expected_tools = test_case["expected_plan_tools"]
    query_type = test_case["query_type"]

    has_rag_search = any("rag_search" in step for step in plan)
    has_synthesize = "synthesize" in plan
    plan_length = len(plan)

    tool_hits = [any(tool in step for step in plan) for tool in expected_tools]
    tool_coverage = sum(tool_hits) / len(expected_tools)

    status = (
        "[PASS]"
        if has_rag_search and has_synthesize and tool_coverage >= 0.6
        else "[FAIL]"
    )

    return {
        "plan": plan,
        "plan_length": plan_length,
        "has_rag_search": has_rag_search,
        "has_synthesize": has_synthesize,
        "tool_coverage": round(tool_coverage, 4),
        "status": status,
    }


# Answer Eval
def evaluate_answer(answer: str, citations: list, test_case: dict) -> dict:
    query = test_case["query"]
    expected_keywords = test_case["expected_keywords"]

    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_coverage = hits / len(expected_keywords)

    has_citations = len(citations) > 0
    answer_length = len(answer)

    judge_prompt = f"""
Question: {query}

Answer:
{answer[:1500]}

Rate 1-5:
Return JSON: {{"score": 1-5, "reason": "..."}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "")
        judge = json.loads(raw)

        llm_score = judge.get("score", 0)
        llm_reason = judge.get("reason", "")

    except Exception as e:
        llm_score = 0
        llm_reason = str(e)

    status = "[PASS]" if llm_score >= 4 else "[FAIL]"

    return {
        "answer_length": answer_length,
        "keyword_coverage": round(keyword_coverage, 4),
        "keyword_hits": f"{hits}/{len(expected_keywords)}",
        "has_citations": has_citations,
        "num_citations": len(citations),
        "llm_score": llm_score,
        "llm_reason": llm_reason,
        "status": status,
    }


# Runner
def run_evaluation():
    print("PAPER RESEARCH AGENT EVALUATION")

    retrieval_passes = 0
    plan_passes = 0
    answer_passes = 0
    all_results = []

    for i, test_case in enumerate(TEST_CASES):
        query = test_case["query"]
        print(f"\n[{i+1}/{len(TEST_CASES)}] {query}")

        print("Retrieval...")
        retrieval = evaluate_retrieval(test_case)
        print(f"{retrieval['status']} avg_score={retrieval['avg_score']} keywords={retrieval['keyword_hits']}")

        if retrieval["status"] == "[PASS]":
            retrieval_passes += 1

        print("Agent running...")
        try:
            t0 = time.time()
            agent_result = run_agent(query)
            elapsed = round(time.time() - t0, 2)

            plan = agent_result.get("plan", [])
            answer = agent_result.get("answer", "")
            citations = agent_result.get("citations", [])

            plan_eval = evaluate_plan(plan, test_case)
            print(f"{plan_eval['status']} plan_len={plan_eval['plan_length']} tool_cov={plan_eval['tool_coverage']}")

            if plan_eval["status"] == "[PASS]":
                plan_passes += 1

            answer_eval = evaluate_answer(answer, citations, test_case)
            print(f"{answer_eval['status']} llm_score={answer_eval['llm_score']} citations={answer_eval['num_citations']} time={elapsed}s")
            all_results.append({
                "query": query,
                "retrieval": retrieval,
                "plan": plan_eval,
                "answer": answer_eval,
                "time": elapsed
            })

            if answer_eval["status"] == "[PASS]":
                answer_passes += 1

        except Exception as e:
            print("[FAIL] agent error:", e)

    n = len(TEST_CASES)

    print("\nSUMMARY")
    print(f"Retrieval: {retrieval_passes}/{n}")
    print(f"Plan: {plan_passes}/{n}")
    print(f"Answer: {answer_passes}/{n}")

    overall = (retrieval_passes + plan_passes + answer_passes) / (n * 3)
    print(f"Overall: {round(overall * 100)}%")

    os.makedirs("eval", exist_ok=True)
    with open("eval/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_evaluation()