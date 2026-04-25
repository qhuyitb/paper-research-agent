# Paper Research Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-0A66C2)

Agent hỗ trợ tra cứu và tổng hợp kiến thức từ các AI/ML papers (nguồn ArXiv) bằng RAG + LangGraph + OpenAI, có giao diện chat Streamlit.
## 🚀 Live Demo
👉 [Paper Research Agent](https://qhuyitb-paper-research-agent-uiapp-sqqsy8.streamlit.app/)

## Nguồn dữ liệu đầy đủ

- Full dataset ArXiv (Kaggle): https://www.kaggle.com/datasets/Cornell-University/arxiv
- File chính hệ thống đang dùng để lọc dữ liệu: `data/raw/arxiv-metadata-oai-snapshot.json`

## 1. Tổng quan

Dự án gồm 3 khối chính:

- Data pipeline: lọc paper, semantic chunking, embedding.
- Retrieval layer: lưu vector vào Qdrant, tìm kiếm paper liên quan.
- Agent layer: planner -> executor -> synthesizer để tạo câu trả lời có trích dẫn.

Luồng xử lý tổng quát:

1. User đặt câu hỏi trên UI.
2. `planner` tạo plan theo query.
3. `executor` chạy các bước `rag_search`, `extract_keypoints`, `compare_papers`.
4. `synthesizer` tổng hợp câu trả lời cuối + citations.

## 2. Tính năng chính

- Chat hỏi đáp paper AI/ML theo ngữ cảnh retrieval.
- Hỗ trợ query kiểu:
  - Giải thích concept (ví dụ: attention mechanism).
  - So sánh papers/phương pháp (ví dụ: Transformer vs RNN).
  - Phân tích hạn chế (limitations) của paper.
- Có theo dõi trace các step agent trên UI.
- Hiển thị citations kèm tiêu đề, tác giả, năm, link paper.

## 3. Cấu trúc thư mục

```text
paper-research-agent/
|- .env                     # Biến môi trường (OPENAI_API_KEY, QDRANT_*)
|- README.md
|- requirements.txt
|- agent/
|  |- graph.py              # LangGraph workflow (planner/executor/synthesizer)
|  |- planner.py            # Tạo kế hoạch tool-use từ query
|  |- tools.py              # Tool wrappers: rag_search, extract_keypoints, compare_papers
|  |- state.py              # Typed state cho toàn bộ graph
|- data/
|  |- filter_data.py        # Lọc ArXiv metadata theo category + năm
|  |- preprocess.py         # Semantic chunking cho title + abstract
|  |- raw/
|  |  |- arxiv-metadata-oai-snapshot.json
|  |  |- papers.json
|  |- processed/
|  |  |- papers_processed.json
|  |  |- embeddings.npy
|- rag/
|  |- embedder.py           # Encode chunks thành vector (all-MiniLM-L6-v2)
|  |- vector_store.py       # Tạo collection và upsert points vào Qdrant
|  |- retriever.py          # Semantic search + format context
|- ui/
|  |- app.py                # Giao diện chat Streamlit
|- eval/
|  |- evaluate.py           # Script đánh giá Retrieval / Plan / Answer
|  |- eval_results.json     # Kết quả evaluation (sinh ra sau khi chạy)
|- qdrant_storage/          # Dữ liệu local Qdrant (nếu chạy local persistent)
|- venv/                    # Python virtual environment
```

## 4. Yêu cầu hệ thống

- Python 3.10+ (khuyến nghị 3.10/3.11).
- Qdrant (local hoặc cloud).
- OpenAI API key.

## 5. Cài đặt

### Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate
```

### Cài dependencies

```bash
pip install -r requirements.txt
```

## 6. Cấu hình biến môi trường

Tạo file `.env` tại root dự án:

```env
OPENAI_API_KEY=your_openai_api_key

# Nếu dùng Qdrant Cloud
QDRANT_URL=https://xxxx.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key

# Nếu dùng Qdrant local thì có thể bỏ trống 2 biến trên
```

Lưu ý:

- Nếu `QDRANT_URL` và `QDRANT_API_KEY` không được set, code sẽ mặc định kết nối `localhost:6333`.
- `OPENAI_API_KEY` là bắt buộc cho planner/synthesizer/tools.

## 7. Chạy dự án

Có 2 cách:

### Cách A: Dùng dữ liệu/index sẵn có (nhanh nhất)

Nếu đã có `data/processed/papers_processed.json`, `data/processed/embeddings.npy` và collection trong Qdrant:

```bash
streamlit run ui/app.py
```

### Cách B: Build lại pipeline từ đầu

#### B1. Lọc dữ liệu paper từ file snapshot

```bash
python data/filter_data.py
```

#### B2. Chunking semantic

```bash
python data/preprocess.py
```

#### B3. Tạo embeddings

```bash
python rag/embedder.py
```

#### B4. Index vào Qdrant

```bash
python rag/vector_store.py
```

#### B5. Chạy UI

```bash
streamlit run ui/app.py
```

Mặc định Streamlit sẽ mở trên `http://localhost:8501`.

## 8. Test nhanh từng thành phần

```bash
python rag/retriever.py     # test retrieval
python agent/planner.py     # test plan creation
python agent/graph.py       # test run_agent với sample queries
```

## 9. Evaluation

Project hiện có script đánh giá tại `eval/evaluate.py`.

Đánh giá theo 3 tầng:

- Retrieval: đo độ phủ keyword trên top-k kết quả truy xuất.
- Plan: kiểm tra plan có dùng đúng tools kỳ vọng và có bước `synthesize`.
- Answer: chấm keyword coverage + số citation + LLM judge score (1-5).

Chạy evaluation:

```bash
python eval/evaluate.py
```

Kết quả được ghi vào:

- `eval/eval_results.json`

Lưu ý khi chạy:

- Cần có `OPENAI_API_KEY` để chấm phần answer bằng LLM judge.
- Qdrant phải sẵn dữ liệu (đã index) để retrieval evaluation có ý nghĩa.

## 10. Một số query gợi ý

- `What is attention mechanism?`
- `Compare Transformer and RNN`
- `What are limitations of BERT?`

## 11. Troubleshooting

### Lỗi kết nối Qdrant

- Kiểm tra Qdrant đã chạy chưa (nếu local: port `6333`).
- Kiểm tra lại `QDRANT_URL`/`QDRANT_API_KEY` nếu dùng cloud.

### Lỗi OpenAI auth

- Kiểm tra `OPENAI_API_KEY` trong `.env`.
- Đảm bảo đã `source venv/bin/activate` trước khi chạy.

### Lần đầu chạy chậm

- `SentenceTransformer` sẽ tải model `all-MiniLM-L6-v2` ở lần đầu.
- Quá trình embedding/index có thể tốn thời gian tùy số lượng papers.

## 12. Ghi chú kỹ thuật

- Embedding model: `all-MiniLM-L6-v2` (384 dimensions).
- Qdrant distance metric: `COSINE`.
- Agent orchestrator: `LangGraph` (`planner -> executor loop -> synthesizer`).
- LLM model trong code hiện tại: `gpt-4o-mini`.


