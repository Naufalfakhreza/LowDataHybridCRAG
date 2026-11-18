# LowDataHybridCRAG: Asisten Akademik FMIPA UNJ 2026

An intelligent, local Corrective Retrieval-Augmented Generation (CRAG) system designed for low-data, dense tabular documents. Tailored specifically for the **Buku Pedoman Akademik (BPA) Fakultas MIPA Universitas Negeri Jakarta 2026/2027**.

---

## 🌟 Key Features

- **Hybrid Lexical & Dense Retrieval:** Combines **Okapi BM25** with dense embeddings (**`intfloat/multilingual-e5-small`**) using **Reciprocal Rank Fusion (RRF)**.
- **HyDE (Hypothetical Document Embeddings):** Leverages a local LLM via Ollama to generate zero-shot hypothetical BPA text snippets, boosting dense retrieval precision for ambiguous questions.
- **CRAG Confidence Evaluation:** A Cross-Encoder (**`BAAI/bge-reranker-v2-m3`**) dynamically evaluates retrieved chunks:
  - `CORRECT` ($\ge 0.55$): Answers directly with refined factual strips.
  - `AMBIGUOUS` ($0.20 - 0.55$): Refines knowledge strips and provides fallback search keywords.
  - `INCORRECT` ($< 0.20$): Honestly abstains from guessing to eliminate hallucinations.
- **Table-Aware Strip Refinement:** Preserves structured Markdown table rows and per-row verbalized summaries while stripping irrelevant noise.
- **Real-Time Token Streaming:** Dynamic streaming responses in both the Streamlit Web UI and CLI.
- **Dual Interfaces:**
  - 🖥️ **Modern Web Chat (Streamlit):** Visual confidence badges, collapsible citations/evidence, and study program filters.
  - ⌨️ **Interactive Terminal CLI:** Fast terminal-based conversation loop.

---

## 🏗️ Architecture

```
User Query
    │
    ├───► BM25 Lexical Retrieval ────────────┐
    │                                        ▼
    └───► HyDE Generation ──► Multilingual-E5 Dense ──► Reciprocal Rank Fusion (RRF)
                                                            │
                                                            ▼ Top-K Candidates
                                                    Cross-Encoder Evaluator
                                                     (bge-reranker-v2-m3)
                                                            │
                            ┌───────────────────────────────┼───────────────────────────────┐
                            ▼                               ▼                               ▼
                     [CORRECT >= 0.55]              [AMBIGUOUS 0.20-0.55]           [INCORRECT < 0.20]
                            │                               │                               │
                            ▼                               ▼                               ▼
                 Refine Knowledge Strips         Refine Strips + Fallback Query      Abstain / Refuse
                            │                               │
                            └───────────────────────────────┴───────────────► Local Ollama (qwen2.5:7b)
                                                                                    │
                                                                                    ▼ Streamed Response
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)

Pull the target language model:
```bash
ollama pull qwen2.5:7b
```

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/LowDataHybridCRAG.git
cd LowDataHybridCRAG
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Copy the example environment file and configure your keys:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here   # Optional: only for dataset creation & PDF conversion
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

---

## 💬 Usage

### Option 1: Streamlit Web Chatbot (Recommended)
Launch the interactive web application:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Interactive Terminal Chat (CLI)
Start the terminal chat loop:
```bash
python RAG.py
```
Special commands in CLI:
- `/prodi <nama>` — Lock filter to a specific study program (e.g. `/prodi ilmu komputer`).
- `/prodi auto` — Enable automatic program detection from query text.
- `/prodi clear` — Search across all FMIPA documents.
- `/k <number>` — Change retrieval depth (default: `10`).
- `keluar` or `exit` — Exit session.

---

## 📊 Evaluation & Benchmark
To run the multi-percentile confidence benchmark against the RAFT evaluation dataset:
```python
from RAG import BPAHybridCRAG

rag = BPAHybridCRAG("bpa_corpus_chunks.jsonl")
rag.benchmark_percentiles("bpa_eval_raft_dataset.jsonl", top_k=10)
```

---

## 📄 License
MIT License.
