# 🏢 Enterprise RAG Knowledge Agent

> An AI-powered knowledge assistant that answers employee questions from company documents with **precise citations** — filename, page number, and line range.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-green)
![Groq](https://img.shields.io/badge/Groq-Llama3.3_70B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

Companies waste enormous time searching internal documents — HR policies, meeting notes, onboarding guides, security guidelines. Employees ask the same questions repeatedly, managers dig through PDFs manually, and knowledge stays locked in files nobody reads.

This agent solves that. Ask a question in plain English, get a precise answer with exact citations in seconds.

---

## ✨ Features

- **Line-level citations** — every answer cites exact filename, page, and line range
- **Custom PDF parser** — extracts text line-by-line preserving positional metadata
- **Semantic search** — finds relevant chunks even when exact words don't match
- **Hallucination prevention** — agent only answers from retrieved documents, never makes things up
- **Chat interface** — conversational UI with persistent history
- **Source transparency** — expandable source viewer with similarity scores per chunk
- **Color-coded confidence** — 🟢 high / 🟡 medium / 🔴 low similarity indicators

---

## 🏗️ Architecture

```
User Question
      ↓
  Embed question (all-MiniLM-L6-v2)
      ↓
  Search ChromaDB (cosine similarity)
      ↓
  Retrieve top 5 chunks with metadata
      ↓
  Build prompt with context + citations
      ↓
  Groq LLM (Llama 3.3 70B) generates answer
      ↓
  Return answer + precise citations
```

---

## 🗂️ Project Structure

```
enterprise-RAG-knowledge-Agent/
│
├── core.py              ← Shared agent logic (single source of truth)
├── meta_parser.py       ← Custom line-by-line PDF parser with metadata
├── emb_ingest.py        ← Embedding + ChromaDB ingestion pipeline
├── agent.py             ← Terminal chat interface
├── app.py               ← Streamlit web UI
│
├── docs/                ← Company PDF documents (knowledge base)
│   ├── employee_leave_policy.pdf
│   ├── expense_reimbursement_policy.pdf
│   ├── client_escalation_process.pdf
│   ├── onboarding_guide.pdf
│   ├── remote_work_policy.pdf
│   ├── performance_review_process.pdf
│   ├── data_security_guidelines.pdf
│   ├── project_handover_template.pdf
│   ├── vendor_management_policy.pdf
│   └── q3_product_meeting_notes.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| PDF Parsing | pdfplumber | Line-by-line text extraction |
| Embeddings | all-MiniLM-L6-v2 | Convert text to vectors (384 dim) |
| Vector Store | ChromaDB | Cosine similarity search |
| LLM | Groq / Llama 3.3 70B | Answer generation |
| UI | Streamlit | Chat web interface |
| Environment | python-dotenv | API key management |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/themaheswar1/enterprise-RAG-knowledge-Agent.git
cd enterprise-RAG-knowledge-Agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Create a .env file in project root:
GROQ_API_KEY=your_groq_api_key_here

# 5. Run ingestion pipeline (builds vector store)
python emb_ingest.py

# 6. Launch the app
streamlit run app.py
```

---

## 💬 Usage

**Web UI:**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Terminal:**
```bash
python agent.py
```

**Example questions:**
- *"How many sick days do employees get per year?"*
- *"What was decided in the Q3 product meeting?"*
- *"What is the hotel expense limit for domestic travel?"*
- *"What are the escalation levels for client complaints?"*
- *"What security requirements apply to remote workers?"*

---

## 🔍 How It Works

### 1. Custom PDF Parser (`meta_parser.py`)
Unlike standard loaders that discard positional data, our parser extracts every line with:
- `doc_id` — unique document identifier
- `page_num` — page number
- `line_num` — global line number across document
- `page_line_num` — line number within the page

### 2. Chunking Strategy
Lines are chunked with a sliding window (`chunk_size=20, overlap=4`). Overlap ensures context at chunk boundaries is never lost.

### 3. Vector Search
Questions are embedded using the same model as documents. ChromaDB finds the top 5 most semantically similar chunks using cosine similarity.

### 4. Answer Generation
Retrieved chunks are injected into a carefully crafted prompt. The LLM answers using **only** the provided context — if the answer isn't there, it says so.

---

## 📁 Adding Your Own Documents

1. Place PDF files in the `docs/` folder
2. Re-run the ingestion pipeline:
```bash
python emb_ingest.py
```
3. Start asking questions — your documents are now searchable

---

## 🤝 Contributing

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: describe your change"

# Push and open a Pull Request
git push origin feature/your-feature-name
```

Branch flow: `feature/* → dev → main`

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Maheswar** — [@themaheswar1](https://github.com/themaheswar1)

> Built as part of a portfolio project to demonstrate production-quality AI engineering skills including custom RAG pipelines, vector databases, LLM integration, and professional Git workflows.
