
# Enterprise RAG Knowledge Agent

Companies waste enormous time searching internal documents — HR policies, meeting notes, onboarding guides, security guidelines. Employees ask the same questions repeatedly, managers dig through PDFs manually, and knowledge stays locked in files nobody reads.
This agent solves that. Ask a question in plain English, get a precise answer with exact citations in seconds.

## 🎯 What It Does

This agent searches across 10 company policy documents and provides accurate answers with exact source citations (filename, page, lines). Employees can ask questions like:

- "What is the leave policy for sick days?"
- "How many sick days do employees get?"
- "What are the escalation levels for client complaints?"


# Documents parsed & User Interface

<img width="1901" height="1017" alt="Screenshot 2026-02-24 210849" src="https://github.com/user-attachments/assets/dafc9d00-953e-4f74-8fd1-7203c133637f" />

**Example Response:**
```
ANSWER
Employees are entitled to 12 days of paid sick leave per year.

RETRIEVED SOURCES<img width="1918" height="1001" alt="Screenshot 2026-02-24 211012" src="https://github.com/user-attachments/assets/4b0aab9b-b8c8-4225-982e-d91ee945a82f" />
<img width="1918" height="1001" alt="Screenshot 2026-02-24 211012" src="https://github.com/user-attachments/assets/4b0aab9b-b8c8-4225-982e-d91ee945a82f" />

[1] employee_leave_policy.pdf | Page 1 | Lines 18-22 | Similarity: 0.91
```

# Sample Response 1


<img width="1918" height="1001" alt="Screenshot 2026-02-24 211012" src="https://github.com/user-attachments/assets/30a1dea4-9ae7-439a-8047-20b440534f3f" />


# Sample Response 2


<img width="1919" height="993" alt="Screenshot 2026-02-24 214224" src="https://github.com/user-attachments/assets/e10198fe-f3a1-4e2c-950d-53aa74636a3e" />


# Sample Response 3


<img width="1919" height="1009" alt="Screenshot 2026-02-24 214601" src="https://github.com/user-attachments/assets/3dd73489-bf3b-484c-9754-cc5e2051367e" />


## 🚀 Features

- ✅ **Custom PDF Parser** - Line-by-line extraction with metadata (doc_id, page_num, line_start, line_end)
- ✅ **ChromaDB Vector Store** - Cosine similarity search with semantic chunking
- ✅ **Groq LLM Integration** - Fast Llama 3.3 70B for answer generation
- ✅ **Streamlit Chat UI** - Professional interface with source citations as expandable cards
- ✅ **Terminal Interface** - CLI alternative for agent.py
- ✅ **10 Sample NovaTech Documents took online** - Pre-loaded company policies as knowledge base

# Response along with the Document Details

<img width="1919" height="1005" alt="Screenshot 2026-02-24 211029" src="https://github.com/user-attachments/assets/0270ef1e-56f7-4a22-bae3-dda4420a1cc0" />

# Expanding chunks

<img width="1916" height="1016" alt="Screenshot 2026-02-24 211101" src="https://github.com/user-attachments/assets/0ef561e1-1dd0-4e77-bace-c6a733e19495" />


## 📁 Project Structure

```
enterprise-knowledge-agent/
├── docs/                          # 10 Sample NovaTech company policy PDFs
├── vectorstore/                   # ChromaDB embeddings (auto-generated)
├── core.py                        # Shared agent logic (single source of truth)
│   ├── load_components()         # Load embedder, collection, groq_client
│   ├── retrieve()                # Search ChromaDB, return top 5 chunks
│   ├── build_prompt()            # Format context + question for LLM
│   └── generate_answer()         # Call Groq API, return response
├── agent.py                       # Terminal interface
│   └── format_response()         # Terminal-specific output formatting
├── app.py                         # Streamlit UI (imports from core.py)
├── meta_parser.py                # Custom PDF p<img width="1919" height="1005" alt="Screenshot 2026-02-24 211029" src="https://github.com/user-attachments/assets/3324d7ab-6732-433c-8c16-6ce083835a8d" />
<img width="1919" height="1005" alt="Screenshot 2026-02-24 211029" src="https://github.com/user-attachments/assets/3324d7ab-6732-433c-8c16-6ce083835a8d" />
arser with line metadata
├── emb_ingest.py                 # Embedding + ChromaDB ingestion pipeline
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq (Llama 3.3 70B) | Answer generation |
| **Vector DB** | ChromaDB | Similarity search |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors |
| **PDF Parser** | pdfplumber | Line-by-line text extraction |
| **UI** | Streamlit | Chat interface |
| **Language** | Python 3.11+ | Core logic |

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Groq API key ([get one free](https://console.groq.com))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/enterprise-RAG-knowledge-Agent.git
cd enterprise-RAG-knowledge-Agent
```

2. **Create virtual environment**
```bash
python -m venv solvenv
source solvenv/bin/activate  # On Windows: solvenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API key**

Create a `.env` file in the project root:
```bash
GROQ_API_KEY=your_key_here
```

Or load it in code (already configured in `core.py`):
```python
from dotenv import load_dotenv
import os
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

5. **Run ingestion pipeline** (first time only)
```bash
python emb_ingest.py
```

This will:
- Parse all PDFs in `/docs`
- Generate embeddings
- Store in ChromaDB at `/vectorstore`

## 🎮 Usage

### Option 1: Streamlit UI (Recommended)
```bash
streamlit run app.py
```
Visit `http://localhost:8501` and start chatting!

### Option 2: Terminal Interface
```bash
python agent.py
```
Type questions and get answers in the terminal. Type `exit`, `quit`, or `q` to exit.

## 📝 How It Works

### 1. **Ingestion** (`emb_ingest.py`)
```
PDFs in /docs → meta_parser.parse() → chunks with metadata 
→ embed with all-MiniLM-L6-v2 → store in ChromaDB
```

### 2. **Retrieval** (`core.retrieve()`)
```
User question → embed → ChromaDB similarity search 
→ top 5 chunks (0.4+ similarity) → return with metadata
```

### 3. **Generation** (`core.generate_answer()`)
```
Retrieved chunks → format as context → build prompt 
→ send to Groq Llama 3.3 70B → return answer
```

### 4. **Formatting** (`core.build_prompt()` + `app.py`/`agent.py`)
```
LLM response → parse → display answer + source citations 
→ show filename, page, lines, similarity score
```

## 🧪 Example Questions

Try asking:
- "What is the sick leave policy?"
- "How do I submit an expense reimbursement?"
- "What are the escalation levels for client complaints?"
- "Can I love my HR?"
- "What are the benefits in the employee package?"

The agent will respond with answers and show exactly where it found the information.

## 🔧 Configuration

### Tuning Parameters (in `core.py`)

```python
# Chunk size & overlap
CHUNK_SIZE = 5 lines
OVERLAP = 4 lines

# Retrieval
TOP_K = 5  # Number of chunks to retrieve
MIN_SIMILARITY = 0.4  # Minimum similarity threshold

# LLM
MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.1  # Low = factual, High = creative
```

## 📊 Knowledge Base

The `/docs` folder contains 10 NovaTech company policy documents:

1. Employee Leave Policy
2. Expense Reimbursement Policy
3. Client Escalation Process
4. Onboarding Guide
5. Remote Work Policy
6. Performance Review Process
7. Data Security Guidelines
8. Project Handover Template
9. Vendor Management Policy
10. Q3 Product Meeting Notes

## 🤝 Contributing

Contributions welcome! Please follow this workflow:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -m "feat: add new feature"`
3. Push to GitHub: `git push -u origin feature/your-feature`
4. Create a Pull Request to `dev` branch
5. After review, merge and delete branch

### Commit Convention
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code restructuring
- `docs:` - Documentation
- `chore:` - Setup, config, maintenance

## 📄 License

MIT License - feel free to use this for your own projects!

## Acknowledgments

- **ChromaDB** - Fast local vector database
- **Groq** - Blazing fast LLM inference
- **Streamlit** - Rapid UI development
- **HuggingFace** - Embedding models

---

Built with ❤️ for enterprise knowledge management

**Questions?** Open an issue or reach out!
