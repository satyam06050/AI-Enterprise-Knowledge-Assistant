# AI Enterprise Knowledge Assistant

A RAG-powered multi-agent AI system that lets employees ask natural-language questions about internal company documents. Built with Python, Streamlit, Groq, FAISS, Sentence Transformers, and LangGraph.

## What It Does

Employees can ask questions like:
- *"What is the leave policy?"*
- *"How do I deploy the application?"*
- *"Who owns Project Alpha?"*

The system routes each question to the right specialist agent, retrieves relevant document chunks via semantic search, and generates a grounded answer with source citations.

## Architecture

```
User
 │
 ▼
Streamlit Chat UI
 │
 ▼
Manager Agent  ──── classifies question into HR / TECHNICAL / PROJECT / GENERAL
 │
 ▼
FAISS Vector Search  ──── retrieves relevant chunks from department documents
 │
 ▼
Specialist Agent  ──── generates grounded answer using only retrieved context
 │
 ▼
Answer + Source Citation
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Chat UI |
| Groq (`openai/gpt-oss-120b`) | LLM |
| Sentence Transformers (`all-MiniLM-L6-v2`) | Embeddings |
| FAISS | Vector search |
| LangGraph | Agent workflow |
| PyPDF | PDF loading |
| Python-dotenv | Environment config |

## Project Structure

```
AI_Enterprise_Assistant/
├── app.py              # Streamlit UI
├── config.py           # API key + model config
├── rag.py              # Document loading, chunking, embeddings, retrieval
├── agents.py           # Manager agent + answer generation
├── graph.py            # LangGraph workflow
├── requirements.txt
├── .env
└── documents/
    ├── hr/             # HR policy PDFs
    ├── technical/      # Technical guide PDFs
    └── projects/       # Project documentation PDFs
```

## Setup

**1. Clone and create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your key from [console.groq.com](https://console.groq.com).

**4. Add documents**

Place PDF or TXT files into the appropriate department folder:
```
documents/hr/           ← HR policies
documents/technical/    ← Technical guides
documents/projects/     ← Project documentation
```

**5. Run**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Usage

1. The app automatically loads documents on startup
2. Use the **Load Documents** button in the sidebar to reload after adding new files
3. Type your question in the chat input
4. The Manager Agent classifies the question and routes it to the right specialist
5. The answer includes the source document and page number

## Running Tests

```bash
pytest tests/test_rag.py
```

Tests use `.txt` files so no PDFs are required.

## How It Works

1. **Document Loading** — PDFs and TXT files are loaded from `documents/` subfolders, each tagged with their department
2. **Chunking** — Documents are split into 700-character chunks with 100-character overlap
3. **Embeddings** — Each chunk is embedded using `all-MiniLM-L6-v2`
4. **FAISS Index** — Embeddings are stored in a FAISS `IndexFlatIP` (inner product / cosine similarity)
5. **Manager Agent** — Classifies the question into HR, TECHNICAL, PROJECT, or GENERAL using the LLM
6. **Retrieval** — Top-5 chunks are retrieved, filtered by department
7. **Answer Generation** — A specialist agent generates a grounded answer using only the retrieved context
8. **Hallucination Control** — If the documents don't contain the answer, the agent responds: *"I could not find this information in the available company documents."*
