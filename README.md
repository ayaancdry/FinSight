# FinRAG: Financial Document Assistant & Sentiment Analyzer

A portfolio project demonstrating RAG (Retrieval-Augmented Generation) and NLP techniques for financial document analysis.

## Features

- **PDF Document Ingestion**: Upload and process SEC 10-K filings and other financial documents
- **RAG-Powered Chat**: Ask questions about documents and get context-aware answers
- **Sentiment Analysis**: Analyze document sentiment using FinBERT (finance-specific BERT)
- **Interactive Dashboard**: Visualize sentiment distribution with Plotly charts

## Tech Stack

- **Python 3.9+**
- **Streamlit** - Web UI
- **LangChain** - RAG orchestration
- **FAISS** - Vector store
- **Sentence Transformers** - Embeddings (all-MiniLM-L6-v2)
- **FinBERT** - Financial sentiment analysis
- **Plotly** - Data visualization

## Project Structure

```
finrag/
├── app.py                 # Streamlit entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── src/
    ├── __init__.py
    ├── config.py         # Configuration management
    ├── data_loader.py    # PDF ingestion & chunking
    ├── rag_engine.py     # RAG with FAISS + LangChain
    └── analytics.py      # FinBERT sentiment analysis
```

## Setup & Installation

### 1. Create Virtual Environment (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Usage

1. **Enter API Key**: Input your OpenAI API key in the sidebar (or set via .env)
2. **Upload Document**: Use the sidebar to upload a PDF (e.g., SEC 10-K filing)
3. **Chat Tab**: Ask questions like "What are the primary liquidity risks?"
4. **Sentiment Tab**: Click "Analyze Document Sentiment" to view sentiment distribution

## Example Questions

- "What are the company's main risk factors?"
- "Summarize the management discussion section"
- "What is the company's revenue for the fiscal year?"
- "Describe the company's competitive landscape"

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PDF Upload │────▶│ DocumentLoader│────▶│   Chunks    │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────────────────┬┴──────────────────────────┐
                    ▼                          ▼                           ▼
             ┌─────────────┐           ┌──────────────┐           ┌───────────────┐
             │  RAG Engine │           │   Embeddings │           │ SentimentAnalyzer│
             │  (LangChain)│           │   (MiniLM)   │           │   (FinBERT)   │
             └──────┬──────┘           └──────┬───────┘           └───────┬───────┘
                    │                         │                           │
                    ▼                         ▼                           ▼
             ┌─────────────┐           ┌─────────────┐           ┌───────────────┐
             │   OpenAI    │           │    FAISS    │           │  Sentiment    │
             │   LLM       │           │ Vector Store│           │  Scores       │
             └─────────────┘           └─────────────┘           └───────────────┘
```

## Author

Built as a portfolio project for self growth.
Ayaan Choudhury

