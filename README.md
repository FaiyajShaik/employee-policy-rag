# Employee Policy RAG

A Retrieval-Augmented Generation application for answering employee policy questions using ChromaDB, Sentence Transformers, Gemini, and Streamlit.

## Overview

Employee Policy RAG is a document-based question-answering application that allows users to ask questions about an employee handbook.

The application retrieves relevant information from the handbook and uses Gemini to generate answers based on the retrieved content.

## How It Works

```text
Employee Handbook
       ↓
Document Preprocessing
       ↓
Document Chunking
       ↓
Sentence Transformers
       ↓
Embeddings
       ↓
ChromaDB
       ↓
User Question
       ↓
Semantic Retrieval
       ↓
Relevant Handbook Sections
       ↓
Gemini
       ↓
Grounded Answer
       ↓
Streamlit UI
```

## Features

- Semantic search over employee handbook content
- ChromaDB vector database
- Sentence Transformers embeddings
- Gemini-powered answer generation
- Retrieval relevance filtering
- Conversational follow-up questions
- Question rewriting for follow-up queries
- Out-of-scope question rejection
- Source section display
- Streamlit web interface
- Environment variable based API-key management

## Example Questions

- How many PTO days do employees receive?
- How many sick-leave days are provided?
- Can employees work remotely?
- What are the remote-work requirements?
- How many days per week can they work remotely?
- When does remote-work eligibility begin?

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database and retrieval |
| Gemini | Answer generation |
| Streamlit | Web interface |
| python-dotenv | Environment variable management |

## Project Structure

```text
employee-policy-rag/
│
├── data/
│   └── rag_source_employee_handbook.txt
│
├── src/
│   ├── chunk_document.py
│   ├── create_embeddings.py
│   ├── generate_answer.py
│   ├── index_document.py
│   ├── load_document.py
│   ├── preprocess_document.py
│   ├── retrieve_document.py
│   └── section_chunk_document.py
│
├── app.py
├── ui.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/FaiyajShaik/employee-policy-rag.git
cd employee-policy-rag
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not upload the `.env` file to GitHub.

## Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Example Conversation

The application supports follow-up questions using conversation context.

```text
User: Can employees work remotely?

User: What are the requirements?

User: How many days per week can they do it?
```

The application uses the previous conversation context to understand follow-up questions.

## Retrieval and Grounding

The application uses semantic similarity search to retrieve relevant sections from the employee handbook.

Retrieved results are filtered using relevance thresholds before being passed to Gemini.

If relevant information cannot be found in the handbook, the application rejects the question instead of generating an answer from unrelated information.

## Current Scope

The current version focuses on the fundamental RAG pipeline:

- Document ingestion
- Preprocessing
- Chunking
- Embedding generation
- Vector indexing
- Semantic retrieval
- Relevance filtering
- Grounded generation
- Conversational context
- Streamlit interface

## Future Improvements

- Advanced chunking strategies
- Reranking
- Hybrid search
- RAG evaluation
- Retrieval quality evaluation
- Response evaluation
- Advanced RAG pipelines
- Agentic RAG
- Multi-agent systems
- Production deployment
- Monitoring and observability

## License

This project is currently provided for educational and portfolio purposes.