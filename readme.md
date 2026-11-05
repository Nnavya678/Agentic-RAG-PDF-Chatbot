# Agentic RAG-Based PDF Chatbot

An interactive Generative AI application that allows users to upload PDF documents and ask questions based on their content using Retrieval-Augmented Generation (RAG).

## Features

- Multiple PDF document uploads using Streamlit.
- Document loading and text chunking using LangChain.
- Semantic embeddings using Hugging Face (`all-MiniLM-L6-v2`).
- Vector storage and similarity-based retrieval using InMemoryVectorStore.
- Agent-based retrieval using LangChain tools.
- LLM integration using Groq (`openai/gpt-oss-20b`).
- Conversational memory using LangGraph checkpointing.
- Interactive question-answering interface.

## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Groq
- Hugging Face Embeddings
- InMemoryVectorStore

## Architecture

PDF Upload → Document Loading → Text Chunking → Embedding Generation → Vector Store → Agentic Retrieval → LLM Response

## Installation

1. Clone the repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   cd Agentic-RAG-PDF-Chatbot
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the environment:

   Windows:
   ```bash
   venv\Scripts\activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file and add your Groq API key:

   ```env
   GROQ_API_KEY=your_api_key
   ```

6. Run the application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. Launch the Streamlit application.
2. Upload one or more PDF documents.
3. Wait for document processing.
4. Ask questions related to the uploaded documents.
5. Receive context-aware answers using the RAG pipeline.