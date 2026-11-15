
# Intelligent RAG - HR Help Desk

An AI-powered HR Help Desk chatbot for Zyro Dynamics Pvt. Ltd.

## Features
- 11 HR policy PDF documents
- Recursive text chunking
- Hugging Face sentence embeddings
- FAISS vector database
- MMR-based retrieval
- Groq LLM
- LCEL RAG chain
- Source citations
- Out-of-scope refusal

## Technology
Python, Streamlit, LangChain, FAISS,
Hugging Face and Groq.

## Run locally

Install requirements:

pip install -r requirements.txt

Configure GROQ_API_KEY in Streamlit secrets.

Start the application:

streamlit run app.py