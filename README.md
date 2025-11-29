# 🚀 RAG Code Helper – AI Codebase Question Answering

This project is a **RAG (Retrieval-Augmented Generation)** system that lets you upload your **entire project as a ZIP file** and ask questions about your code.  
The backend processes your files, creates embeddings, stores them in ChromaDB, and uses a local LLM to answer questions **based only on your codebase**.

---

## 🧠 Features

- 📁 Upload ZIP file containing your project  
- 🔍 Extracts & reads all code/text files  
- ✂ Splits content into semantic chunks  
- 🧠 Creates embeddings using MiniLM  
- 🗂 Stores vectors in ChromaDB  
- 🤖 Answers code questions using local LLM  
- 🌐 Works with Ngrok so you can access it from a browser  
- ⚡ Clean and simple HTML + JavaScript frontend  

---

## 📦 Tech Stack

**Backend**
- FastAPI  
- ChromaDB  
- LangChain  
- Sentence Transformers  
- HuggingFace LLMs (OPT / Phi / etc.)  
- PyTorch  

**Frontend**
- HTML  
- TailwindCSS  
- JavaScript (Fetch API)

---

## 📂 Workflow

1. User uploads `.zip` through frontend  
2. Backend extracts `.zip`  
3. Loads code/text files  
4. Splits them into chunks  
5. Generates embeddings  
6. Saves to ChromaDB  
7. User asks a question  
8. System retrieves relevant code chunks  
9. LLM generates an answer

📄 Supported File Types
.py, .js, .ts, .java, .cpp, .c, .cs,
.html, .css, .json, .md, .txt
