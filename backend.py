import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
import zipfile
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load LLM Model (faster one recommended) ---
model_name = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)


# --- Global Vector Database (filled after upload) ---
vectordb = None


# =============== 1) FILE UPLOAD ENDPOINT =================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectordb

    # Cleanup older codebase
    if os.path.exists("codebase"):
        shutil.rmtree("codebase")

    os.makedirs("codebase", exist_ok=True)

    # Save ZIP file
    zip_path = "uploaded.zip"
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("codebase")

    # Load code
    loader = DirectoryLoader(
        "codebase",
        glob="**/*",
        loader_cls=TextLoader
    )
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    # Build Vector Database
    vectordb = Chroma.from_documents(
        chunks, embedding, persist_directory="db", collection_name="codebase"
    )

    vectordb.persist()

    return {"message": "Project processed successfully!"}


# =============== 2) ASK / QUERY ENDPOINT =================

@app.get("/ask")
async def ask(query: str):
    global vectordb

    if vectordb is None:
        return {"answer": "❌ Please upload codebase first!"}

    # Retrieve similar chunks
    docs = vectordb.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a code expert. Answer based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return {"answer": answer}
