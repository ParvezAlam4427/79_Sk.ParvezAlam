from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ---------------- CONFIG ----------------
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ESCALATION_KEYWORDS = [
    "refund", "cancel", "complaint", "legal", "fraud", "chargeback"
]
# ----------------------------------------

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY loaded:", bool(GOOGLE_API_KEY))

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Initialize FastAPI
app = FastAPI(title="AI Customer Service Agent")

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL
)

# Load vector store
vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Request schema
class QueryRequest(BaseModel):
    query: str

# Escalation logic
def should_escalate(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ESCALATION_KEYWORDS)

# API endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    # 1️⃣ Retrieve relevant documents
    docs = retriever.get_relevant_documents(request.query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # 2️⃣ Build prompt
    prompt = f"""
You are a telecom customer support assistant.

Use the following past support interactions to answer the question.

Context:
{context}

Question:
{request.query}

Answer clearly and professionally.
"""

    # 3️⃣ Call Gemini directly
    response = model.generate_content(prompt)

    # 4️⃣ Collect sources
    sources = []
    for i, doc in enumerate(docs):
        row_id = doc.metadata.get("row_id", i)
        sources.append(f"row_{row_id}")

    return {
        "answer": response.text,
        "sources": sources,
        "escalate": should_escalate(request.query)
    }
