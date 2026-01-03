from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import google.generativeai as genai

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv(override=True)

VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ESCALATION_KEYWORDS = ["refund", "cancel", "complaint", "legal", "fraud"]

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-flash-latest")

app = FastAPI(title="AI Customer Service Agent (Gemini)")

@app.get("/")
def root():
    return {"message": "AI Customer Service Agent running"}

embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class QueryRequest(BaseModel):
    query: str

def should_escalate(query: str) -> bool:
    return any(k in query.lower() for k in ESCALATION_KEYWORDS)

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        docs = retriever.invoke(request.query)

        if not docs:
            return {
                "answer": "No relevant data found. Escalating.",
                "sources": [],
                "escalate": True
            }

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a telecom customer support assistant.

Context:
{context}

Question:
{request.query}

Answer clearly and professionally.
"""

        response = model.generate_content(prompt)

        sources = [f"row_{doc.metadata.get('row_id')}" for doc in docs]

        return {
            "answer": response.text,
            "sources": sources,
            "escalate": should_escalate(request.query)
        }
    except Exception as e:
        return {
            "answer": "I'm sorry, I encountered an error processing your request.",
            "sources": [],
            "escalate": True,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
