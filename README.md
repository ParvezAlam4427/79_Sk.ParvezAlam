## 📌 Problem Statement
T1 – AI Customer Service Agent (RAG over Tickets & Dialogues)
---
## Overview
Telecom customer support teams handle a high volume of repetitive queries related to network issues, billing, data speed, and SIM services. This project presents a **Retrieval-Augmented Generation (RAG)** based AI assistant that answers common customer queries using historical telecom support interactions and escalates sensitive issues when required.

---

## Solution
The system retrieves relevant past support tickets and dialogues, uses them as context for response generation, and returns both the generated answer and the source document references. A lightweight rule-based escalation mechanism flags high-risk queries for human review.
---

## System Architecture
1. Data ingestion from historical telecom support interactions  
2. Text chunking for efficient retrieval  
3. Sentence-level embeddings for semantic search  
4. Vector similarity search using ChromaDB  
5. Retrieval-Augmented Generation via LangChain  
6. Rule-based escalation logic  
7. REST API interface using FastAPI  

---

## Dataset
- Telecom Agent–Customer Interaction Text  
  https://www.kaggle.com/datasets/avinashok/telecomagentcustomerinteractiontext  

---

## Technology Stack
- Python 3.10+
- LangChain
- ChromaDB
- SentenceTransformers
- FastAPI
- Uvicorn

---

## API Interface

### Endpoint
