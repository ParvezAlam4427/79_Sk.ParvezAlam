# AI Customer Service Agent (Gemini)

This project is an AI-powered customer service agent for a telecom company. It uses Google's Gemini model (Gemini Flash) for generating responses and ChromaDB for vector search (RAG - Retrieval Augmented Generation).

## Features
- Retrieves relevant customer interaction history from a CSV dataset.
- Uses RAG (Retrieval Augmented Generation) to answer customer queries.
- Escalates queries based on keywords (e.g., "refund", "legal").
- Powered by Google Gemini and LangChain.

## Prerequisites
- Python 3.10+
- Google Cloud API Key (for Gemini)

## Setup

1.  **Clone the repository** (if not already done).

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables**:
    Create a `.env` file in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

## Usage

### 1. Ingest Data
Before running the app, you need to ingest the data into the vector store.
```bash
python ingest.py
```
This will read `data/telecom_interactions.csv`, chunk the text, generate embeddings, and store them in `vectorstore/`.

### 2. Run the Application
Start the FastAPI server:
```bash
python app.py
```
The server will run at `http://0.0.0.0:8000`.

### 3. Test the Agent
You can test the agent using the provided test script:
```bash
python test_gemini.py
```
Or send a POST request to the API:
```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"query\": \"I want to port out\"}"
```

## API Endpoints
- `GET /`: Health check.
- `POST /ask`: Ask a question. Body: `{"query": "your question"}`.

## Troubleshooting
- **API Key Error**: Ensure your `GOOGLE_API_KEY` is valid and has access to Gemini API.
- **ModuleNotFoundError**: Ensure you have installed all requirements and activated the virtual environment.
