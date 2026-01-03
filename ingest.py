import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_PATH = "data/telecom_interactions.csv"
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_data():
    """
    Loads telecom interaction data from CSV file.
    Returns a list of Document objects.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    docs = []

    for i, row in df.iterrows():
        text = " ".join([str(v) for v in row.values if pd.notna(v)])
        docs.append(
            Document(
                page_content=text,
                metadata={"row_id": i}
            )
        )

    return docs

def create_vectorstore(docs):
    """
    Splits documents into chunks and creates a Chroma vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    
    # Create and persist the vector store
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )
    
if __name__ == "__main__":
    import os
    
    print("📥 Loading telecom interaction data...")
    try:
        docs = load_data()
        
        print("📦 Creating vector store (ChromaDB)...")
        create_vectorstore(docs)
        
        print("✅ Ingestion completed successfully!")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
