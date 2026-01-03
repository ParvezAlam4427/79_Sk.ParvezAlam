import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os

# ---------- CONFIG ----------
DATA_PATH = "data/telecom_interactions.csv"
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# ----------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    texts = []

    for i, row in df.iterrows():
        combined_text = " ".join([str(v) for v in row.values if pd.notna(v)])
        texts.append(
            Document(
                page_content=combined_text,
                metadata={"row_id": i}
            )
        )
    return texts


def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    vectordb.persist()
    print("Vector store created and saved.")


if __name__ == "__main__":
    print("Loading data...")
    docs = load_data()

    print("Creating vector store...")
    create_vectorstore(docs)

    print("Ingestion completed successfully.")
