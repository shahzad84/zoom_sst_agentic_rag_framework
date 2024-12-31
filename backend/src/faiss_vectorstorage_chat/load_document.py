import os
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()
global stored_documents

stored_documents=[]

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free and open-source embedding model

def load_documents(file_path: str):
    """
    Load and split documents into chunks.
    """
    loader = PyPDFLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    pages = loader.load_and_split()
    docs = text_splitter.split_documents(pages)

    return docs

def create_faiss_index(docs, index_name="vector_store"):
    """
    Create a FAISS index from document chunks.
    """
    # Generate embeddings for document chunks
    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index
    faiss.write_index(index, f"{index_name}.faiss")
    return index
