from src.mongodb_vector_store_chat_llm.embedding import get_embedding
from datetime import datetime
# MongoDB components
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os

# MongoDB Configuration
MONGO_URI = os.getenv("MONGODB_ATLAS_URI")
DB_NAME = "chat_with_docs"
COLLECTION_NAME = "document_chunks"

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def store_chunks_in_mongodb(docs, document_source):
    """Store document chunks with embeddings in MongoDB"""
    records = []
    for doc in docs:
        record = {
            "text": doc.page_content,
            "embedding": get_embedding(doc.page_content),
            "document_source": document_source,
            "created_at": datetime.utcnow(),
            "metadata": doc.metadata
        }
        records.append(record)
    
    result = collection.insert_many(records)
    return result.inserted_ids
