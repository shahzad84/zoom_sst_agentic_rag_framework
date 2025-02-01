from src.mongodb_vector_store_chat_llm.embedding import get_embedding

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



def mongodb_vector_search(query: str, k: int = 5):
    """Perform vector search using MongoDB Atlas"""
    query_embedding = get_embedding(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k*10,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
                "metadata": 1
            }
        }
    ]
    
    results = collection.aggregate(pipeline)
    # print(list(results))
    return list(results)


