from typing import List
import requests,os
from fastapi import HTTPException

EMBEDDING_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}



def get_embedding(text: str) -> List[float]:
    """Get text embedding using Hugging Face API"""
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            headers=HEADERS,
            json={"inputs": text, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding API error: {str(e)}")

  
    
# from src.mongodb_vector_store_chat_llm.mongodb_vector_search import mongodb_vector_search
# query = "what is poona college?"  # Replace with an actual query
# embedding = get_embedding(query)

# if embedding:
#     print("Embedding received successfully:", embedding[:5])  # Print first few values
# else:
#     print("Embedding function returned None or an error")

# results = mongodb_vector_search(query, k=5)
# print("Search Results:", results)
