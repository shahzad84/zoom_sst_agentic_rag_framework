from pydantic import BaseModel
class PromptRequest(BaseModel):
    prompt: str


class ChatRequest(BaseModel):
    user_id: str
    message: str



from typing import List,Dict,Any

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):  # Fixed the typo in the class name
    response: str
    retrieved_docs: List[Dict[str, Any]]  # Changed List[str] to List[Dict[str, Any]]
    session_id: str



# steps to implement:
"""Get text embedding using Hugging Face API"""
"""Load and split PDF document into chunks"""
"""Store document chunks with embeddings in MongoDB"""
"""Perform vector search using MongoDB Atlas"""
"""Generate answer using Hugging Face Generation API"""