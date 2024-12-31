from pydantic import BaseModel
from typing import List
class QueryRequest(BaseModel):
    query: str
    index_name: str = "vector_store"
    k: int = 10


class QueryResponse(BaseModel):
    response: str
    retrieved_docs: List[str]