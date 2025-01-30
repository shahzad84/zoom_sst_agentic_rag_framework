from pydantic import BaseModel
class PromptRequest(BaseModel):
    prompt: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
