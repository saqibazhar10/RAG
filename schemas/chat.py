from uuid import UUID
from pydantic import BaseModel
from typing import List, Optional

class LLMRequest(BaseModel):
    doc_id: Optional[List]=None
    query: str
    conversation_id: Optional[UUID] = None
    conv_title: Optional[str] = "Untitled Conversation"
    agent_id: Optional[str]= ""

class LLMQueryResponse(BaseModel):
    doc_id: int
    query: str
    Response: str
    chunks_used: List[dict]

class Message(BaseModel):
    id: str
    conv_id: str
    message: str

class AgentCreateReq(BaseModel):
    name:str
    description:str
    instructions:str
    creativity:float