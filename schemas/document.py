from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from database.models import DocumentStatus

class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_size: int
    file_type: str

class DocumentCreate(DocumentBase):
    file_path: str

class DocumentResponse(DocumentBase):
    id: int
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DocumentStatusResponse(BaseModel):
    id: int
    status: DocumentStatus
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

class DocumentQuery(BaseModel):
    doc_id: int
    query: str
    top_k: Optional[int] = 5

class DocumentQueryResponse(BaseModel):
    doc_id: int
    query: str
    results: List[dict]
    total_results: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int

class UploadResponse(BaseModel):
    message: str
    document_id: int
    status: str
