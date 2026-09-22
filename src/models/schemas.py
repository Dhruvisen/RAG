from pydantic import BaseModel

class DocumentRecord(BaseModel):
    doc_id:      str
    filename:    str
    size_bytes:  int
    uploaded_at: str
    file_type:   str

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status:     str
    model_ready: bool
    doc_count:  int
