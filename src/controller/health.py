from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.models.schemas import HealthResponse
from src.models.db import Document
from src.utils.core import get_db
from src.utils.storage import get_s3_client

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health(db: Session = Depends(get_db)):
    import requests as req
    model_ready = False
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=3)
        model_ready = r.status_code == 200
    except Exception:
        pass
        
    doc_count = db.query(Document).count()
    
    return HealthResponse(
        status="ok",
        model_ready=model_ready,
        doc_count=doc_count,
    )
