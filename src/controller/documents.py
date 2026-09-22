import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from src.models.schemas import DocumentRecord
from src.models.db import Document
from src.utils.storage import upload_to_minio, get_from_minio, delete_from_minio
from src.utils.core import get_crag, logger, get_db

router = APIRouter()

@router.post("/documents", response_model=DocumentRecord, status_code=201)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a file and ingest it into the RAG pipeline."""
    allowed = {".pdf", ".txt", ".md", ".docx"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(415, f"Unsupported type '{suffix}'. Allowed: {', '.join(allowed)}")

    content = await file.read()
    
    # Hash content to prevent duplicates in production
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    doc_id = file_hash

    # Check DB
    existing = db.query(Document).filter(Document.doc_id == doc_id).first()
    if existing:
        logger.info("Document '%s' already ingested (hash match). Skipping.", file.filename)
        return DocumentRecord(
            doc_id=existing.doc_id,
            filename=existing.filename,
            size_bytes=existing.size_bytes,
            uploaded_at=existing.uploaded_at.isoformat(),
            file_type=existing.file_type
        )

    # Upload to MinIO
    object_name = f"{doc_id}{suffix}"
    if not upload_to_minio(object_name, content):
        raise HTTPException(500, "Failed to store raw file in MinIO.")

    # Build doc dict for RAG ingestion
    doc: Dict[str, Any] = {"id": doc_id}
    if suffix == ".pdf":
        import tempfile
        # Write temporarily for pdfplumber if needed
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        doc["pdf_path"] = tmp_path
    elif suffix == ".docx":
        try:
            from docx import Document as _Docx
            d = _Docx(io.BytesIO(content))
            doc["text"] = "\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as exc:
            delete_from_minio(object_name)
            raise HTTPException(500, f"DOCX read error: {exc}")
    else:
        doc["text"] = content.decode("utf-8", errors="replace")

    try:
        get_crag().ingest(doc)
    except Exception as exc:
        delete_from_minio(object_name)
        raise HTTPException(500, f"Ingestion failed: {exc}")
        
    # Cleanup temp pdf
    if suffix == ".pdf" and "tmp_path" in locals():
        Path(tmp_path).unlink(missing_ok=True)

    db_doc = Document(
        doc_id=doc_id,
        filename=file.filename,
        size_bytes=len(content),
        file_type=suffix.lstrip(".").upper(),
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    record = DocumentRecord(
        doc_id=db_doc.doc_id,
        filename=db_doc.filename,
        size_bytes=db_doc.size_bytes,
        uploaded_at=db_doc.uploaded_at.isoformat(),
        file_type=db_doc.file_type,
    )
    logger.info("Ingested '%s' as %s.", file.filename, doc_id)
    return record


@router.get("/documents", response_model=List[DocumentRecord])
async def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [
        DocumentRecord(
            doc_id=d.doc_id,
            filename=d.filename,
            size_bytes=d.size_bytes,
            uploaded_at=d.uploaded_at.isoformat(),
            file_type=d.file_type,
        ) for d in docs
    ]


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    
    ext = doc.file_type.lower()
    delete_from_minio(f"{doc_id}.{ext}")

    # Remove chunks from vector store (ChromaDB)
    try:
        crag = get_crag()
        collection = crag._rag.vector_store.collection
        # Find all chunk IDs belonging to this document
        existing = collection.get(where={"document_id": doc_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info("Deleted %d chunks from vector store for doc %s.", len(existing["ids"]), doc_id)
    except Exception as exc:
        logger.warning("Failed to delete chunks from vector store for doc %s: %s", doc_id, exc)
    
    db.delete(doc)
    db.commit()
    logger.info("Deleted document %s.", doc_id)


@router.get("/documents/{doc_id}/content")
async def get_document_content(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(404, f"Document '{doc_id}' not found.")
    
    ext = doc.file_type.lower()
    content = get_from_minio(f"{doc_id}.{ext}")
    if not content:
        raise HTTPException(404, f"File content for '{doc_id}' not found in MinIO.")
    
    media_type = "application/pdf" if ext == "pdf" else "text/plain"
    return StreamingResponse(io.BytesIO(content), media_type=media_type)
