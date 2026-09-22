from fastapi import APIRouter, HTTPException
from src.models.schemas import QueryRequest, QueryResponse
from src.utils.core import get_crag, logger

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    try:
        result = get_crag().query(req.question)
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Pipeline error: {exc}")
    return QueryResponse(answer=result.answer)
