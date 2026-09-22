import logging
import warnings
from contextlib import asynccontextmanager

# ── Suppress noisy third-party loggers ──────────────────────────────────
for _noisy in (
    "httpx", "httpcore", "huggingface_hub", "huggingface_hub.file_download",
    "sentence_transformers", "transformers", "nltk",
    "urllib3.connectionpool", "chromadb", "opentelemetry",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Suppress the NLTK data warning
warnings.filterwarnings("ignore", category=UserWarning, module="nltk")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.utils.core import FRONTEND_DIR, get_crag, logger, engine
from src.controller import health, documents, query
from src.models.db import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    Base.metadata.create_all(bind=engine)
    try:
        get_crag()
    except Exception as exc:
        logger.warning("Deferred CRAG init (%s). Will retry on first request.", exc)
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(title="Corrective RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])
app.include_router(query.router, prefix="/api", tags=["Query"])


# Serve frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    target = FRONTEND_DIR / path
    if target.exists() and target.is_file():
        return FileResponse(str(target))
    return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

