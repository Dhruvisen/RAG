import json
import logging
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
FRONTEND_DIR = BASE_DIR / "frontend"

# Database Configuration
POSTGRES_URL = "postgresql://rag_user:rag_password@localhost:5440/rag_db"
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# CRAG singleton
_crag = None

def get_crag():
    global _crag
    if _crag is None:
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        from src.corrective_rag import CorrectiveRAG, CRAGConfig
        from src.generator import GeneratorConfig
        logger.info("Initialising CRAG pipeline...")
        _crag = CorrectiveRAG(
            CRAGConfig(
                collection_name="webapp_collection",
                generator_config=GeneratorConfig(model="qwen2.5:1.5b"),
            )
        )
        logger.info("CRAG pipeline ready.")
    return _crag

# DB Tables are initialized in main.py
