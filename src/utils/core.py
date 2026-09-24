import json
import logging
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── Colored log formatter ───────────────────────────────────────────────

class ColoredFormatter(logging.Formatter):
    """ANSI-colored log formatter for terminal readability."""

    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG:    "\033[90m",        # grey
        logging.INFO:     "\033[92m",        # green
        logging.WARNING:  "\033[93m",        # yellow
        logging.ERROR:    "\033[91m",        # red
        logging.CRITICAL: "\033[91;1m",      # bold red
    }
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        ts = self.formatTime(record, self.datefmt)

        # Highlight STEP markers in bold cyan
        msg = record.getMessage()
        if "[STEP " in msg:
            msg = msg.replace("[STEP ", f"{self.BOLD}\033[96m[STEP ")
            msg = msg.replace("]", f"]{self.RESET}{color}", 1)

        return (
            f"{self.CYAN}{ts}{self.RESET} "
            f"{color}[{record.levelname}]{self.RESET} "
            f"{color}{msg}{self.RESET}"
        )


# Set up root logger with colored handler
_handler = logging.StreamHandler()
_handler.setFormatter(ColoredFormatter(datefmt="%H:%M:%S"))
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.resolve()
FRONTEND_DIR = BASE_DIR / "frontend"

# Database Configuration
POSTGRES_URL = "postgresql://rag_user:rag_password@localhost:5440/rag_db"
engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# CRAG singleton
_crag = None

def get_crag():
    global _crag
    if _crag is None:
        import sys
        if str(BASE_DIR) not in sys.path:
            sys.path.insert(0, str(BASE_DIR))
        import os
        from src.corrective_rag import CorrectiveRAG, CRAGConfig
        from src.generator import GeneratorConfig
        logger.info("Initialising CRAG pipeline...")
        _crag = CorrectiveRAG(
            CRAGConfig(
                collection_name="webapp_collection",
                generator_config=GeneratorConfig(model=os.environ.get("PRIMARY_LLM_MODEL", "qwen2.5:7b")),
            )
        )
        logger.info("CRAG pipeline ready.")
    return _crag

# DB Tables are initialized in main.py
