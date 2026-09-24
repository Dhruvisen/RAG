import io
import pandas as pd
from .base import DocumentExtractor
from src.utils.core import logger

class CSVExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        logger.info("Extracting data from CSV document.")
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_markdown(index=False)
        except Exception as exc:
            raise RuntimeError(f"CSV read error: {exc}")
