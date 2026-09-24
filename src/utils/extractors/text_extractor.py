from .base import DocumentExtractor
from src.utils.core import logger

class TextExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        logger.info("Extracting data from Text document.")
        return content.decode("utf-8", errors="replace")
