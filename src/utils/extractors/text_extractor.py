from .base import DocumentExtractor

class TextExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        return content.decode("utf-8", errors="replace")
