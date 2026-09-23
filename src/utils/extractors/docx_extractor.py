import io
from docx import Document as _Docx
from .base import DocumentExtractor

class DocxExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        try:
            d = _Docx(io.BytesIO(content))
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as exc:
            raise RuntimeError(f"DOCX read error: {exc}")
