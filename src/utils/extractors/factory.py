from .base import DocumentExtractor
from .pdf_extractor import PDFExtractor
from .docx_extractor import DocxExtractor
from .text_extractor import TextExtractor
from .csv_extractor import CSVExtractor
from .excel_extractor import ExcelExtractor

class ExtractorFactory:
    @staticmethod
    def get_extractor(extension: str) -> DocumentExtractor:
        ext = extension.lower().strip()
        if not ext.startswith("."):
            ext = f".{ext}"
            
        if ext == ".pdf":
            return PDFExtractor()
        elif ext == ".docx":
            return DocxExtractor()
        elif ext == ".csv":
            return CSVExtractor()
        elif ext in [".xlsx", ".xls"]:
            return ExcelExtractor()
        else:
            # Default to text extractor for .txt, .md, etc.
            return TextExtractor()
