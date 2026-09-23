import os
import tempfile
from pathlib import Path
import pdfplumber
from .base import DocumentExtractor

class PDFExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        # Write temporarily for pdfplumber
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            return self._extract_pdf_with_tables(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _extract_pdf_with_tables(self, pdf_path: str) -> str:
        """Extract text and tables from a PDF, formatting tables as markdown."""
        all_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            print(f"--- Extracting from PDF: {os.path.basename(pdf_path)} ---")
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if tables:
                    print(f"Page {page_num}: Found {len(tables)} tables")
                    table_bboxes = [t.bbox for t in (page.find_tables() or [])]

                    non_table_text = self._extract_non_table_text(page, table_bboxes)
                    if non_table_text.strip():
                        all_parts.append(non_table_text.strip())

                    for t_idx, table in enumerate(tables, 1):
                        md_table = self._table_to_markdown(table)
                        if md_table:
                            print(f"  Table {t_idx} extracted (first 100 chars): {md_table[:100].replace(chr(10), ' ')}")
                            all_parts.append(md_table)
                else:
                    text = page.extract_text() or ""
                    if text.strip():
                        all_parts.append(text.strip())

        return "\n\n".join(all_parts)

    def _extract_non_table_text(self, page, table_bboxes) -> str:
        text = ""
        last_bottom = 0
        for bbox in sorted(table_bboxes, key=lambda x: x[1]):
            top, bottom = bbox[1], bbox[3]
            if top > last_bottom:
                crop = page.crop((0, last_bottom, page.width, top))
                t = crop.extract_text()
                if t: text += t + "\n"
            last_bottom = bottom
            
        if last_bottom < page.height:
            crop = page.crop((0, last_bottom, page.width, page.height))
            t = crop.extract_text()
            if t: text += t + "\n"
            
        return text

    def _table_to_markdown(self, table) -> str:
        if not table or not table[0]:
            return ""
        
        md = []
        headers = [str(h).replace('\n', ' ') if h else "" for h in table[0]]
        md.append("| " + " | ".join(headers) + " |")
        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        for row in table[1:]:
            cleaned_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
            # Pad row if it has fewer columns than headers
            cleaned_row += [""] * (len(headers) - len(cleaned_row))
            md.append("| " + " | ".join(cleaned_row) + " |")
            
        return "\n".join(md)
