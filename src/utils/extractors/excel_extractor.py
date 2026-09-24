import io
import pandas as pd
from .base import DocumentExtractor
from src.utils.core import logger

class ExcelExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        logger.info("Extracting data from Excel document.")
        try:
            dfs = pd.read_excel(io.BytesIO(content), sheet_name=None)
            md_parts = []
            for sheet, df in dfs.items():
                md_parts.append(f"### Sheet: {sheet}")
                md_parts.append(df.to_markdown(index=False))
            return "\n\n".join(md_parts)
        except Exception as exc:
            raise RuntimeError(f"Excel read error: {exc}")
