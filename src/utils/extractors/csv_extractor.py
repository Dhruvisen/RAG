import io
import pandas as pd
from .base import DocumentExtractor

class CSVExtractor(DocumentExtractor):
    def extract(self, content: bytes) -> str:
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df.to_markdown(index=False)
        except Exception as exc:
            raise RuntimeError(f"CSV read error: {exc}")
