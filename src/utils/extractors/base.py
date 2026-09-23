from abc import ABC, abstractmethod

class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, content: bytes) -> str:
        """
        Extract text from the given file content bytes.
        Returns the parsed text/markdown.
        """
        pass
