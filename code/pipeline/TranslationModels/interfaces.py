from abc import ABC, abstractmethod
from typing import List

class TranslationModel(ABC):
    @abstractmethod
    def predict(self, source_texts: List[str]) -> List[str]:
        """
        Takes a list of source strings, handles tokenization internally,
        runs inference, and returns a list of translated strings.
        """
        pass
