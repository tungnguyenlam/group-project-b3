import re
from abc import ABC, abstractmethod
from typing import List, Union, Tuple


class Translator(ABC):
    def __init__(self):
        self.skip_gating: bool = False

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration: {key}")
        return self

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def unload_model(self):
        pass

    @staticmethod
    def contains_japanese(text: str) -> bool:
        pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        return bool(pattern.search(text))

    def _gate(self, texts: List[str]) -> Tuple[List[str], List[int], List[int]]:
        if self.skip_gating:
            return texts, [], list(range(len(texts)))

        texts_to_translate = []
        skip_indices = []
        translate_indices = []

        for i, text in enumerate(texts):
            if self.contains_japanese(text):
                translate_indices.append(i)
                texts_to_translate.append(text)
            else:
                skip_indices.append(i)

        return texts_to_translate, skip_indices, translate_indices

    def preprocess(self, texts: List[str]) -> List[str]:
        return texts

    def postprocess(self, texts: List[str]) -> List[str]:
        return texts

    @abstractmethod
    def _translate(self, texts: List[str]) -> List[str]:
        pass

    def predict(self, source_texts: Union[str, List[str]]) -> Union[str, List[str]]:
        single_input = isinstance(source_texts, str)
        if single_input:
            source_texts = [source_texts]

        texts_to_translate, skip_indices, translate_indices = self._gate(source_texts)
        results = list(source_texts)

        if texts_to_translate:
            preprocessed = self.preprocess(texts_to_translate)
            translated = self._translate(preprocessed)
            postprocessed = self.postprocess(translated)

            for idx, text in zip(translate_indices, postprocessed):
                results[idx] = text

        return results[0] if single_input else results
