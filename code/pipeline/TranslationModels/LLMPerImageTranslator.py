from typing import List

from .LLMTranslator import LLMTranslator


DEFAULT_SYS_PROMPT = """You are a manga translator. You translate Japanese dialogue bubbles to natural English.

You will receive a manga page with numbered dialogue bubbles in reading order (right-to-left, top-to-bottom).
One bubble is marked with [TRANSLATE THIS] - translate ONLY that bubble.
Use the surrounding context to ensure the translation flows naturally with the conversation.

Output format: Only the English translation, nothing else.

Example:
---
Page context:
[1] おい、待ってくれ！
[2] [TRANSLATE THIS] 何だよ、急に…
[3] 大事な話があるんだ

Translation: What is it, all of a sudden...
---

Keep translations:
- Natural and conversational
- Appropriate to the tone (casual, formal, dramatic)
- Consistent with surrounding dialogue context
"""

DEFAULT_BAD_PHRASES = [
    "I can't", "I cannot", "I'm sorry", "I am sorry",
    "As an AI", "cannot help", "unable to", "I won't", "I will not",
    "I cannot comply", "I cannot assist", "Translation:", "Here is"
]


def format_input_with_context(source_texts: List[str], target_index: int) -> str:
    lines = ["Page context:"]
    for i, text in enumerate(source_texts):
        if i == target_index:
            lines.append(f"[{i+1}] [TRANSLATE THIS] {text}")
        else:
            lines.append(f"[{i+1}] {text}")
    lines.append("\nTranslation:")
    return "\n".join(lines)


def format_input_batch(source_texts: List[str]) -> List[str]:
    return [format_input_with_context(source_texts, i) for i in range(len(source_texts))]


class LLMPerImageTranslator(LLMTranslator):
    def __init__(self):
        super().__init__()
        self.system_prompt: str = DEFAULT_SYS_PROMPT
        self.bad_phrases: List[str] = DEFAULT_BAD_PHRASES
        self.use_batch: bool = False

    def _translate_single(self, source_texts: List[str], target_index: int) -> str:
        formatted = format_input_with_context(source_texts, target_index)
        result = self._generate([formatted])
        return result[0]

    def _translate_batch(self, source_texts: List[str]) -> List[str]:
        if not source_texts:
            return []
        formatted = format_input_batch(source_texts)
        return self._generate(formatted)

    def _translate(self, texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.use_batch:
            return self._translate_batch(texts)
        else:
            return [self._translate_single(texts, i) for i in range(len(texts))]
