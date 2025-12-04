from typing import List, Union

from .LLMTranslator import LLMTranslator

# Per-image specific system prompt for context-aware manga translation
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
    """
    Format input for a single bubble with full page context.
    
    Args:
        source_texts: List of all Japanese texts from the page (in bubble order)
        target_index: Index of the bubble to translate (0-indexed)
        
    Returns:
        Formatted prompt string with context and marked target
    """
    lines = ["Page context:"]
    for i, text in enumerate(source_texts):
        if i == target_index:
            lines.append(f"[{i+1}] [TRANSLATE THIS] {text}")
        else:
            lines.append(f"[{i+1}] {text}")
    lines.append("\nTranslation:")
    return "\n".join(lines)


def format_input_batch(source_texts: List[str]) -> List[str]:
    """
    Format inputs for all bubbles in a page, each with full context.
    
    Args:
        source_texts: List of all Japanese texts from the page (in bubble order)
        
    Returns:
        List of formatted prompts, one per bubble
    """
    return [format_input_with_context(source_texts, i) for i in range(len(source_texts))]


class LLMPerImageTranslator(LLMTranslator):
    """
    LLM Translator that translates manga bubbles with page context.
    
    Unlike the base LLMTranslator which translates texts independently,
    this class provides context from all bubbles on a page when translating
    each individual bubble, improving translation coherence.
    """
    
    def __init__(self):
        super().__init__()
        # Override defaults specific to per-image translation
        self.system_prompt: str = DEFAULT_SYS_PROMPT
        self.bad_phrases: List[str] = DEFAULT_BAD_PHRASES
        # Default to single for memory safety on MPS (context prompts are larger)
        self.use_batch: bool = False

    def predict_single(self, source_texts: List[str], target_index: int) -> str:
        """
        Translate a single bubble with page context.
        
        Args:
            source_texts: List of all Japanese texts from the page
            target_index: Index of the bubble to translate
            
        Returns:
            English translation for the target bubble
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        formatted = format_input_with_context(source_texts, target_index)
        # Use parent's _generate with a single-item list
        result = self._generate([formatted])
        return result[0]

    def batch_predict(self, source_texts: List[str]) -> List[str]:
        """
        Translate all bubbles from a page in one batch.
        
        Args:
            source_texts: List of all Japanese texts from the page
            
        Returns:
            List of English translations for all bubbles
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not source_texts:
            return []
            
        formatted = format_input_batch(source_texts)
        return self._generate(formatted)

    def predict(self, source_texts: Union[str, List[str]]) -> List[str]:
        """
        Main predict method used by evaluators.
        Behavior controlled by self.use_batch flag.
        
        Args:
            source_texts: List of Japanese texts from one page
            
        Returns:
            List of translations (same length as source_texts)
        """
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        
        if self.use_batch:
            return self.batch_predict(source_texts)
        else:
            # Translate one by one (safer for MPS memory)
            return [self.predict_single(source_texts, i) for i in range(len(source_texts))]