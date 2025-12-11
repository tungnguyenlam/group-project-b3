from typing import List
import re
import torch

from .LLMTranslator import LLMTranslator


DEFAULT_THINKING_SYS_PROMPT = """Translate Japanese to English.

Examples:
- ん? → Huh?
- ふん → Hmph
- うん → Yeah
- おはよう → Good morning
- ありがとう → Thank you
- 綴じ眼のシオラ → Siora of the Closed Eyes
- 朽鷹みつき → Mitsuki Kuchitaka

Output only the English translation, nothing else."""

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-Instruct"


class ThinkingLLMTranslator(LLMTranslator):
    def __init__(self):
        super().__init__()
        self.system_prompt: str = DEFAULT_THINKING_SYS_PROMPT
        self.max_new_tokens: int = 100
        self.default_model: str = DEFAULT_MODEL

    def _apply_chat_template(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return chat_prompt

    def _extract_answer(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = text.strip()

        if not text:
            return text

        for prefix in ["translation:", "output:", "english:"]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        text = text.replace("```", "").strip()

        return text

    def _generate(self, formatted_prompts: List[str]) -> List[str]:
        bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self.bad_phrases
            if self.tokenizer(p, add_special_tokens=False).input_ids
        ]

        chat_prompts = [self._apply_chat_template(prompt) for prompt in formatted_prompts]

        individual_lengths = []
        for prompt in chat_prompts:
            tokens = self.tokenizer(prompt, return_tensors="pt")
            individual_lengths.append(tokens["input_ids"].shape[1])

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids
            )

        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[individual_lengths[i]:],
                skip_special_tokens=True
            )
            response = self._extract_answer(response)
            responses.append(response)

        return responses
