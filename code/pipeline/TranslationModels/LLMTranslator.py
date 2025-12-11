from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import torch
import gc

from .Translator import Translator


DEFAULT_SYS_PROMPT = """Translate Japanese to English. Examples:

Casual dialogue:
Japanese: おはよう！元気？
English: Good morning! How are you?

Formal speech:
Japanese: 本日はお忙しいところ、ありがとうございます。
English: Thank you for taking time out of your busy schedule today.

Action/dramatic:
Japanese: 逃げるな！戦え！
English: Don't run away! Fight!

Emotional:
Japanese: 信じられない...本当にそうなの？
English: I can't believe it... Is that really true?

Question:
Japanese: どうしてそんなことをしたの？
English: Why did you do that?

Statement:
Japanese: 明日は試験があるから、勉強しなければならない。
English: I have an exam tomorrow, so I need to study.

"""

DEFAULT_BAD_PHRASES = [
    "I can't", "I cannot", "I'm sorry", "I am sorry",
    "As an AI", "cannot help", "unable to", "I won't", "I will not",
    "I cannot comply", "I cannot assist"
]


def format_input(source_texts: List[str], add_begin: str = "Japanese: ", add_ending: str = "\nEnglish: ") -> List[str]:
    return [add_begin + item + add_ending for item in source_texts]


class LLMTranslator(Translator):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None

        self.max_new_tokens: int = 100
        self.temperature: float = 1.0
        self.top_p: float = 1.0
        self.do_sample: bool = False
        self.system_prompt: str = DEFAULT_SYS_PROMPT
        self.bad_phrases: List[str] = DEFAULT_BAD_PHRASES
        self.use_batch: bool = True
        self.skip_gating: bool = True

    def load_model(self, model_name="Qwen/Qwen2.5-0.5B", device='auto'):
        if self.model is not None:
            print("Model is already loaded")
            return

        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self._clear_memory()

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_name))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True
        )

        if self.model is None or self.tokenizer is None:
            raise TypeError("Error: No model loaded")

    def _clear_memory(self):
        if self.device and 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        elif self.device and 'mps' in str(self.device):
            torch.mps.empty_cache()
        gc.collect()

    def _generate(self, formatted_prompts: List[str]) -> List[str]:
        bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in self.bad_phrases
            if self.tokenizer(p, add_special_tokens=False).input_ids
        ]

        individual_lengths = []
        for prompt in formatted_prompts:
            tokens = self.tokenizer(prompt, return_tensors="pt")
            individual_lengths.append(tokens["input_ids"].shape[1])

        inputs = self.tokenizer(
            formatted_prompts,
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
            responses.append(response.strip())

        return responses

    def _translate_single(self, text: str) -> str:
        formatted = format_input([text])
        result = self._generate(formatted)
        return result[0]

    def _translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        formatted = format_input(texts)
        return self._generate(formatted)

    def _translate(self, texts: List[str]) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if self.use_batch:
            return self._translate_batch(texts)
        else:
            return [self._translate_single(text) for text in texts]

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._clear_memory()
        print("Model unloaded")
