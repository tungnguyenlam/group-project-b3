from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Optional, Callable
from functools import partial
import torch
import gc

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


class LLMTranslator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Default predict arguments (can be configured before evaluation)
        self.max_new_tokens: int = 100
        self.temperature: float = 1.0
        self.top_p: float = 1.0
        self.do_sample: bool = False  # False = greedy decoding (more stable), True = sampling
        self.system_prompt: str = DEFAULT_SYS_PROMPT
        self.bad_phrases: List[str] = DEFAULT_BAD_PHRASES
        
        # Control whether predict uses batch or single mode
        self.use_batch: bool = True

    def configure(self, **kwargs):
        """
        Configure predict arguments before evaluation.
        
        Usage:
            model.configure(max_new_tokens=50, temperature=0.7, use_batch=False)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration: {key}")
        return self

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
        """Clear GPU/MPS memory."""
        if self.device and 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        elif self.device and 'mps' in str(self.device):
            torch.mps.empty_cache()
        gc.collect()

    def _generate(self, formatted_prompts: List[str]) -> List[str]:
        """
        Core generation logic used by both predict and batch_predict.
        
        Args:
            formatted_prompts: List of formatted prompts ready for the model
            
        Returns:
            List of generated responses
        """
        bad_words_ids = [
            self.tokenizer(p, add_special_tokens=False).input_ids 
            for p in self.bad_phrases 
            if self.tokenizer(p, add_special_tokens=False).input_ids
        ]
        
        # Track individual prompt lengths for proper decoding
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

    def predict_single(self, source_text: str) -> str:
        """
        Translate a single text.
        
        Args:
            source_text: Japanese text to translate
            
        Returns:
            English translation
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        formatted = format_input([source_text])
        result = self._generate(formatted)
        return result[0]

    def batch_predict(self, source_texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            source_texts: List of Japanese texts to translate
            
        Returns:
            List of English translations
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not source_texts:
            return []
            
        formatted = format_input(source_texts)
        return self._generate(formatted)

    def predict(self, source_texts: Union[str, List[str]]) -> List[str]:
        """
        Main predict method used by evaluators.
        Behavior controlled by self.use_batch flag.
        
        Args:
            source_texts: Single text or list of texts
            
        Returns:
            List of translations
        """
        if isinstance(source_texts, str):
            source_texts = [source_texts]
        
        if self.use_batch:
            return self.batch_predict(source_texts)
        else:
            # Translate one by one
            return [self.predict_single(text) for text in source_texts]

    def unload_model(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._clear_memory()
        print("Model unloaded")