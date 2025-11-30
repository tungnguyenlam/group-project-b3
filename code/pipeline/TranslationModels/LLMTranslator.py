from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union
import torch
import gc

test_sys_prompt = """Translate Japanese to English. Examples:

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

bad_phrases = [
    "I can't", "I cannot", "I'm sorry", "I am sorry",
    "As an AI", "cannot help", "unable to", "I won't", "I will not",
    "I cannot comply", "I cannot assist"
]

def format_input(source_texts: List[str], add_begin: str = "Japanese: ", add_ending: str = "\nEnglish: "):
    formated_texts = []
    for item in source_texts:
        formated_texts.append(add_begin + item + add_ending)
    return formated_texts

class LLMTranslator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None

    def load_model(self, model_name="Qwen/Qwen2.5-0.5B", device='auto'):
        if self.model is None or self.tokenizer is None:
            if device == 'auto':
                if torch.cuda.is_available():
                    device = 'cuda'
                elif torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'

            self.device = device

            if self.device and 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            elif self.device and 'mps' in str(self.device):
                torch.mps.empty_cache()

            gc.collect()

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_name))
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map = device,
                trust_remote_code = True
            )

            if self.model is None or self.tokenizer is None:
                raise TypeError("Error: No model loaded")

        else:
            print("Model is already loaded")
        
    def predict(self, source_texts: Union[str, List[str]], 
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_p: float = 1.0,
                system_prompt = test_sys_prompt,
                bad_phrases = bad_phrases) -> List[str]:
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if isinstance(source_texts, str):
            source_texts = [source_texts]

        source_texts = format_input(source_texts)

        bad_words_ids = [ self.tokenizer(p, add_special_tokens=False).input_ids for p in bad_phrases if self.tokenizer(p, add_special_tokens=False).input_ids
]
            
        individual_lengths = []
        for prompt in source_texts:
            tokens = self.tokenizer(prompt, return_tensors="pt")
            individual_lengths.append(tokens["input_ids"].shape[1])

        inputs = self.tokenizer(source_texts, return_tensors = "pt", padding = True, truncation = True)
        input_ids_length = inputs["input_ids"].shape[1]

        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                top_p = top_p,
                do_sample = True,
                pad_token_id = self.tokenizer.eos_token_id,
                bad_words_ids = bad_words_ids
            )

        batch_responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(
                output[individual_lengths[i]:], 
                skip_special_tokens=True
            )
            batch_responses.append(response.strip())

        translated_texts = batch_responses

        del source_texts, inputs

        return translated_texts

    def unload_model(self):
        del self.model
        del self.tokenizer
        del self.current_model_name

        if self.device and 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        elif self.device and 'mps' in str(self.device):
             torch.mps.empty_cache()

        gc.collect()

        self.model = self.tokenizer = self.current_model_name = None

        print("Model unloaded")