"""Loading and running causal language models."""

import gc
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def free_memory(report: bool = False) -> None:
    """Collect garbage and release cached CUDA memory; optionally print the memory still in use."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if report:
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    elif report:
        print("CUDA is not available.")


def default_dtype() -> torch.dtype:
    """fp16 on CUDA, fp32 on CPU."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


def extract_tag(text: str, tag: str) -> str | None:
    """Content of the first <tag>...</tag> block, or None if the tag is missing."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


class LLM:
    """A causal LM with its tokenizer, using greedy decoding."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, model_id: str, *, dtype=None, load_in_4bit: bool = False,
             trust_remote_code: bool = False) -> "LLM":
        """Load a model and tokenizer from the Hugging Face Hub."""
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype or default_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        print("Model loaded successfully!")
        return cls(model, tokenizer)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy-decode a continuation of `prompt` and return only the new text."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat_prompt(self, messages: list[dict]) -> str:
        """Render chat messages with the model's own chat template."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def close(self) -> None:
        """Drop the model and tokenizer and free the memory."""
        self.model = None
        self.tokenizer = None
        free_memory()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
