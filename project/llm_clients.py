from __future__ import annotations
import os

class BaseLLM:
    name: str
    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

class NoLLM(BaseLLM):
    name = "none"
    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        # "citations only" mode: just returns the prompt
        return prompt

class OpenAIChat(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"

    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

class HFLocal(BaseLLM):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        self.mdl.to(self.device)
        self.name = f"hf:{model_id}"

    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        s = self.tok.decode(out[0], skip_special_tokens=True)
        return s.split("<|assistant|>")[-1].strip()
