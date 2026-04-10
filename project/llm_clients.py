from __future__ import annotations
import os
from typing import Iterator


class BaseLLM:
    name: str

    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError

    def stream_generate(self, system: str, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        """Yield text chunks as they are generated. Default: emit full result at once."""
        yield self.generate(system, prompt, max_tokens)


class NoLLM(BaseLLM):
    name = "none"

    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        # "citations only" mode: just returns the prompt
        return prompt
    # stream_generate inherited from BaseLLM (yields full result in one shot)


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
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def stream_generate(self, system: str, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        with self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta


class AnthropicChat(BaseLLM):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.name = f"anthropic:{model}"

    def generate(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    def stream_generate(self, system: str, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text


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

    def stream_generate(self, system: str, prompt: str, max_tokens: int = 512) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        import threading
        text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        ids = self.tok(text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        t = threading.Thread(
            target=lambda: self.mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=False, streamer=streamer),
            daemon=True,
        )
        t.start()
        for chunk in streamer:
            yield chunk
        t.join()
