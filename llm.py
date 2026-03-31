from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResult:
    text: str

class OpenAILLM:
    """
    Gerçek LLM çağrısı.
    """
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY .env içinde bulunamadı.")
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> LLMResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return LLMResult(text=resp.choices[0].message.content.strip())


class MockLLM:
    """
    (İstersen kalsın) Test için.
    """
    def generate(self, prompt: str) -> LLMResult:
        trimmed = prompt.strip().replace("\n", " ")
        if len(trimmed) > 260:
            trimmed = trimmed[:260] + "..."
        return LLMResult(text=f"[MOCK OUTPUT] {trimmed}")