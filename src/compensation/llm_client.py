from __future__ import annotations

import os


class OpenAICompatibleClient:
    def __init__(self, model_name: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("openai package is required for compensation generation") from exc
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
