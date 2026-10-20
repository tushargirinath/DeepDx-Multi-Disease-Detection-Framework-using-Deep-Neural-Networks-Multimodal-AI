"""
OpenRouter client — OpenAI-compatible interface for multi-model fallback.

Provides seamless access to OpenRouter models (e.g. Minimax, Llama, Gemini, Mistral)
as a high-performance fallback tier if Claude is unavailable or credit-limited.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Optional

from app.config import Settings

logger = logging.getLogger("openrouter")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openrouter_api_key
        self.model = settings.openrouter_model

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-or-"))

    def interpret(self, system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Optional[dict]:
        if not self.available:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "DeepDx Multi-Disease Detection",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max(max_tokens, 3000),
            "temperature": 0.2,
        }

        try:
            req_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(OPENROUTER_ENDPOINT, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))

            msg = res_body["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content:
                content = msg.get("reasoning") or ""
            if not content and "reasoning_details" in msg:
                content = "".join(b.get("text", "") for b in msg["reasoning_details"] if isinstance(b, dict))

            parsed = _extract_json(content)
            if parsed is not None:
                parsed["_source"] = f"openrouter:{self.model}"
                logger.info("OpenRouter (%s) interpretation generated successfully.", self.model)
            return parsed
        except Exception as exc:
            logger.warning("OpenRouter call failed: %s", exc)
            return None


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()

    # 1. Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Markdown code fences (e.g., ```json { ... } ```)
    matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if isinstance(data, dict) and "headline" in data:
                return data
        except Exception:
            pass

    # 3. Outer bracket extraction
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 4. Backward scan for JSON object containing 'headline'
    end = text.rfind("}")
    while end != -1:
        start = text.find("{")
        while start != -1 and start < end:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "headline" in data:
                    return data
            except Exception:
                pass
            start = text.find("{", start + 1)
        end = text.rfind("}", 0, end)

    return None
