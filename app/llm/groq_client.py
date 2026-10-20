"""
Groq client — ultra-fast LPU inference for medical reasoning synthesis.

Connects to the Groq Cloud API (OpenAI-compatible) for sub-second clinical
report generation.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Optional

from app.config import Settings

logger = logging.getLogger("groq")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("gsk_"))

    def interpret(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> Optional[dict]:
        if not self.available:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DeepDx/1.0",
        }

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            req_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(GROQ_ENDPOINT, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_body = json.loads(resp.read().decode("utf-8"))

            content = res_body["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if parsed is not None:
                parsed["_source"] = f"groq:{self.model}"
                logger.info("Groq (%s) interpretation generated successfully.", self.model)
            return parsed
        except Exception as exc:
            logger.warning("Groq call failed: %s", exc)
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

    # 2. Markdown code fences
    matches = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if isinstance(data, dict):
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

    return None
