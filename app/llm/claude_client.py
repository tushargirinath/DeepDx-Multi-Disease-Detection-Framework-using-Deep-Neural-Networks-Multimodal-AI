"""
Claude client — the reasoning orchestrator's connection to Anthropic.

Returns a parsed JSON interpretation, or None if Claude is unavailable (missing
key / network / parse error) so the orchestrator can fall back gracefully.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.config import Settings

logger = logging.getLogger("claude")


class ClaudeClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.claude_model
        self._client = None
        if settings.has_claude:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Anthropic SDK init failed: %s", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def interpret(self, system_prompt: str, user_prompt: str, max_tokens: int = 1600) -> Optional[dict]:
        if not self.available:
            return None
        try:
            msg = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
            parsed = _extract_json(text)
            if parsed is not None:
                parsed["_source"] = f"claude:{self.model}"
            return parsed
        except Exception as exc:  # pragma: no cover
            logger.warning("Claude call failed: %s", exc)
            return None


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None
