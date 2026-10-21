"""
Medical image validator using OpenRouter Multimodal Vision API.

Verifies that an uploaded image is a legitimate medical/clinical image
(histopathology, X-ray, CT, MRI, dermatoscopy, cytology, ultrasound, etc.)
before running CNN inference.

Uses Google Gemma 4 26B A4B (free) with automatic fallback to Minimax / Nemotron
via OpenRouter. If an image is rejected (e.g., screenshots, selfies, landscapes),
the API returns HTTP 422 with actionable guidance.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from typing import Optional

from app.config import Settings

logger = logging.getLogger("image_validator")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

VALIDATION_PROMPT = """You are an automated medical diagnostic filter. Your ONLY task is to check if this image is a legitimate medical or clinical image.

VALID MEDICAL IMAGES include:
- Histopathology tissue slides (H&E stain, IHC, biopsy)
- X-rays (chest, dental, musculoskeletal, etc.)
- CT scans or MRI scans
- Ultrasound images
- Dermatoscopy or clinical photographs of skin lesions/rashes
- Cytology slides (Pap smear, blood smear)
- Fundoscopy / retinal photography
- Mammography
- Endoscopy images
- Clinical surgical/wound photographs

INVALID IMAGES include:
- Screenshots of computer screens, web browsers, emails, software, or code
- Text documents, receipts, book pages, PDFs
- Selfies, casual portraits, family photos
- Food, animals, pets, nature, landscapes
- Memes, artwork, drawings, cartoons, icons
- Random everyday household or office objects

Respond with ONLY a valid JSON object (no markdown, no extra text):
{"valid": true, "category": "<type of medical image>", "reason": "<brief description>"}
or
{"valid": false, "reason": "<specific explanation of why this image is not a medical image>"}"""


class ImageValidator:
    """Pre-inference gate that validates medical images via OpenRouter Vision."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openrouter_api_key or settings.groq_api_key
        self.enabled = settings.enable_image_validation
        
        # Primary model is Google Gemma 4 26B (free), with zero-downtime fallbacks
        self.models = [settings.image_validation_model] + [
            m for m in settings.image_validation_fallbacks if m != settings.image_validation_model
        ]

    @property
    def available(self) -> bool:
        """Validator is available if enabled and an API key is present."""
        return self.enabled and bool(self.api_key and (
            self.api_key.startswith("sk-or-") or self.api_key.startswith("gsk_")
        ))

    def validate(self, image_bytes: bytes, mime_type: str = "image/png") -> dict:
        """
        Validate whether the image is a medical/clinical image.

        Returns:
            dict with:
                - valid (bool): True if legitimate medical image
                - reason (str): Explanation
                - category (str, optional): Medical category
                - model_used (str, optional): Model that evaluated
                - skipped (bool): True if validation was bypassed
        """
        if not self.available:
            logger.info("Image validation skipped (disabled or no OpenRouter API key).")
            return {"valid": True, "skipped": True, "reason": "Image validation not configured."}

        try:
            return self._call_openrouter_vision(image_bytes, mime_type)
        except Exception as exc:
            logger.warning("Image validation call failed: %s (falling back to open)", exc)
            return {"valid": True, "skipped": True, "reason": f"Validation service warning: {exc}"}

    def _call_openrouter_vision(self, image_bytes: bytes, mime_type: str) -> dict:
        """Send image to OpenRouter Vision API with automatic model fallbacks."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://deepdx.myvnc.com",
            "X-Title": "DeepDx Multi-Disease Detection Framework",
            "User-Agent": "DeepDx/1.0",
        }

        body = {
            "models": self.models,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VALIDATION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "max_tokens": 250,
        }

        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(OPENROUTER_ENDPOINT, data=req_data, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))

        model_used = res_body.get("model", "unknown")
        msg = res_body["choices"][0]["message"]
        
        # OpenRouter models may return text in content, reasoning, or reasoning_details
        raw_text = msg.get("content") or ""
        if not raw_text and msg.get("reasoning"):
            raw_text = msg.get("reasoning") or ""
        if not raw_text and "reasoning_details" in msg:
            raw_text = "".join(b.get("text", "") for b in msg["reasoning_details"] if isinstance(b, dict))

        parsed = self._parse_response(raw_text)
        if parsed is not None:
            parsed["model_used"] = model_used
            logger.info(
                "Image validation [%s]: valid=%s, reason=%s",
                model_used, parsed.get("valid"), parsed.get("reason", ""),
            )
            return parsed

        # Fallback keyword scan if JSON formatting was loose
        lower_text = raw_text.lower()
        if "valid\": false" in lower_text or "not a medical" in lower_text or "screenshot" in lower_text:
            return {
                "valid": False,
                "reason": "Image was identified as non-medical content.",
                "model_used": model_used,
            }

        logger.warning("Could not parse validator JSON response from [%s]: %s", model_used, raw_text[:200])
        return {"valid": True, "skipped": True, "reason": "Unparsed validation response (allowed)."}

    @staticmethod
    def _parse_response(text: str) -> Optional[dict]:
        """Extract JSON validation result from LLM output."""
        if not text:
            return None
        text = text.strip()

        # 1. Direct parse
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "valid" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Markdown code fences
        for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text):
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "valid" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # 3. Outer bracket extraction
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict) and "valid" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return None
