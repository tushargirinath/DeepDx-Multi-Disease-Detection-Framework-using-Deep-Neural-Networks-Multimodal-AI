"""
Medical image validator using OpenRouter Multimodal Vision API.

Verifies that an uploaded image is a legitimate medical/clinical image
(histopathology, X-ray, CT, MRI, dermatoscopy, cytology, ultrasound, etc.)
before running CNN inference.

Uses multimodal models (NVIDIA Nemotron 3.5 Super VL, Google Gemma 4 26B/31B)
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

VALIDATION_PROMPT = """You are a strict automated medical image triage system. Check if this image is a legitimate clinical medical scan or pathology slide.

CRITICAL RULES:
- Photos of people, faces, selfies, portraits, casual body shots, clothing, everyday objects, and room/vehicle interiors are STRICTLY INVALID.
- Screenshots of computer screens, web browsers, software, or phones are STRICTLY INVALID.
- Landscapes, animals, food, memes, art, and drawings are STRICTLY INVALID.

VALID CLINICAL IMAGES include ONLY:
- Histopathology tissue slides (H&E stain, IHC, biopsy)
- X-rays (chest, dental, musculoskeletal, etc.)
- CT scans or MRI scans
- Ultrasound images
- Dermatoscopy or clinical close-up photographs of skin lesions/rashes
- Cytology slides (Pap smear, blood smear)
- Fundoscopy / retinal photography
- Mammography
- Endoscopy images

You MUST respond with valid JSON:
{"valid": false, "reason": "<clear explanation of why this image is not a medical scan>"}
or
{"valid": true, "category": "<type of medical image>", "reason": "<brief description>"}"""

# Non-medical keyword indicators
NON_MEDICAL_KEYWORDS = [
    "not a medical", "not medical", "invalid", "false",
    "selfie", "portrait", "person", "man", "woman", "face", "hair",
    "beard", "headshot", "casual photo", "photo of a man", "photo of a person",
    "vehicle", "car", "bus", "train", "seat", "interior", "hoodie", "shirt",
    "screenshot", "webpage", "browser", "desktop", "interface", "monitor",
    "food", "animal", "pet", "dog", "cat", "nature", "landscape", "drawing"
]


class ImageValidator:
    """Pre-inference gate that validates medical images via OpenRouter Vision."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.openrouter_api_key or settings.groq_api_key
        self.enabled = settings.enable_image_validation
        
        # OpenRouter enforces a strict maximum of 3 items in the 'models' fallback array
        all_models = [settings.image_validation_model] + [
            m for m in settings.image_validation_fallbacks if m != settings.image_validation_model
        ]
        self.models = all_models[:3]

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
            "max_tokens": 1000,
        }

        req_data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(OPENROUTER_ENDPOINT, data=req_data, headers=headers)

        with urllib.request.urlopen(req, timeout=35) as resp:
            res_body = json.loads(resp.read().decode("utf-8"))

        model_used = res_body.get("model", "unknown")
        msg = res_body["choices"][0]["message"]
        
        # Combine content and reasoning for full analysis
        content_text = msg.get("content") or ""
        reasoning_text = msg.get("reasoning") or ""
        if "reasoning_details" in msg:
            reasoning_text += "".join(b.get("text", "") for b in msg["reasoning_details"] if isinstance(b, dict))

        full_text = f"{content_text}\n{reasoning_text}".strip()

        # Try parsing structured JSON first from content, then reasoning, then combined
        parsed = self._parse_response(content_text) or self._parse_response(reasoning_text) or self._parse_response(full_text)
        if parsed is not None:
            parsed["model_used"] = model_used
            logger.info(
                "Image validation [%s]: valid=%s, reason=%s",
                model_used, parsed.get("valid"), parsed.get("reason", ""),
            )
            return parsed

        # Fallback keyword scan if JSON formatting was incomplete
        lower_text = full_text.lower()
        
        # If any non-medical keywords are mentioned in model's observations:
        matched_indicators = [kw for kw in NON_MEDICAL_KEYWORDS if kw in lower_text]
        if matched_indicators:
            detected_type = matched_indicators[0]
            logger.info("Image rejected by fallback keyword scan: detected '%s'", detected_type)
            return {
                "valid": False,
                "reason": f"Image was identified as non-medical content ({detected_type}).",
                "model_used": model_used,
            }

        logger.warning("Could not parse validator JSON response from [%s]: %s", model_used, full_text[:200])
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

        # 4. Search for regex pattern {"valid": false ...}
        pattern = r'\{\s*"valid"\s*:\s*(false|true)\s*,\s*"reason"\s*:\s*"([^"]+)"\s*\}'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "valid": match.group(1).lower() == "true",
                "reason": match.group(2)
            }

        return None
