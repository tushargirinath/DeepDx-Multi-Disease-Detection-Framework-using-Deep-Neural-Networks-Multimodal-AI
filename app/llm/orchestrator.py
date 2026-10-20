"""
Reasoning orchestrator.

Coordinates the contextual-memory sources (KB + BioGPT + Med-PaLM) and asks
Claude to synthesize a calibrated clinical interpretation of the CNN output.
If Claude is unavailable, a deterministic rule-based interpretation is returned
so the UI always shows a complete, useful result.
"""
from __future__ import annotations

from app.config import Settings
from app.llm import disease_metadata
from app.llm.biogpt_adapter import BioGPTAdapter
from app.llm.groq_client import GroqClient
from app.llm.medpalm_adapter import MedPaLMAdapter
from app.llm.prompts import SYSTEM_PROMPT, build_synthesis_prompt

DISCLAIMER = (
    "Research decision-support only — not a medical diagnosis. "
    "All findings require confirmation by a licensed clinician."
)


class ReasoningOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.groq = GroqClient(settings)
        self.biogpt = BioGPTAdapter(enabled=settings.enable_biogpt)
        self.medpalm = MedPaLMAdapter(enabled=settings.enable_medpalm)

    def interpret(self, prediction: dict, gradcam_focus: str, patient_context: dict | None) -> dict:
        top = prediction["top_prediction"]
        info = disease_metadata.resolve(top["label"])

        kb_text = (
            f"{info.description} Group: {info.disease_group}; organ: {info.organ_system}; "
            f"modality: {info.modality}; category: {info.category} (severity: {info.severity_hint})."
        )
        biogpt_text = self.biogpt.generate_context(info.disease_group, top["label"])
        medpalm_text = self.medpalm.generate_context(info.disease_group, top["label"], patient_context)
        memory = {"kb": kb_text, "biogpt": biogpt_text, "medpalm": medpalm_text}

        payload = {
            "demo_mode": prediction.get("demo_mode", False),
            "predictions": prediction["predictions"],
            "disease_info": info.as_dict(),
            "gradcam_focus": gradcam_focus,
            "patient_context": patient_context,
            "memory": memory,
        }

        prompt = build_synthesis_prompt(payload)

        # --- Primary: Groq Cloud API ---
        result = self.groq.interpret(SYSTEM_PROMPT, prompt)

        # --- Safety Net: If Groq fails, fall back to deterministic clinical synthesizer ---
        if result is None:
            result = self._fallback(payload, info)

        result["_memory"] = memory
        result["_engine"] = result.get("_source", "rule-based-fallback")
        result["_sources_active"] = {
            "groq": self.groq.available,
            "biogpt_live": self.settings.enable_biogpt,
            "medpalm_live": self.settings.enable_medpalm,
        }
        result.setdefault("disclaimer", DISCLAIMER)
        return result

    # ------------------------------------------------------------------ #
    def _fallback(self, payload: dict, info: disease_metadata.DiseaseInfo) -> dict:
        preds = payload["predictions"]
        top = preds[0]
        prob = top["probability"] * 100
        gap = (preds[0]["probability"] - preds[1]["probability"]) * 100 if len(preds) > 1 else 100.0
        demo = payload.get("demo_mode", False)

        if demo:
            conf = "DEMO MODE — the prediction is simulated (weights not loaded) and carries no clinical meaning."
        elif prob >= 85 and gap >= 20:
            conf = f"High model confidence ({prob:.1f}%) with a clear {gap:.1f} pt margin over the runner-up."
        elif prob >= 60:
            conf = f"Moderate confidence ({prob:.1f}%); margin to runner-up is {gap:.1f} pts — consider the differential."
        else:
            conf = f"Low confidence ({prob:.1f}%); predictions are close ({gap:.1f} pt margin) — treat as non-discriminative."

        pc = payload.get("patient_context") or {}
        pc_bits = [f"{k}: {v}" for k, v in pc.items() if v]
        patient_line = (
            "Provided context (" + "; ".join(pc_bits) + ") should be correlated with the finding."
            if pc_bits else "No patient context was provided; add age/sex/symptoms/history for richer correlation."
        )

        return {
            "headline": f"Model suggests {info.disease_group} ({top['label']}) at {prob:.1f}% confidence.",
            "model_finding": f"The CNN's top class is '{top['label']}' ({info.category}, {info.organ_system}) on {info.modality}, at {prob:.1f}%.",
            "confidence_assessment": conf,
            "differential": [
                {"condition": p["label"], "rationale": f"ranked #{p['rank']} at {p['probability']*100:.1f}%"}
                for p in preds[:3]
            ],
            "gradcam_correlation": f"Model attention was {payload.get('gradcam_focus','n/a')}. Verify this region is anatomically plausible for {info.modality}.",
            "biomedical_context": f"[KB] {payload['memory']['kb']}  {payload['memory']['biogpt']}  {payload['memory']['medpalm']}",
            "patient_considerations": patient_line,
            "recommended_next_steps": [
                "Correlate with the full clinical picture and prior imaging.",
                "Obtain gold-standard confirmation appropriate to the modality (e.g., biopsy/histology or specialist review).",
                "Escalate to the relevant specialist if the category is malignant or high-severity.",
            ],
            "uncertainty_and_limitations": [
                "Interpretation generated WITHOUT the Claude reasoning model (offline fallback).",
                "Model trained on specific public datasets; performance may not transfer to other scanners/populations.",
                "A confident softmax score is not calibrated probability of disease.",
            ],
            "disclaimer": DISCLAIMER,
            "_source": "rule-based-fallback",
        }
