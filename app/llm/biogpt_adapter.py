"""
BioGPT contextual-memory adapter.

If ENABLE_BIOGPT=true AND the optional deps are installed (`uv sync --extra biogpt`),
this runs Microsoft's BioGPT locally to generate biomedical context. Otherwise it
returns a curated, clearly-labeled fallback so the pipeline still works end-to-end.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("biogpt")

# Curated fallbacks keyed by disease group (used when the real model is off/unavailable).
_FALLBACK = {
    "Lung Cancer": "Lung carcinomas include adenocarcinoma and squamous cell carcinoma; squamous subtypes are strongly smoking-associated and often central, while adenocarcinomas are frequently peripheral. Histologic confirmation guides staging and therapy.",
    "Colon Cancer": "Colorectal adenocarcinoma arises from glandular epithelium, commonly through the adenoma–carcinoma sequence. Histopathology grading and staging inform prognosis and management.",
    "Breast Cancer": "Breast carcinomas span benign and malignant histologies; malignant lesions show architectural and nuclear atypia. Receptor status (ER/PR/HER2) and grade drive treatment planning.",
    "Oral Cancer": "Oral squamous cell carcinoma is the most common oral malignancy, linked to tobacco and alcohol; early histologic detection markedly improves outcomes.",
    "Malignant Lymphoma": "Lymphomas are clonal lymphoid malignancies (e.g., CLL/SLL, follicular, mantle cell) distinguished by architecture, immunophenotype, and genetics; subtype dictates aggressiveness and therapy.",
    "Acute Lymphoblastic Leukemia (ALL)": "ALL is characterized by lymphoblast proliferation in marrow and blood; morphologic stages and immunophenotyping (pre-B/pro-B) guide risk stratification.",
    "Brain Tumor": "Primary brain tumors include gliomas, meningiomas, and pituitary adenomas, differing in location, growth pattern, and MRI appearance; imaging plus histology establish the diagnosis.",
    "Alzheimer's Disease (dementia staging)": "Alzheimer's disease shows progressive medial-temporal and hippocampal atrophy on MRI, correlating with clinical dementia staging from non-demented to moderate.",
    "Kidney Disease": "Renal masses on CT are evaluated for size, enhancement, and morphology; solid enhancing lesions raise concern for renal cell carcinoma and warrant further workup.",
    "Cervical Cytology (SIPaKMeD)": "Cervical Pap cytology classifies squamous cells from normal (superficial/intermediate, parabasal) to abnormal (koilocytotic, dyskeratotic), aiding HPV-related lesion screening.",
}


class BioGPTAdapter:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._pipe = None
        self._load_failed = False

    def _ensure_loaded(self) -> bool:
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False
        try:
            from transformers import pipeline  # type: ignore

            logger.info("Loading microsoft/biogpt (first call may download ~1.5 GB)…")
            self._pipe = pipeline("text-generation", model="microsoft/biogpt")
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("BioGPT unavailable (%s). Using curated fallback.", exc)
            self._load_failed = True
            return False

    def generate_context(self, disease_group: str, class_name: str) -> str:
        fallback = _FALLBACK.get(disease_group, f"General biomedical context for {disease_group.lower()}.")

        if not self.enabled:
            return f"{fallback}  [BioGPT fallback — set ENABLE_BIOGPT=true to run the real model]"

        if not self._ensure_loaded():
            return f"{fallback}  [BioGPT fallback — model not installed; run `uv sync --extra biogpt`]"

        try:
            seed = disease_group.split("(")[0].strip()
            out = self._pipe(
                seed,
                max_new_tokens=70,
                num_return_sequences=1,
                do_sample=False,
                truncation=True,
            )
            text = out[0]["generated_text"].strip().replace("\n", " ")
            return f"{text}  [BioGPT: microsoft/biogpt]"
        except Exception as exc:  # pragma: no cover
            logger.warning("BioGPT generation failed: %s", exc)
            return f"{fallback}  [BioGPT fallback — generation error]"
