# Sample Pipeline Outputs (5)

These are **real outputs from the app's own reasoning code** (`orchestrator.py` →
`disease_metadata.py` → `prompts.py`), generated on 2026-08-24. They were produced
**offline**, so the interpretation engine shown is the deterministic **rule-based
fallback**. When you run locally with your `ANTHROPIC_API_KEY`, the same call path
instead returns Claude's richer JSON (the `_engine` field will read `claude:claude-sonnet-4-5`).

The CNN probability vectors below are representative examples across organ systems;
run `uv run python tools/smoke_test.py` to replace them with your trained model's
actual predictions.

---

## 1 — Lung histopathology (H&E)

- **Top-1:** `lung_squamous_cell_carcinoma` — **91.3%**
- **Headline:** Model suggests Lung Cancer (lung_squamous_cell_carcinoma) at 91.3% confidence.
- **Confidence:** High model confidence (91.3%) with a clear 86.2 pt margin over the runner-up.
- **Differential:** lung_squamous_cell_carcinoma (91.3%) › lung_adenocarcinoma (5.1%) › lung_benign_tissue (1.9%)
- **Grad-CAM:** a tightly localized focal region in the central portion (~11% high-salience area) — verify plausible for Histopathology (H&E).
- **Patient:** age 63, M, chronic cough + weight loss, 40 pack-year smoker — correlate with the finding.

## 2 — Brain MRI

- **Top-1:** `brain_glioma` — **61.2%**
- **Headline:** Model suggests Brain Tumor (brain_glioma) at 61.2% confidence.
- **Confidence:** Moderate confidence (61.2%); margin to runner-up is 33.1 pts — consider the differential.
- **Differential:** brain_glioma (61.2%) › brain_meningioma (28.1%) › brain_pituitary (7.4%)
- **Grad-CAM:** a moderately sized region in the upper-left portion (~18% high-salience area).
- **Patient:** age 47, F, morning headaches + blurred vision.

## 3 — Breast imaging

- **Top-1:** `breast_benign` — **88.4%**
- **Headline:** Model suggests Breast Cancer (breast_benign) at 88.4% confidence.
- **Confidence:** High model confidence (88.4%) with a clear 76.8 pt margin over the runner-up.
- **Differential:** breast_benign (88.4%) › breast_malignant (11.6%)
- **Grad-CAM:** a tightly localized focal region in the lower-right portion (~7% high-salience area).
- **Patient:** age 39, F, palpable mobile lump.

## 4 — Brain MRI (Alzheimer staging)

- **Top-1:** `Alzheimers_ModerateDemented` — **50.3%**
- **Headline:** Model suggests Alzheimer's Disease (dementia staging) at 50.3% confidence.
- **Confidence:** Low confidence (50.3%); predictions are close (13.1 pt margin) — treat as non-discriminative.
- **Differential:** Alzheimers_ModerateDemented (50.3%) › Alzheimers_MildDemented (37.2%) › Alzheimers_VeryMildDemented (8.1%)
- **Grad-CAM:** a broad, distributed region in the central portion (~31% high-salience area).
- **Patient:** age 78, F, progressive memory loss, family history of dementia.

## 5 — Colon histopathology (H&E)

- **Top-1:** `colon_adenocarcinoma` — **94.7%**
- **Headline:** Model suggests Colon Cancer (colon_adenocarcinoma) at 94.7% confidence.
- **Confidence:** High model confidence (94.7%) with a clear 89.4 pt margin over the runner-up.
- **Differential:** colon_adenocarcinoma (94.7%) › colon_benign_tissue (5.3%)
- **Grad-CAM:** a tightly localized focal region in the central portion (~9% high-salience area).
- **Patient:** age 58, M, rectal bleeding + altered bowel habit.

---

### What these demonstrate

The pipeline **calibrates its language to the CNN's confidence**: cases 1/3/5 (large
margins) read "High confidence," case 2 (33 pt gap) reads "Moderate — consider the
differential," and case 4 (13 pt gap) is honestly flagged "Low / non-discriminative."
Every output ends with confirmatory next steps and a not-a-diagnosis disclaimer — the
safety boundary is enforced in code, not left to the model.
