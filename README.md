# Multi-Disease Detection Framework — Deep Neural Networks & Multimodal AI

A hybrid clinical-decision-**support** framework that pairs your trained **custom CNN**
(94.82% validation accuracy across **30 disease classes**) with an **LLM reasoning layer**.
The CNN is the *perception* engine; **Claude** is the *reasoning* engine, enriched by three
**contextual-memory sources** — a structured Knowledge Base, **Microsoft BioGPT**, and
**Med-PaLM 2** — to turn a raw softmax vector into a calibrated, source-attributed,
safety-bounded interpretation.

It ships as an **interactive web application** you can actually test in the browser:
drag in a medical image → get the CNN prediction, a **Grad-CAM** attention heatmap, the
LLM's contextual reasoning, a **prediction history log**, and a one-click **PDF clinical report**.

> ⚠️ **Research / educational prototype — not a medical device.** Every output is decision-support, never a diagnosis.

---

## What's inside

```
multi-disease-detection-app/
├── app/
│   ├── main.py                 # FastAPI app: /api/health, /api/analyze, /api/report, serves the UI
│   ├── config.py               # settings from .env (pydantic-settings)
│   ├── schemas.py              # pydantic request/response models
│   ├── ml/
│   │   ├── cnn_model.py        # your Cell-10 architecture, weight loading, demo mode, predict()
│   │   └── gradcam.py          # Grad-CAM heatmap + plain-English "where the model looked"
│   ├── llm/
│   │   ├── prompts.py          # the prompt engineering (system + synthesis prompt)
│   │   ├── claude_client.py    # Anthropic call + robust JSON extraction
│   │   ├── disease_metadata.py # structured Knowledge Base for all 30 classes  [KB]
│   │   ├── biogpt_adapter.py   # Microsoft BioGPT source (real via HF, else curated fallback) [BioGPT]
│   │   ├── medpalm_adapter.py  # Med-PaLM 2 source (labeled placeholder; Vertex MedLM path documented) [Med-PaLM]
│   │   └── orchestrator.py     # coordinates the 3 memory sources + Claude, with rule-based fallback
│   └── report/
│       └── pdf_report.py       # clinical PDF export (reportlab)
├── frontend/                   # vanilla HTML/CSS/JS "diagnostic console" (no build step)
├── models/
         └──best_model.weights.h5
         └──readme.md                   
├── tools/
│   └── export_class_names_snippet.py   # run in your Colab to regenerate the authoritative class_names.json
├── class_names.json            # ordered label list (index -> class) — VERIFY THIS (see below)
├── pyproject.toml              # uv-managed dependencies
├── .python-version             # 3.13
└── .env.example                # copy to .env and add your key
```

---

## ⭐ 3 things I need from you (important)

The app **runs immediately in demo mode** without any of these, but for *real* results:

### 1. Add your Anthropic API key
Claude powers the reasoning layer. Copy the example env file and paste your key:

```bash
cp .env.example .env
```

Then open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

Without a valid `sk-ant-...` key the app still works, but the interpretation comes from the
built-in **rule-based fallback** instead of Claude (the UI shows "claude offline").

### 2. Drop your trained weights into `models/`
Copy your file so the path matches `.env` (`CNN_WEIGHTS_PATH=models/best_model.weights.h5`):

```
models/best_model.weights.h5
```

Without it, predictions are **simulated (demo mode)** — the UI clearly flags this with a banner.

### 3. ✅ Verify `class_names.json` matches your training label order  ← most important
This is the one thing I could **not** fully confirm from the notebook. Your CNN outputs a vector of
30 probabilities; index *i* only means something if `class_names.json[i]` is the label that index
was trained on. In your notebook the order came from `sorted(df['label'].unique())`
(Python case-sensitive sort — **uppercase before lowercase**). I reconstructed that list, but a few
leaf-label spellings were guesses.

**To generate the authoritative file**, run the snippet in `tools/export_class_names_snippet.py`
inside your Colab/training notebook (where `df` exists) and replace `class_names.json`:

```python
# in your notebook, after df is built:
import json
class_names = sorted(df['label'].unique())          # EXACT training order
json.dump(class_names, open('class_names.json', 'w'), indent=2)
print(len(class_names), "classes written")
```

Download that `class_names.json` and drop it in the project root. If the order is wrong, the model
will still run but labels will be **mismatched**, so please do this before trusting any real prediction.

---

## Run it in VS Code with `uv` (step by step)

This project uses [**uv**](https://docs.astral.sh/uv/) — a fast Python package manager — **not pip**.

### Step 1 — Open the project directory

```bash
cd DeepDx-Multi-Disease-Detection-Framework
```

### Step 2 — Install uv (once, if you don't have it)

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm installation:

```bash
uv --version
```

### Step 3 — Install dependencies

Run this command from the project root:

```bash
uv sync
```

### Step 4 — Configure your API key

Create your `.env` file:

```bash
cp .env.example .env      # macOS / Linux
copy .env.example .env    # Windows
```

Open `.env` and configure your **Groq Cloud API key** (recommended for sub-second live clinical reasoning):

```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

*(Optional: Anthropic Claude and OpenRouter keys are also supported).*

### Step 5 — Configure the trained model

Ensure your trained model weights are available at:

```text
models/best_model.weights.h5
```

Verify that `class_names.json` in the project root matches the class order used during training.

### Step 6 — Start the application

**Native via uv:**

```bash
uv run python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Or with Docker Compose:**

```bash
docker compose up --build
```

### Step 7 — Open the application

Once the server starts, open:

* **Web application:** http://localhost:8000/
* **Interactive API documentation:** http://localhost:8000/docs
* **Alternative API documentation:** http://localhost:8000/redoc



## Optional — enable the *real* BioGPT model

By default BioGPT returns curated, clearly-labeled biomedical context (fast, no download). To run
the actual **Microsoft BioGPT** locally (adds `torch` + `transformers`, ~1.5 GB on first use):

```bash
uv sync --extra biogpt
```

Then set `ENABLE_BIOGPT=true` in `.env` and restart. (Med-PaLM 2 has no public API — it stays a
labeled placeholder; the real Google **Vertex AI MedLM** wiring is sketched in `medpalm_adapter.py`.)

---

## How the reasoning pipeline works

```
   image ─► CNN (perception) ─► top-k probabilities + Grad-CAM attention
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                            ▼                           ▼
        [KB] knowledge base          [BioGPT] biomedical         [Med-PaLM] clinical-QA
        (organ, modality,             narrative context           workup context
         category, severity)                                       (placeholder)
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                            Claude  (reasoning / synthesis)
                                          ▼
              calibrated, source-attributed, safety-bounded JSON interpretation
                                          ▼
                       web console  +  downloadable PDF clinical report
```

If Claude is unavailable (no key/offline), a deterministic **rule-based fallback** produces the same
JSON shape, so the UI always shows a complete result.

---

## Features you asked for

- **Grad-CAM heatmaps** — visual "where the model looked", with a heatmap/original toggle.
- **Patient context input** — age, sex, symptoms, history, notes feed the reasoning layer.
- **PDF report export** — one-click clinical-style report (prediction ladder, Grad-CAM, interpretation).
- **Prediction history log** — every analysis in the session is kept; click to revisit.

---

## API quick reference

| Method | Route           | Purpose                                                      |
|--------|-----------------|--------------------------------------------------------------|
| GET    | `/api/health`   | model + LLM source status (used by the status pills)         |
| POST   | `/api/analyze`  | multipart image (+ patient fields) → prediction, Grad-CAM, interpretation |
| POST   | `/api/report`   | analysis JSON → downloadable PDF                             |
| GET    | `/`             | the web console                                              |

---

## Troubleshooting

- **"claude offline" pill / fallback reasoning** — check `ANTHROPIC_API_KEY` in `.env` starts with `sk-ant-`.
- **"demo mode" banner** — weights not found; confirm `models/best_model.weights.h5` exists and `CNN_WEIGHTS_PATH` matches.
- **Labels look wrong** — regenerate `class_names.json` from your notebook (see step 3 / the ⭐ section).
- **`uv sync` slow the first time** — that's TensorFlow downloading; subsequent runs are cached.
- **`No matching distribution found for tensorflow` on Python 3.13** — your interpreter is newer than any published TF wheel. Fix with `uv sync --python 3.12` (uv fetches a private 3.12 just for this project; your system 3.13 is untouched), then run everything with `uv run` as normal.
- **Weights won't load / shape mismatch** — the architecture in `cnn_model.py` must match training exactly; it was copied verbatim from Cell 10. If you changed the model, update `build_custom_cnn()` to match.
- **Quick offline sanity check** — `uv run python verify_offline.py` runs a dependency-light smoke test of the metadata, prompts, adapters, orchestrator fallback, demo predictor, Grad-CAM helpers, and PDF render (49 checks, no TF/network needed).
- **Full real-pipeline check** — `uv run python tools/smoke_test.py` loads the *real* weights, runs 5 CNN predictions with Grad-CAM, and does a live Claude call. It fails loudly if it drops to demo mode. Pass image paths to use your own: `uv run python tools/smoke_test.py img1.png img2.png`.

---

*Built to extend your trained model with an LLM contextual-reasoning layer. Not for clinical use.*
