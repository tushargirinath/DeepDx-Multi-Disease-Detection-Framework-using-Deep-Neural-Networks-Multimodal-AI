"""
Offline verification harness.

The sandbox cannot install fastapi/pydantic/anthropic/tensorflow (no PyPI),
so we shim `app.config` and exercise every piece of pure logic that does NOT
require those heavy deps:  metadata KB, prompt builder, BioGPT/Med-PaLM adapters,
Claude JSON extraction, the rule-based orchestrator fallback, the CNN demo-mode
predictor (numpy+PIL), Grad-CAM helpers, and a REAL PDF render (reportlab).
"""
import io
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------- #
#  Shim app.config so config-dependent modules import without pydantic.
# --------------------------------------------------------------------- #
fake_config = types.ModuleType("app.config")


class Settings:
    def __init__(self):
        self.claude_model = "claude-sonnet-4-5"
        self.anthropic_api_key = ""            # -> has_claude False -> no anthropic import
        self.enable_biogpt = False
        self.enable_medpalm = False
        self.img_size = 224
        self.cnn_weights_path = "models/best_model.weights.h5"
        self.class_names_path = "class_names.json"

    @property
    def has_claude(self):
        return False

    @property
    def weights_file(self):
        return ROOT / "models" / "best_model.weights.h5"

    @property
    def class_names_file(self):
        return ROOT / "class_names.json"

    def load_class_names(self):
        with open(self.class_names_file, encoding="utf-8") as fh:
            return [str(x) for x in json.load(fh)]


fake_config.Settings = Settings
fake_config.ROOT_DIR = ROOT
fake_config.get_settings = lambda: Settings()
sys.modules["app.config"] = fake_config

PASS, FAIL = 0, 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {extra}")


# --------------------------------------------------------------------- #
print("\nA) disease_metadata — resolve all 30 classes")
from app.llm import disease_metadata

names = json.load(open(ROOT / "class_names.json", encoding="utf-8"))
cats = {}
ok_all = True
VALID_CAT = {"Normal", "Benign", "Precursor", "Malignant", "Neurodegenerative"}
VALID_SEV = {"low", "moderate", "high", "informational"}
for n in names:
    di = disease_metadata.resolve(n)
    cats[di.category] = cats.get(di.category, 0) + 1
    if not (di.category in VALID_CAT and di.severity_hint in VALID_SEV
            and di.disease_group and di.organ_system and di.modality and di.description
            and di.as_dict()["class_name"] == n):
        ok_all = False
        print(f"      bad resolve: {n} -> {di}")
check("all 30 classes resolve with valid category+severity+fields", ok_all)
check("category distribution sane", sum(cats.values()) == 30, str(cats))
print("      categories:", cats)
# spot-check a couple of known mappings
check("lung_squamous_cell_carcinoma -> Malignant/high",
      disease_metadata.resolve("lung_squamous_cell_carcinoma").category == "Malignant")
check("kidney_normal -> Normal/informational",
      disease_metadata.resolve("kidney_normal").severity_hint == "informational")
check("unknown prefix still resolves (robust fallback)",
      disease_metadata.resolve("pancreas_weirdlabel").category in VALID_CAT)

# --------------------------------------------------------------------- #
print("\nB) prompts — system + synthesis builder")
from app.llm import prompts

sp = prompts.SYSTEM_PROMPT
check("system prompt demands strict JSON only", "ONLY a single valid JSON object" in sp)
check("system prompt lists all 3 memory sources", all(t in sp for t in ("[KB]", "[BioGPT]", "[Med-PaLM]")))
check("system prompt sets decision-SUPPORT boundary", "NOT making a diagnosis" in sp)
up = prompts.build_synthesis_prompt(prompts.example_payload())
check("synthesis prompt has perception section", "=== CNN PERCEPTION OUTPUT ===" in up)
check("synthesis prompt has memory section", "=== CONTEXTUAL MEMORY SOURCES ===" in up)
check("synthesis prompt embeds gap", "probability gap" in up)

# demo banner appears when demo_mode
demo_payload = dict(prompts.example_payload(), demo_mode=True)
check("demo banner injected in demo mode", "DEMO MODE" in prompts.build_synthesis_prompt(demo_payload))

# --------------------------------------------------------------------- #
print("\nC) BioGPT + Med-PaLM adapters (fallback/placeholder)")
from app.llm.biogpt_adapter import BioGPTAdapter
from app.llm.medpalm_adapter import MedPaLMAdapter

b = BioGPTAdapter(enabled=False).generate_context("Lung Cancer", "lung_squamous_cell_carcinoma")
check("BioGPT disabled -> tagged fallback", "[BioGPT fallback" in b, b[:60])
m = MedPaLMAdapter(enabled=False).generate_context("Lung Cancer", "lung_squamous_cell_carcinoma", None)
check("Med-PaLM disabled -> tagged PLACEHOLDER", "[Med-PaLM 2: PLACEHOLDER" in m, m[:60])
# unknown group still yields text
b2 = BioGPTAdapter(enabled=False).generate_context("Zzz Group", "zzz")
check("BioGPT unknown group still returns text", isinstance(b2, str) and len(b2) > 10)

# --------------------------------------------------------------------- #
print("\nD) Claude client — JSON extraction + graceful no-key")
from app.llm.claude_client import ClaudeClient, _extract_json

check("extract fenced ```json", _extract_json('```json\n{"a": 1}\n```') == {"a": 1})
check("extract bare object", _extract_json('{"b": 2}') == {"b": 2})
check("extract embedded object", _extract_json('noise {"c": 3} tail') == {"c": 3})
check("extract junk -> None", _extract_json("not json at all") is None)
cc = ClaudeClient(Settings())
check("no key -> Claude unavailable", cc.available is False)
check("interpret() returns None when unavailable", cc.interpret("sys", "user") is None)

# --------------------------------------------------------------------- #
print("\nE) Orchestrator — rule-based fallback (Claude off)")
from app.llm.orchestrator import ReasoningOrchestrator

orch = ReasoningOrchestrator(Settings())
prediction = {
    "demo_mode": False,
    "top_prediction": {"rank": 1, "label": "lung_squamous_cell_carcinoma", "probability": 0.913, "index": 24},
    "predictions": [
        {"rank": 1, "label": "lung_squamous_cell_carcinoma", "probability": 0.913, "index": 24},
        {"rank": 2, "label": "lung_adenocarcinoma", "probability": 0.051, "index": 22},
        {"rank": 3, "label": "lung_benign_tissue", "probability": 0.019, "index": 23},
    ],
}
res = orch.interpret(prediction, "a tightly localized focal region in the central portion (~11% high-salience area)",
                     {"age": "63", "sex": "M", "symptoms": "chronic cough, weight loss"})
required = ["headline", "model_finding", "confidence_assessment", "differential", "gradcam_correlation",
           "biomedical_context", "patient_considerations", "recommended_next_steps",
           "uncertainty_and_limitations", "disclaimer"]
check("fallback returns all required keys", all(k in res for k in required),
      str([k for k in required if k not in res]))
check("differential is a non-empty list of dicts",
      isinstance(res["differential"], list) and res["differential"] and isinstance(res["differential"][0], dict))
check("recommended_next_steps is a list", isinstance(res["recommended_next_steps"], list) and len(res["recommended_next_steps"]) >= 1)
check("evidence chain (_memory) attached with 3 sources",
      set(res["_memory"].keys()) == {"kb", "biogpt", "medpalm"})
check("_engine marked rule-based-fallback", res["_engine"] == "rule-based-fallback")
check("_sources_active reports claude False", res["_sources_active"]["claude"] is False)
check("high-confidence phrasing chosen", "High model confidence" in res["confidence_assessment"], res["confidence_assessment"])

# demo-mode fallback wording
demo_pred = dict(prediction, demo_mode=True)
res_demo = orch.interpret(demo_pred, "n/a", None)
check("demo-mode fallback flags simulated", "DEMO MODE" in res_demo["confidence_assessment"])

# --------------------------------------------------------------------- #
print("\nF) CNN classifier — DEMO MODE prediction (numpy+PIL)")
from PIL import Image
from app.ml.cnn_model import DiseaseClassifier

clf = DiseaseClassifier(Settings())
check("no TF -> demo_mode True", clf.demo_mode is True)
check("num_classes == 30", clf.num_classes == 30, str(clf.num_classes))
buf = io.BytesIO(); Image.new("RGB", (96, 96), (123, 45, 201)).save(buf, "PNG"); img_bytes = buf.getvalue()
p1 = clf.predict(img_bytes, top_k=5)
check("predict returns 5 ranked items", len(p1["predictions"]) == 5)
check("probabilities sum to ~1", abs(sum(p1["probabilities"]) - 1.0) < 1e-4, str(sum(p1["probabilities"])))
check("top label is a real class", p1["top_prediction"]["label"] in clf.class_names)
check("ranks are 1..5 in order", [d["rank"] for d in p1["predictions"]] == [1, 2, 3, 4, 5])
check("probabilities are descending",
      all(p1["predictions"][i]["probability"] >= p1["predictions"][i + 1]["probability"] for i in range(4)))
p2 = clf.predict(img_bytes, top_k=5)
check("demo predictions deterministic for same image", p1["probabilities"] == p2["probabilities"])
buf2 = io.BytesIO(); Image.new("RGB", (96, 96), (10, 220, 90)).save(buf2, "PNG")
p3 = clf.predict(buf2.getvalue(), top_k=5)
check("different image -> different distribution", p1["probabilities"] != p3["probabilities"])
check("input_array shape (1,224,224,3)", tuple(p1["input_array"].shape) == (1, 224, 224, 3), str(p1["input_array"].shape))
health = clf.health()
check("health() reports tensorflow_available False here", health["tensorflow_available"] is False)

# --------------------------------------------------------------------- #
print("\nG) Grad-CAM — no-model path + pure helpers")
import numpy as np
from app.ml import gradcam

g = gradcam.generate(None, None, b"", 224, None, 0)
check("no model -> available False, no overlay", g["available"] is False and g["overlay_png_b64"] is None)
hm = np.zeros((7, 7), dtype="float32"); hm[3, 3] = 1.0; hm[3, 4] = 0.9
desc = gradcam._describe_focus(hm)
check("_describe_focus returns a description string", isinstance(desc, str) and "portion" in desc, desc)
diffuse = gradcam._describe_focus(np.zeros((7, 7), dtype="float32"))
check("_describe_focus handles empty heatmap", "diffuse" in diffuse, diffuse)
jet = gradcam._jet(np.linspace(0, 1, 49).reshape(7, 7))
check("_jet -> HxWx3 uint8", jet.shape == (7, 7, 3) and jet.dtype == np.uint8)

# --------------------------------------------------------------------- #
print("\nH) PDF report — REAL render with reportlab")
from app.report.pdf_report import build_report

# a small real PNG for the Grad-CAM slot
cam_buf = io.BytesIO()
Image.new("RGB", (224, 224), (200, 60, 40)).save(cam_buf, "PNG")
import base64 as _b64
cam_b64 = _b64.b64encode(cam_buf.getvalue()).decode("ascii")

report_data = {
    "image_name": "sample_lung_histology.png",
    "demo_mode": False,
    "predictions": prediction["predictions"],
    "disease_info": disease_metadata.resolve("lung_squamous_cell_carcinoma").as_dict(),
    "interpretation": res,                       # reuse the orchestrator output from E
    "patient_context": {"age": "63", "sex": "M", "symptoms": "chronic cough, weight loss"},
    "gradcam_overlay_b64": cam_b64,
}
pdf = build_report(report_data)
check("PDF is bytes", isinstance(pdf, (bytes, bytearray)))
check("PDF magic header %PDF", pdf[:4] == b"%PDF", str(pdf[:8]))
check("PDF has EOF trailer", b"%%EOF" in pdf[-1024:])
check("PDF non-trivial size (>3KB)", len(pdf) > 3000, f"{len(pdf)} bytes")
out_pdf = ROOT / "verification_report.pdf"
out_pdf.write_bytes(pdf)
print(f"      wrote {out_pdf.name} ({len(pdf)} bytes)")

# also render a demo-mode report (banner path)
pdf_demo = build_report(dict(report_data, demo_mode=True, interpretation=res_demo))
check("demo-mode PDF also renders", pdf_demo[:4] == b"%PDF" and len(pdf_demo) > 3000)

# --------------------------------------------------------------------- #
print(f"\n=================  {PASS} passed, {FAIL} failed  =================")
sys.exit(1 if FAIL else 0)
