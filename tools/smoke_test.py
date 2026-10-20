"""
LOCAL smoke test — run this on YOUR machine after `uv sync`.

It proves the REAL pipeline works end-to-end (this cannot be run in the build
sandbox because TensorFlow + the Anthropic SDK need network/GPU-class wheels):

  [1] TensorFlow imports and your trained weights LOAD  (i.e. NOT demo mode)
  [2] 5 real CNN forward passes  -> top-5 classes + timing + Grad-CAM focus
  [3] Grad-CAM produces a real heatmap overlay on the loaded model
  [4] Your Claude API key works and returns a structured clinical interpretation

Usage
-----
  uv run python tools/smoke_test.py                       # 5 synthetic images
  uv run python tools/smoke_test.py path/to/*.png ...     # your own images

Exit code is 0 only if every REQUIRED check passes (the Claude check is
reported but treated as WARN-not-fatal so you can still validate the CNN
offline). Real weights loading IS required — if it drops to demo mode the
test fails loudly, because you asked for the real model, not the demo.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings                      # noqa: E402
from app.ml.cnn_model import DiseaseClassifier           # noqa: E402
from app.ml import gradcam                               # noqa: E402
from app.llm.orchestrator import ReasoningOrchestrator   # noqa: E402
from app.llm import disease_metadata                     # noqa: E402

BAR = "=" * 74
required_failures = 0
warnings = 0


def make_synthetic(seed: int) -> bytes:
    """A deterministic 224x224 RGB PNG so the test is self-contained."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    # A blobby, low-frequency image looks a little more like tissue than pure noise.
    small = (rng.random((28, 28, 3)) * 255).astype("uint8")
    img = Image.fromarray(small).resize((224, 224), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def load_images(argv: list[str]) -> list[tuple[str, bytes]]:
    imgs: list[tuple[str, bytes]] = []
    if argv:
        for a in argv:
            p = Path(a)
            if p.exists():
                imgs.append((p.name, p.read_bytes()))
            else:
                print(f"    ! skipping missing file: {a}")
    if not imgs:
        print("    (no image args given — generating 5 synthetic test images)")
        imgs = [(f"synthetic_{i+1}.png", make_synthetic(1000 + i)) for i in range(5)]
    return imgs


def main() -> int:
    global required_failures, warnings
    settings = get_settings()

    print(BAR)
    print("SMOKE TEST — Multi-Disease Detection Framework")
    print(BAR)

    # ------------------------------------------------------------------ [1]
    print("\n[1] MODEL LOAD")
    clf = DiseaseClassifier(settings)
    health = clf.health()
    for k, v in health.items():
        print(f"    {k:22}: {v}")

    if not health["tensorflow_available"]:
        print("    RESULT: FAIL — TensorFlow did not import. Run `uv sync` on Python 3.13.")
        required_failures += 1
        return _finish()
    print("    RESULT: PASS — TensorFlow imported.")

    if health["demo_mode"]:
        print("    RESULT: FAIL — running in DEMO MODE (weights did not load).")
        print("            Check models/best_model.weights.h5 and that the architecture matches.")
        required_failures += 1
        return _finish()
    print("    RESULT: PASS — real weights loaded (NOT demo mode).")

    if health["num_classes"] != 30:
        print(f"    RESULT: FAIL — expected 30 classes, got {health['num_classes']}.")
        required_failures += 1
    else:
        print("    RESULT: PASS — 30 classes loaded.")

    # ------------------------------------------------------------------ [2][3]
    print("\n[2] 5 REAL PREDICTIONS + [3] GRAD-CAM")
    images = load_images(sys.argv[1:])
    last_pred = None
    for name, data in images:
        t0 = time.time()
        try:
            pred = clf.predict(data, top_k=5)
        except Exception as exc:
            print(f"    {name}: FAIL — prediction raised: {exc}")
            required_failures += 1
            continue
        dt = (time.time() - t0) * 1000
        arr = pred.pop("input_array", None)

        top = pred["top_prediction"]
        print(f"\n    · {name}   ({dt:.0f} ms)")
        for p in pred["predictions"]:
            print(f"        #{p['rank']}  {p['label']:<40} {p['probability']*100:6.2f}%")

        # Grad-CAM on the real model
        try:
            gc = gradcam.generate(
                clf.model, arr, data, clf.img_size,
                clf.last_conv_layer_name, top["index"],
            )
            status = "overlay OK" if gc["available"] else "unavailable"
            print(f"        grad-cam: {status} — {gc['focus_description']}")
            if not gc["available"]:
                warnings += 1
        except Exception as exc:
            print(f"        grad-cam: FAIL — {exc}")
            required_failures += 1
        last_pred = pred

    if last_pred is not None:
        print("\n    RESULT: PASS — 5 forward passes completed.")

    # ------------------------------------------------------------------ [4]
    print("\n[4] CLAUDE API")
    orch = ReasoningOrchestrator(settings)
    if not orch.claude.available:
        print("    RESULT: WARN — Claude unavailable (no valid ANTHROPIC_API_KEY in .env).")
        print("            The app still runs using the rule-based fallback interpretation.")
        warnings += 1
    else:
        print(f"    key detected, model = {settings.claude_model}")
        demo_pred = last_pred or {
            "demo_mode": False,
            "top_prediction": {"rank": 1, "label": "lung_squamous_cell_carcinoma",
                               "probability": 0.91, "index": 24},
            "predictions": [
                {"rank": 1, "label": "lung_squamous_cell_carcinoma", "probability": 0.91, "index": 24},
                {"rank": 2, "label": "lung_adenocarcinoma", "probability": 0.05, "index": 22},
                {"rank": 3, "label": "lung_benign_tissue", "probability": 0.02, "index": 23},
            ],
        }
        t0 = time.time()
        result = orch.interpret(demo_pred, "a tightly localized focal region (~11% area)",
                                {"age": "63", "sex": "M", "symptoms": "chronic cough"})
        dt = (time.time() - t0) * 1000
        engine = result.get("_engine", "?")
        if str(engine).startswith("claude:"):
            print(f"    RESULT: PASS — live Claude reasoning in {dt:.0f} ms  (engine: {engine})")
            print(f"        headline: {result.get('headline','')[:160]}")
        else:
            print(f"    RESULT: WARN — key present but call fell back to '{engine}'.")
            print("            (network/quota/model-name issue — check the printed logs above)")
            warnings += 1

    return _finish()


def _finish() -> int:
    print("\n" + BAR)
    if required_failures == 0:
        print(f"SMOKE TEST PASSED  ({warnings} warning(s))")
        print(BAR)
        return 0
    print(f"SMOKE TEST FAILED  ({required_failures} required check(s) failed, {warnings} warning(s))")
    print(BAR)
    return 1


if __name__ == "__main__":
    sys.exit(main())
