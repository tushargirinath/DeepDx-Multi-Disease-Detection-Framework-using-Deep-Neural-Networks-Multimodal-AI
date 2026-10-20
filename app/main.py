"""
FastAPI application.

Routes:
  GET  /                -> web UI (served from ./frontend)
  GET  /api/health      -> model + LLM source status
  POST /api/analyze     -> multipart image (+patient fields) -> prediction, Grad-CAM, interpretation
  POST /api/report      -> JSON analysis -> downloadable clinical PDF
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR, get_settings
from app.llm import disease_metadata
from app.llm.orchestrator import ReasoningOrchestrator
from app.ml import gradcam
from app.ml.cnn_model import DiseaseClassifier
from app.report.pdf_report import build_report
from app.schemas import ReportRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-8s  %(message)s")
logger = logging.getLogger("app")

FRONTEND_DIR = ROOT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Loading CNN classifier…")
    app.state.settings = settings
    app.state.classifier = DiseaseClassifier(settings)
    app.state.orchestrator = ReasoningOrchestrator(settings)
    logger.info("Ready. %s", app.state.classifier.health())
    yield


app = FastAPI(title="Multi-Disease Detection Framework", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health():
    clf = app.state.classifier
    s = app.state.settings
    return {
        "status": "ok",
        "model": clf.health(),
        "class_names": clf.class_names,
        "llm": {
            "groq_available": app.state.orchestrator.groq.available,
            "groq_model": s.groq_model if app.state.orchestrator.groq.available else None,
            "biogpt_enabled": s.enable_biogpt,
            "medpalm_enabled": s.enable_medpalm,
        },
    }


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    age: str = Form(""),
    sex: str = Form(""),
    symptoms: str = Form(""),
    history: str = Form(""),
    notes: str = Form(""),
):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    clf = app.state.classifier
    orch = app.state.orchestrator

    t0 = time.time()
    try:
        pred = clf.predict(image_bytes, top_k=max(1, min(top_k, clf.num_classes)))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Could not read/parse image: {exc}")

    input_array = pred.pop("input_array", None)
    t_pred = time.time() - t0

    # Grad-CAM for the top class.
    gc = gradcam.generate(
        clf.model, input_array, image_bytes, clf.img_size,
        clf.last_conv_layer_name, pred["top_prediction"]["index"],
    )

    patient = {k: v for k, v in
               {"age": age, "sex": sex, "symptoms": symptoms, "history": history, "notes": notes}.items() if v}

    t1 = time.time()
    interpretation = orch.interpret(pred, gc["focus_description"], patient or None)
    t_llm = time.time() - t1

    disease_info = disease_metadata.resolve(pred["top_prediction"]["label"]).as_dict()

    return JSONResponse({
        "demo_mode": pred["demo_mode"],
        "image_name": file.filename,
        "top_prediction": pred["top_prediction"],
        "predictions": pred["predictions"],
        "disease_info": disease_info,
        "gradcam": {
            "overlay_png_b64": gc["overlay_png_b64"],
            "focus_description": gc["focus_description"],
            "available": gc["available"],
        },
        "interpretation": interpretation,
        "patient_context": patient,
        "timing_ms": {"prediction": round(t_pred * 1000), "llm": round(t_llm * 1000)},
    })


@app.post("/api/report")
def report(req: ReportRequest):
    data = req.model_dump()
    if req.patient_context:
        data["patient_context"] = req.patient_context.non_empty()
    try:
        pdf = build_report(data)
    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="disease_detection_report.pdf"'},
    )


# ---- Frontend (registered LAST so /api routes win) ----
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
