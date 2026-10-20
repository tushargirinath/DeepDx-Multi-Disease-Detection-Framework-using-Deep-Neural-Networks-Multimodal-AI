"""
Clinical-style PDF report generation with reportlab.

Renders the prediction, confidence ladder, Grad-CAM overlay, patient context,
and the LLM interpretation into a downloadable A4 report.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF  # noqa: F401  (ensures graphics backend loads)

INK = colors.HexColor("#0F2438")
TEAL = colors.HexColor("#0E8B96")
AMBER = colors.HexColor("#C98A1B")
RED = colors.HexColor("#C0392B")
GREY = colors.HexColor("#5B6B7B")
LIGHT = colors.HexColor("#EEF3F7")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Heading1"], textColor=INK, fontSize=18, spaceAfter=2))
    ss.add(ParagraphStyle("Subx", parent=ss["Normal"], textColor=GREY, fontSize=9, spaceAfter=8))
    ss.add(ParagraphStyle("Secx", parent=ss["Heading2"], textColor=TEAL, fontSize=11,
                          spaceBefore=10, spaceAfter=3))
    ss.add(ParagraphStyle("Bodyx", parent=ss["Normal"], fontSize=9.5, leading=13, alignment=TA_LEFT))
    ss.add(ParagraphStyle("Smallx", parent=ss["Normal"], fontSize=8, textColor=GREY, leading=10))
    ss.add(ParagraphStyle("Discx", parent=ss["Normal"], fontSize=8.5, textColor=RED, leading=11))
    return ss


def _severity_color(category: str) -> colors.Color:
    c = (category or "").lower()
    if "malignant" in c or ("neuro" in c and "degenerative" in c):
        return RED
    if "precursor" in c:
        return AMBER
    if "benign" in c:
        return TEAL
    return GREY


def _prob_bar(prob: float, width=70 * mm, height=6 * mm) -> Drawing:
    d = Drawing(width, height)
    d.add(Rect(0, 0, width, height, fillColor=LIGHT, strokeColor=None))
    fill = colors.HexColor("#0E8B96") if prob >= 0.5 else colors.HexColor("#8FB3BE")
    d.add(Rect(0, 0, max(1.0, width * float(prob)), height, fillColor=fill, strokeColor=None))
    d.add(String(width - 2, height / 2 - 3, f"{prob*100:.1f}%", textAnchor="end", fontSize=8, fillColor=INK))
    return d


def _para_list(items, style) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(str(x), style), leftIndent=6) for x in items],
        bulletType="bullet", start="•", leftIndent=10,
    )


def build_report(data: dict) -> bytes:
    """data mirrors schemas.ReportRequest (as dict)."""
    ss = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
        title="Multi-Disease Detection Report",
    )
    flow = []

    # ---- Header ----
    flow.append(Paragraph("Multi-Disease Detection Report", ss["H1x"]))
    flow.append(Paragraph(
        f"CNN + LLM contextual reasoning &nbsp;•&nbsp; generated {datetime.now():%Y-%m-%d %H:%M}",
        ss["Subx"],
    ))

    if data.get("demo_mode"):
        flow.append(Table(
            [[Paragraph("DEMO MODE — prediction is simulated (model weights not loaded). "
                        "Not for any clinical interpretation.", ss["Smallx"])]],
            colWidths=[178 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBECEC")),
                ("BOX", (0, 0), (-1, -1), 0.5, RED),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]),
        ))
        flow.append(Spacer(1, 6))

    # ---- Patient context ----
    pc = data.get("patient_context") or {}
    pc = {k: v for k, v in pc.items() if v}
    flow.append(Paragraph("Patient context", ss["Secx"]))
    if pc:
        rows = [[Paragraph(f"<b>{k.title()}</b>", ss["Bodyx"]), Paragraph(str(v), ss["Bodyx"])]
                for k, v in pc.items()]
        t = Table(rows, colWidths=[35 * mm, 143 * mm])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, -2), 0.25, LIGHT),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        flow.append(t)
    else:
        flow.append(Paragraph("None provided.", ss["Bodyx"]))

    # ---- Prediction ladder ----
    info = data.get("disease_info") or {}
    preds = data.get("predictions") or []
    flow.append(Paragraph("Model prediction", ss["Secx"]))
    if preds:
        top = preds[0]
        chip = _severity_color(info.get("category", ""))
        head = Table(
            [[Paragraph(f"<b>{top['label']}</b>", ss["Bodyx"]),
              Paragraph(f"<font color='white'><b>&nbsp;{info.get('category','—')}&nbsp;</b></font>", ss["Bodyx"])]],
            colWidths=[140 * mm, 38 * mm],
        )
        head.setStyle(TableStyle([
            ("BACKGROUND", (1, 0), (1, 0), chip),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        flow.append(head)
        meta = (f"{info.get('disease_group','')} · {info.get('organ_system','')} · {info.get('modality','')}")
        flow.append(Paragraph(meta, ss["Smallx"]))
        flow.append(Spacer(1, 4))

        table_rows = [[Paragraph("<b>#</b>", ss["Smallx"]),
                       Paragraph("<b>Condition</b>", ss["Smallx"]),
                       Paragraph("<b>Confidence</b>", ss["Smallx"])]]
        for p in preds[:5]:
            table_rows.append([
                Paragraph(str(p["rank"]), ss["Bodyx"]),
                Paragraph(p["label"], ss["Bodyx"]),
                _prob_bar(p["probability"]),
            ])
        pt = Table(table_rows, colWidths=[10 * mm, 90 * mm, 78 * mm])
        pt.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        flow.append(pt)

    # ---- Grad-CAM ----
    b64 = data.get("gradcam_overlay_b64")
    if b64:
        try:
            img_bytes = base64.b64decode(b64)
            flow.append(Paragraph("Grad-CAM (model attention)", ss["Secx"]))
            flow.append(RLImage(io.BytesIO(img_bytes), width=55 * mm, height=55 * mm))
        except Exception:
            pass

    # ---- Interpretation ----
    interp = data.get("interpretation") or {}
    flow.append(Paragraph("LLM contextual interpretation", ss["Secx"]))

    def add_text(label, key):
        val = interp.get(key)
        if val:
            flow.append(Paragraph(f"<b>{label}.</b> {val}", ss["Bodyx"]))
            flow.append(Spacer(1, 2))

    if interp.get("headline"):
        flow.append(Paragraph(f"<b>{interp['headline']}</b>", ss["Bodyx"]))
        flow.append(Spacer(1, 3))
    add_text("Model finding", "model_finding")
    add_text("Confidence", "confidence_assessment")

    diff = interp.get("differential") or []
    if diff:
        flow.append(Paragraph("<b>Differential</b>", ss["Bodyx"]))
        items = [f"<b>{d.get('condition','')}</b> — {d.get('rationale','')}" if isinstance(d, dict) else str(d)
                 for d in diff]
        flow.append(_para_list(items, ss["Bodyx"]))

    add_text("Grad-CAM correlation", "gradcam_correlation")
    add_text("Biomedical context", "biomedical_context")
    add_text("Patient considerations", "patient_considerations")

    steps = interp.get("recommended_next_steps") or []
    if steps:
        flow.append(Paragraph("<b>Recommended next steps</b>", ss["Bodyx"]))
        flow.append(_para_list(steps, ss["Bodyx"]))

    limits = interp.get("uncertainty_and_limitations") or []
    if limits:
        flow.append(Paragraph("<b>Uncertainty &amp; limitations</b>", ss["Bodyx"]))
        flow.append(_para_list(limits, ss["Bodyx"]))

    flow.append(Spacer(1, 8))
    disc = interp.get("disclaimer") or "Research decision-support only — not a medical diagnosis."
    flow.append(Paragraph(f"⚠ {disc}", ss["Discx"]))

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(16 * mm, 10 * mm,
                      "Multi-Disease Detection Framework — CNN + LLM contextual reasoning · research/educational use only")
    canvas.drawRightString(194 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()
