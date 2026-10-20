/* =====================================================================
   Multi-Disease Detection — Diagnostic Console (client logic)
   ===================================================================== */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  file: null,
  originalDataUrl: null,
  last: null,      // last analysis response
  history: [],     // session log
};

/* ---------- helpers ---------- */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
function sevKey(info) {
  const s = (info && info.severity_hint || "").toLowerCase();
  if (s === "high") return "malignant";
  if (s === "moderate") return "precursor";
  if (s === "low") return "benign";
  return "normal";
}

/* ---------- health / status ---------- */
async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const pills = [];
    if (!h.model.tensorflow_available)
      pills.push(["pill-warn", "tensorflow off"]);
    if (h.model.demo_mode) pills.push(["pill-warn", "demo mode"]);
    else pills.push(["pill-ok", `model · ${h.model.num_classes} classes`]);
    if (h.llm.groq_available) {
      const shortName = String(h.llm.groq_model || "").split("/").pop();
      pills.push(["pill-live", `groq · ${shortName}`]);
    } else {
      pills.push(["pill-muted", "groq offline"]);
    }
    if (h.llm.biogpt_enabled) pills.push(["pill-ok", "biogpt live"]);
    renderPills(pills);
  } catch (e) {
    renderPills([["pill-warn", "server unreachable"]]);
  }
}
function renderPills(items) {
  const box = $("statusPills");
  box.innerHTML = "";
  items.forEach(([cls, txt]) => {
    const p = el("span", `pill ${cls}`, txt);
    box.appendChild(p);
  });
}

/* ---------- file handling ---------- */
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    state.originalDataUrl = e.target.result;
    $("previewImg").src = e.target.result;
    $("dzEmpty").hidden = true;
    $("dzPreview").hidden = false;
    $("analyzeBtn").disabled = false;
  };
  reader.readAsDataURL(file);
}
function clearFile() {
  state.file = null;
  state.originalDataUrl = null;
  $("fileInput").value = "";
  $("dzEmpty").hidden = false;
  $("dzPreview").hidden = true;
  $("analyzeBtn").disabled = true;
}

/* ---------- analyze ---------- */
async function analyze() {
  if (!state.file) return;
  const btn = $("analyzeBtn");
  btn.disabled = true; btn.classList.add("loading");
  $("emptyState").hidden = true;
  $("results").hidden = true;
  $("analyzing").hidden = false;

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("top_k", $("topk").value);
  fd.append("age", $("age").value);
  fd.append("sex", $("sex").value);
  fd.append("symptoms", $("symptoms").value);
  fd.append("history", $("history").value);
  fd.append("notes", $("notes").value);

  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || "Analysis failed");
    }
    const data = await r.json();
    state.last = data;
    renderResults(data);
    pushHistory(data);
  } catch (e) {
    $("analyzing").hidden = true;
    $("emptyState").hidden = false;
    $("emptyState").innerHTML =
      `<div class="es-glyph"></div><p><b>Something went wrong.</b><br>${e.message}</p>`;
  } finally {
    btn.disabled = false; btn.classList.remove("loading");
  }
}

/* ---------- render ---------- */
function renderResults(data) {
  $("analyzing").hidden = true;
  $("emptyState").hidden = true;
  $("results").hidden = false;

  $("demoBanner").hidden = !data.demo_mode;

  const info = data.disease_info || {};
  const top = data.top_prediction;
  const verdict = $("results").querySelector(".verdict");
  verdict.dataset.sev = sevKey(info);
  $("topLabel").textContent = top.label;
  $("topMeta").textContent =
    [info.disease_group, info.organ_system, info.modality, info.category]
      .filter(Boolean).join("  ·  ");
  $("topConf").textContent = (top.probability * 100).toFixed(1) + "%";

  // ladder
  const ladder = $("ladder"); ladder.innerHTML = "";
  const sevColor = getComputedStyle(document.documentElement)
    .getPropertyValue("--sev-" + sevKey(info)).trim() || "#0E9AA7";
  data.predictions.forEach((p, i) => {
    const rung = el("div", "rung");
    const track = el("div", "rung-track");
    const fill = el("div", "rung-fill");
    fill.style.width = Math.max(2, p.probability * 100) + "%";
    if (i === 0) fill.style.background = sevColor;
    const name = el("div", "rung-name", p.label);
    track.appendChild(fill); track.appendChild(name);
    const pct = el("div", "rung-pct", (p.probability * 100).toFixed(1) + "%");
    rung.appendChild(track); rung.appendChild(pct);
    ladder.appendChild(rung);
  });

  // grad-cam
  const cam = data.gradcam || {};
  const camImg = $("camImg"), camNone = $("camNone"), camToggle = $("camToggle");
  if (cam.overlay_png_b64) {
    state.overlayUrl = "data:image/png;base64," + cam.overlay_png_b64;
    camImg.src = state.overlayUrl;
    camImg.hidden = false; camNone.hidden = true; camToggle.hidden = false;
    setCamView("overlay");
  } else {
    camImg.hidden = true; camToggle.hidden = true;
    camNone.hidden = false;
    camNone.textContent = cam.focus_description || "Grad-CAM unavailable in demo mode.";
  }
  $("camFocus").innerHTML = cam.focus_description
    ? `<b>Model focus:</b> ${cam.focus_description}` : "";

  // evidence chain
  const interp = data.interpretation || {};
  const mem = interp._memory || {};
  $("memKb").textContent = mem.kb || "—";
  $("memBio").textContent = mem.biogpt || "—";
  $("memMed").textContent = mem.medpalm || "—";
  $("engineLine").textContent = interp._engine || "—";

  // interpretation
  renderInterpretation(interp);

  const t = data.timing_ms || {};
  $("timing").textContent = `perception ${t.prediction || 0} ms · reasoning ${t.llm || 0} ms`;
  $("disclaimer").textContent = interp.disclaimer ||
    "Research decision-support only — not a medical diagnosis.";
}

function renderInterpretation(interp) {
  const box = $("interp"); box.innerHTML = "";
  if (interp.headline) box.appendChild(el("div", "i-headline", interp.headline));

  const sec = (title, body) => {
    if (!body) return;
    const s = el("div", "i-sec");
    s.appendChild(el("h4", null, title));
    s.appendChild(el("p", null, body));
    box.appendChild(s);
  };
  const listSec = (title, items, isDiff) => {
    if (!items || !items.length) return;
    const s = el("div", "i-sec");
    s.appendChild(el("h4", null, title));
    const ul = el("ul");
    items.forEach((it) => {
      const li = el("li");
      if (isDiff && typeof it === "object") {
        const b = el("b", null, it.condition || "");
        li.appendChild(b);
        li.appendChild(document.createTextNode(" — " + (it.rationale || "")));
      } else {
        li.textContent = typeof it === "object" ? JSON.stringify(it) : it;
      }
      ul.appendChild(li);
    });
    s.appendChild(ul); box.appendChild(s);
  };

  sec("Model finding", interp.model_finding);
  sec("Confidence assessment", interp.confidence_assessment);
  listSec("Differential", interp.differential, true);
  sec("Grad-CAM correlation", interp.gradcam_correlation);
  sec("Biomedical context", interp.biomedical_context);
  sec("Patient considerations", interp.patient_considerations);
  listSec("Recommended next steps", interp.recommended_next_steps, false);
  listSec("Uncertainty & limitations", interp.uncertainty_and_limitations, false);
}

function setCamView(view) {
  const camImg = $("camImg");
  camImg.src = view === "original" && state.originalDataUrl
    ? state.originalDataUrl : state.overlayUrl;
  document.querySelectorAll("#camToggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view));
}

/* ---------- history ---------- */
function pushHistory(data) {
  const info = data.disease_info || {};
  const entry = {
    thumb: state.originalDataUrl,
    label: data.top_prediction.label,
    conf: (data.top_prediction.probability * 100).toFixed(1) + "%",
    sev: sevKey(info),
    demo: data.demo_mode,
    data,
  };
  state.history.unshift(entry);
  if (state.history.length > 12) state.history.pop();
  renderHistory();
}
function renderHistory() {
  const panel = $("historyPanel"), list = $("histList");
  if (!state.history.length) { panel.hidden = true; return; }
  panel.hidden = false; list.innerHTML = "";
  state.history.forEach((h, idx) => {
    const li = el("li", "hist-item");
    const dot = el("span", "hist-dot");
    dot.style.background = getComputedStyle(document.documentElement)
      .getPropertyValue("--sev-" + h.sev).trim();
    const img = el("img", "hist-thumb"); if (h.thumb) img.src = h.thumb;
    const body = el("div", "hist-body");
    body.appendChild(el("div", "hl", h.label));
    body.appendChild(el("div", "hs", h.conf + (h.demo ? " · demo" : "")));
    li.appendChild(dot); li.appendChild(img); li.appendChild(body);
    li.addEventListener("click", () => {
      state.last = h.data; state.originalDataUrl = h.thumb;
      renderResults(h.data);
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    list.appendChild(li);
  });
}

/* ---------- PDF export ---------- */
async function exportPdf() {
  if (!state.last) return;
  const d = state.last;
  const btn = $("exportBtn");
  const old = btn.textContent; btn.textContent = "Building PDF…"; btn.disabled = true;
  try {
    const body = {
      image_name: d.image_name || "uploaded_image",
      demo_mode: d.demo_mode,
      predictions: d.predictions,
      disease_info: d.disease_info,
      interpretation: d.interpretation,
      patient_context: d.patient_context || {},
      gradcam_overlay_b64: (d.gradcam && d.gradcam.overlay_png_b64) || null,
    };
    const r = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error("report failed");
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "disease_detection_report.pdf";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert("Could not generate the PDF report: " + e.message);
  } finally {
    btn.textContent = old; btn.disabled = false;
  }
}

/* ---------- wire up ---------- */
function init() {
  loadHealth();

  const dz = $("dropzone"), fi = $("fileInput");
  fi.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  $("clearImg").addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); clearFile(); });
  $("analyzeBtn").addEventListener("click", analyze);
  $("exportBtn").addEventListener("click", exportPdf);
  document.querySelectorAll("#camToggle button").forEach((b) =>
    b.addEventListener("click", () => setCamView(b.dataset.view)));
}
document.addEventListener("DOMContentLoaded", init);
