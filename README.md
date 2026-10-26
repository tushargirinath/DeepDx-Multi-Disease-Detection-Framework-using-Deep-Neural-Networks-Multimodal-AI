# Multi-Disease Detection Framework — Deep Neural Networks & Multimodal AI

A hybrid clinical-decision-**support** framework that pairs a trained **custom CNN** (94.82% validation accuracy across **30 disease classes**) with a **high-speed LLM reasoning layer**.

The CNN acts as the *perception* engine; **Groq Cloud LLM** (`openai/gpt-oss-120b`) and **Claude** act as the *reasoning* engine, enriched by three **contextual-memory sources** — a structured Knowledge Base, **Microsoft BioGPT**, and **Med-PaLM 2** — to turn a raw softmax vector into a calibrated, source-attributed, safety-bounded interpretation.

Ships as an **interactive web application**: drag in a medical image → get the CNN prediction, a **Grad-CAM** attention heatmap, the LLM's contextual reasoning, a **prediction history log**, and a one-click **downloadable clinical PDF report**.

> ⚠️ **Research / educational prototype — not a medical device.** Every output is decision-support, never a diagnosis.

---

## 🌐 Live Production Cloud Deployment

The application is deployed live in **AWS Mumbai (`ap-south-1`)**:

| Attribute | Production Configuration |
|---|---|
| **Direct HTTP Web Console** | **[http://65.2.235.24:8000](http://65.2.235.24:8000)** ⚡ *(Direct Static IP Access)* |
| **Direct API Health Status** | **[http://65.2.235.24:8000/api/health](http://65.2.235.24:8000/api/health)** |
| **Domain (HTTPS / SSL)** | **[https://deepdx.myvnc.com](https://deepdx.myvnc.com)** 🔒 *(Automated Let's Encrypt TLS via Caddy)* |
| **Permanent Static IP** | `65.2.235.24` *(AWS Elastic IP)* |
| **Cloud Infrastructure** | AWS EC2 `m7i-flex.large` (2 vCPUs, 8 GB RAM, 4th Gen Intel Xeon) |
| **Model Storage** | AWS S3 (`deepdx-models-production-dbmbzc`), AES-256 encrypted |
| **SSL/TLS Reverse Proxy**| Caddy Reverse Proxy with automated Let's Encrypt SSL certificates |
| **Vision Triage Gate** | OpenRouter Multimodal Vision (`google/gemma-4-26b-a4b-it:free`) |
| **IaC Provisioning** | Terraform (`terraform/` directory) |
| **Cloud Region** | AWS Mumbai (`ap-south-1`) |

> 💡 **Tip:** If your local ISP (e.g., Airtel Broadband) blocks free dynamic DNS domains (`*.myvnc.com`), access the app directly via the static IP: **[http://65.2.235.24:8000](http://65.2.235.24:8000)** or enable Secure DNS (Google 8.8.8.8 / Cloudflare 1.1.1.1) in your browser settings.

---

## Repository Structure

```
DeepDx-Multi-Disease-Detection-Framework/
├── app/
│   ├── main.py                 # FastAPI application: /api/health, /api/analyze, /api/report
│   ├── config.py               # Central settings from .env (pydantic-settings)
│   ├── schemas.py              # Pydantic request/response models
│   ├── ml/
│   │   ├── cnn_model.py        # Custom 5-block CNN architecture, weight loading, predict()
│   │   └── gradcam.py          # Grad-CAM heatmap generator + plain-English localization
│   ├── llm/
│   │   ├── image_validator.py  # Pre-inference vision triage gate via OpenRouter (Gemma 4 free tier)
│   │   ├── groq_client.py      # Sub-second live clinical reasoning via Groq Cloud API
│   │   ├── claude_client.py    # Anthropic Claude 3.5/Sonnet integration + JSON extractor
│   │   ├── disease_metadata.py # Structured Knowledge Base for all 30 disease classes [KB]
│   │   ├── biogpt_adapter.py   # Microsoft BioGPT source adapter [BioGPT]
│   │   ├── medpalm_adapter.py  # Med-PaLM 2 clinical Q&A source adapter [Med-PaLM]
│   │   ├── orchestrator.py     # Coordinates memory sources + LLM reasoning synthesis
│   │   └── prompts.py          # Multimodal medical system & synthesis prompts
│   └── report/
│       └── pdf_report.py       # Clinical diagnostic PDF export (ReportLab)
├── frontend/                   # Modern clinical workstation console (Vanilla HTML/CSS/JS)
│   ├── index.html              # 2-column clinical dashboard (Acquisition & Readout)
│   ├── style.css               # Clean responsive styling with CSS tokens
│   └── app.js                  # Frontend state machine & API client
├── models/
│   ├── best_model.weights.h5   # Trained CNN weights (105.7 MB)
│   └── readme.md               # Model architecture & evaluation notes
├── terraform/                  # Production Infrastructure-as-Code (AWS Mumbai)
│   ├── main.tf                 # S3 bucket, IAM Role, Security Group, EC2, Elastic IP
│   ├── variables.tf            # Region (ap-south-1), instance type (m7i-flex.large)
│   ├── outputs.tf              # S3 URI, EC2 Static IP, live HTTPS URL
│   ├── user_data.sh.tpl        # Automated cloud-init bootstrapping script
│   ├── upload_model.bat        # Windows one-click model weights S3 uploader
│   ├── upload_model.sh         # Linux/Mac model weights S3 uploader
│   └── README.md               # Complete Terraform deployment guide
├── tools/
│   └── export_class_names_snippet.py   # Colab script to export class_names.json
├── class_names.json            # Ordered 30-class label list (index -> disease name)
├── docker-compose.yml          # Docker Compose production container configuration
├── Dockerfile                  # Multi-stage container build with Python 3.11 & uv
├── pyproject.toml              # uv/pip dependencies
├── .env.example                # Template configuration file
├── sample_output.md            # Sample clinical reasoning outputs
└── sample_prompt.md            # Sample multimodal prompts and LLM inputs
```

---

## 30 Supported Clinical Pathology Classes

The CNN classifier and medical knowledge base cover 30 classes across major organ systems:

- **Brain MRI**: `brain_glioma`, `brain_meningioma`, `brain_pituitary`
- **Alzheimer's Neuroimaging**: `Alzheimers_MildDemented`, `Alzheimers_ModerateDemented`, `Alzheimers_NonDemented`, `Alzheimers_VeryMildDemented`
- **Lung Histopathology**: `lung_adenocarcinoma`, `lung_benign_tissue`, `lung_squamous_cell_carcinoma`
- **Breast Histopathology**: `breast_benign`, `breast_malignant`
- **Colon Histopathology**: `colon_adenocarcinoma`, `colon_benign_tissue`
- **Kidney Histopathology**: `kidney_normal`, `kidney_tumor`
- **Oral Pathology**: `oral_normal`, `oral_squamous_cell_carcinoma`
- **Cervical Cytology**: `cervix_dyskeratotic`, `cervix_koilocytotic`, `cervix_metaplastic`, `cervix_parabasal`, `cervix_superficial_intermediate`
- **Lymphoma**: `lymph_chronic_lymphocytic_leukemia`, `lymph_follicular_lymphoma`, `lymph_mantle_cell_lymphoma`
- **Acute Lymphoblastic Leukemia**: `all_benign`, `all_early`, `all_pre`, `all_pro`

---

## 📊 Dataset Access

The deep learning model is trained and evaluated on the **Multi-Cancer Dataset** containing histopathology, cytology, MRI, and CT imaging across all 30 disease classes:

- **Kaggle Dataset**: **[Multi-Cancer Dataset (Version 1) on Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/versions/1/data)**  
- **Direct Access URL**: [https://www.kaggle.com/datasets/obulisainaren/multi-cancer/versions/1/data](https://www.kaggle.com/datasets/obulisainaren/multi-cancer/versions/1/data)  
- **Dataset Author**: Obulisainaren  
- **Modalities Included**: Histopathology (H&E), Brain MRI, Kidney CT, Oral biopsy, Cervical cytology, Blood smear  
- **Classes**: 30 distinct categories across oncology and neurology  

---

## How the Hybrid Reasoning Pipeline Works

```
   Uploaded Image
         │
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Multimodal Vision Triage Gate (Google Gemma 4 / OpenRouter) │
 └───────────────────────────────┬──────────────────────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼ (Invalid non-medical image)               ▼ (Valid medical image)
    HTTP 422 Rejection                          Custom CNN (Perception)
 (Reason + Modality Guidance)                          │
                                                       ▼
                                         Top-k Probabilities + Grad-CAM Heatmap
                                                       │
                 ┌─────────────────────────────────────┼─────────────────────────────────────┐
                 ▼                                     ▼                                     ▼
           [KB] Knowledge Base                   [BioGPT] Biomedical                   [Med-PaLM] Clinical
      (organ, modality, severity)                narrative context                     workup context
                 └─────────────────────────────────────┼─────────────────────────────────────┘
                                                       ▼
                                        Groq LLM Reasoning Synthesis
                                           (openai/gpt-oss-120b)
                                                       ▼
                              Calibrated, Source-Attributed Clinical Report
                                                       ▼
                             Web Console  +  Downloadable Clinical PDF Report
```

---

## Local Development Setup

### 1. Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (fast Python package manager) or Docker

### 2. Clone & Install
```bash
git clone https://github.com/tushargirinath/DeepDx-Multi-Disease-Detection-Framework-using-Deep-Neural-Networks-Multimodal-AI.git
cd DeepDx-Multi-Disease-Detection-Framework

# Install dependencies using uv
uv sync
```

### 3. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and add your **Groq Cloud API Key** (free at [console.groq.com](https://console.groq.com)):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
CNN_WEIGHTS_PATH=models/best_model.weights.h5
CLASS_NAMES_PATH=class_names.json
```

### 4. Run the Server
```bash
# Direct execution:
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or via Docker Compose:
docker compose up -d --build
```
Open **`http://localhost:8000`** in your browser.

---

## AWS Cloud Deployment (Terraform)

The complete AWS infrastructure is automated via Terraform in the [terraform/](terraform/) directory:

```bash
cd terraform

# 1. Initialize AWS providers
terraform init

# 2. Preview resources (S3, EC2, IAM, Security Group, Elastic IP)
terraform plan

# 3. Provision infrastructure in Mumbai (ap-south-1)
terraform apply

# 4. Upload model weights to S3
./upload_model.bat    # Windows
./upload_model.sh     # Linux / Mac
```

For complete cloud deployment documentation, see [terraform/README.md](terraform/README.md).

---

## REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web clinical console UI |
| `GET` | `/api/health` | Returns health status, loaded weights info, and LLM availability |
| `POST` | `/api/analyze` | Multipart medical image + patient fields → CNN prediction, Grad-CAM, LLM reasoning |
| `POST` | `/api/report` | Analysis JSON payload → Downloadable clinical PDF report |

---

## License & Disclaimer

This project is licensed under the MIT License.  
**Educational and research decision-support prototype only. Not certified as a medical device for independent diagnostic use.**
