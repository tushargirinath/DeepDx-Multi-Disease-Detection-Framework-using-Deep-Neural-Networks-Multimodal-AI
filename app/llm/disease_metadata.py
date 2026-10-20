"""
Structured disease knowledge base — the *structured* contextual-memory source.

Each of the 30 classes resolves to metadata (organ system, imaging modality,
clinical category, severity hint, short description). If a class name is not in
the curated table, we infer sensible values from its prefix so the app stays
robust even if your exact leaf-label spellings differ from the reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class DiseaseInfo:
    class_name: str
    disease_group: str
    organ_system: str
    modality: str
    category: str          # Normal | Benign | Precursor | Malignant | Neurodegenerative
    severity_hint: str     # low | moderate | high | informational
    description: str

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------- prefix-level defaults ----------------------- #
_PREFIX = {
    "Alzheimers_": ("Alzheimer's Disease (dementia staging)", "Neurological / Brain", "Brain MRI", "Neurodegenerative"),
    "all_": ("Acute Lymphoblastic Leukemia (ALL)", "Hematologic / Bone marrow & blood", "Peripheral blood smear microscopy", "Malignant"),
    "brain_": ("Brain Tumor", "Neurological / Brain", "Brain MRI", "Malignant"),
    "breast_": ("Breast Cancer", "Breast", "Histopathology (H&E)", "Malignant"),
    "cervix_": ("Cervical Cytology (SIPaKMeD)", "Cervix / Gynecologic", "Pap smear cytology", "Precursor"),
    "colon_": ("Colon Cancer", "Gastrointestinal / Colon", "Histopathology (H&E)", "Malignant"),
    "kidney_": ("Kidney Disease", "Renal / Kidney", "CT imaging", "Malignant"),
    "lung_": ("Lung Cancer", "Respiratory / Lung", "Histopathology (H&E)", "Malignant"),
    "lymph_": ("Malignant Lymphoma", "Hematologic / Lymphatic", "Histopathology (H&E)", "Malignant"),
    "oral_": ("Oral Cancer", "Head & Neck / Oral cavity", "Histopathology (H&E)", "Malignant"),
}

# ----------------------- curated per-class table ----------------------- #
_CURATED: dict[str, tuple] = {
    # Alzheimer's staging (MRI) — informational neurodegenerative staging
    "Alzheimers_NonDemented": ("Normal", "informational", "No radiological signs of dementia-related atrophy on MRI."),
    "Alzheimers_VeryMildDemented": ("Neurodegenerative", "low", "Earliest stage; subtle medial-temporal / hippocampal changes may appear."),
    "Alzheimers_MildDemented": ("Neurodegenerative", "moderate", "Mild-stage Alzheimer's pattern with measurable atrophy."),
    "Alzheimers_ModerateDemented": ("Neurodegenerative", "high", "More pronounced cortical/hippocampal atrophy consistent with moderate dementia."),
    # Acute Lymphoblastic Leukemia (blood smear)
    "all_benign": ("Benign", "low", "Hematogones / benign lymphoid cells — no leukemic blasts identified."),
    "all_early": ("Precursor", "moderate", "Early-stage lymphoblast morphology suggestive of early ALL."),
    "all_pre": ("Malignant", "high", "Pre-B stage lymphoblasts consistent with ALL."),
    "all_pro": ("Malignant", "high", "Pro-B stage lymphoblasts consistent with more advanced ALL."),
    # Brain tumor (MRI)
    "brain_glioma": ("Malignant", "high", "Glioma — tumor of glial cells, often infiltrative."),
    "brain_meningioma": ("Malignant", "moderate", "Meningioma — usually slower-growing tumor of the meninges."),
    "brain_pituitary": ("Malignant", "moderate", "Pituitary tumor affecting the sellar region / hormonal axis."),
    # Breast (histopathology)
    "breast_benign": ("Benign", "low", "Benign breast tissue — no malignant features."),
    "breast_malignant": ("Malignant", "high", "Malignant breast carcinoma features present."),
    # Cervical cytology (SIPaKMeD)
    "cervix_dyskeratotic": ("Precursor", "moderate", "Dyskeratotic squamous cells — often HPV-associated, abnormal keratinization."),
    "cervix_koilocytotic": ("Precursor", "moderate", "Koilocytes — perinuclear halos indicating HPV cytopathic effect."),
    "cervix_metaplastic": ("Benign", "low", "Metaplastic squamous cells — typically benign transformation-zone change."),
    "cervix_parabasal": ("Normal", "informational", "Parabasal cells — normal immature squamous epithelium."),
    "cervix_superficial_intermediate": ("Normal", "informational", "Superficial/intermediate squamous cells — normal mature epithelium."),
    # Colon (histopathology, LC25000)
    "colon_adenocarcinoma": ("Malignant", "high", "Colon adenocarcinoma — malignant glandular tissue."),
    "colon_benign_tissue": ("Benign", "low", "Benign colonic tissue — no malignant features."),
    # Kidney (CT)
    "kidney_normal": ("Normal", "informational", "Normal kidney parenchyma on CT."),
    "kidney_tumor": ("Malignant", "high", "Renal mass/tumor identified on CT."),
    # Lung (histopathology, LC25000)
    "lung_adenocarcinoma": ("Malignant", "high", "Lung adenocarcinoma — malignant glandular tissue."),
    "lung_benign_tissue": ("Benign", "low", "Benign lung tissue — no malignant features."),
    "lung_squamous_cell_carcinoma": ("Malignant", "high", "Lung squamous cell carcinoma."),
    # Lymphoma (histopathology)
    "lymph_chronic_lymphocytic_leukemia": ("Malignant", "high", "CLL/SLL — small mature B-cell lymphoproliferative disorder."),
    "lymph_follicular_lymphoma": ("Malignant", "high", "Follicular lymphoma — nodular B-cell lymphoma."),
    "lymph_mantle_cell_lymphoma": ("Malignant", "high", "Mantle cell lymphoma — aggressive B-cell lymphoma."),
    # Oral (histopathology)
    "oral_normal": ("Normal", "informational", "Normal oral mucosa — no malignant features."),
    "oral_squamous_cell_carcinoma": ("Malignant", "high", "Oral squamous cell carcinoma."),
}


def _infer_from_prefix(class_name: str) -> tuple:
    for prefix, (group, organ, modality, category) in _PREFIX.items():
        if class_name.startswith(prefix):
            return group, organ, modality, category
    return ("Unknown condition", "Unspecified", "Medical image", "Malignant")


def _infer_category_severity(class_name: str) -> tuple:
    n = class_name.lower()
    if any(k in n for k in ("malignant", "carcinoma", "glioma", "lymphoma", "leukemia", "tumor", "adenocarcinoma", "_pre", "_pro")):
        return "Malignant", "high"
    if "moderate" in n:
        return "Neurodegenerative", "high"
    if any(k in n for k in ("normal", "nondemented", "benign_tissue")) or n.endswith("_normal"):
        return "Normal", "informational"
    if "benign" in n:
        return "Benign", "low"
    if any(k in n for k in ("early", "pre", "dyskeratotic", "koilocytotic")):
        return "Precursor", "moderate"
    return "Malignant", "moderate"


def resolve(class_name: str) -> DiseaseInfo:
    group, organ, modality, prefix_category = _infer_from_prefix(class_name)

    if class_name in _CURATED:
        category, severity, description = _CURATED[class_name]
    else:
        category, severity = _infer_category_severity(class_name)
        pretty = class_name.replace("_", " ")
        description = f"{pretty} — (auto-generated description; add to _CURATED for a curated note)."

    return DiseaseInfo(
        class_name=class_name,
        disease_group=group,
        organ_system=organ,
        modality=modality,
        category=category,
        severity_hint=severity,
        description=description,
    )
