"""
CrackVision AI — Premium Streamlit Application
-----------------------------------------------
Drop-in replacement for the repository's app.py.

Expected model location:
    models/best.pt

Run locally:
    streamlit run app.py

This application provides AI-assisted visual screening only. It is not a
substitute for an inspection by a qualified structural engineer.
"""

from __future__ import annotations

import io
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Keep the UI available even when an optional native dependency (commonly
# OpenCV/cv2) fails to import. The diagnostic is shown inside the app.
YOLO = None
YOLO_IMPORT_ERROR = None
try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
except Exception as exc:
    YOLO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="CrackVision AI | Concrete Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# THEME & GLOBAL STYLES
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --cv-bg: #08111f;
        --cv-panel: #101d30;
        --cv-panel-2: #14243a;
        --cv-border: rgba(148, 163, 184, .16);
        --cv-text: #edf4ff;
        --cv-muted: #91a4bd;
        --cv-accent: #39d6c2;
        --cv-blue: #6ba7ff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 78% 0%, rgba(28, 112, 145, .16), transparent 28%),
            radial-gradient(circle at 0% 35%, rgba(50, 92, 160, .10), transparent 28%),
            var(--cv-bg);
        color: var(--cv-text);
    }

    [data-testid="stHeader"] { background: rgba(8, 17, 31, .72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a2b 0%, #0a1422 100%);
        border-right: 1px solid var(--cv-border);
    }
    [data-testid="stSidebar"] * { color: var(--cv-text); }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -.035em;
        color: var(--cv-text);
    }
    h1 { font-weight: 800 !important; }
    p, li, label { color: #c6d3e4; }

    .hero {
        padding: 30px 32px;
        border: 1px solid rgba(91, 208, 201, .22);
        border-radius: 22px;
        background:
            linear-gradient(115deg, rgba(18, 48, 67, .96), rgba(15, 29, 49, .94)),
            #102238;
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
        margin-bottom: 22px;
    }
    .eyebrow {
        color: var(--cv-accent);
        text-transform: uppercase;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .16em;
        margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 800;
        line-height: 1.08;
        color: #f4f8ff;
        margin: 0 0 12px 0;
    }
    .hero-copy {
        color: #b4c5d9;
        max-width: 760px;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0;
    }
    .section-kicker {
        color: var(--cv-accent);
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .14em;
        margin-bottom: 5px;
    }
    .panel {
        background: linear-gradient(145deg, rgba(20,36,58,.96), rgba(14,27,45,.96));
        border: 1px solid var(--cv-border);
        border-radius: 17px;
        padding: 19px 20px;
        min-height: 118px;
    }
    .panel-label {
        color: var(--cv-muted);
        font-size: .78rem;
        font-weight: 600;
        margin-bottom: 9px;
    }
    .panel-value {
        color: #f3f7ff;
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .panel-foot {
        color: #8fa5bf;
        font-size: .75rem;
        margin-top: 8px;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid rgba(57,214,194,.32);
        background: rgba(57,214,194,.09);
        color: #70eadb;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: .75rem;
        font-weight: 700;
    }
    .muted { color: var(--cv-muted); }
    .info-box {
        background: rgba(107,167,255,.08);
        border: 1px solid rgba(107,167,255,.2);
        border-radius: 13px;
        padding: 14px 16px;
        color: #c8d9ef;
    }
    .warning-box {
        background: rgba(245,158,11,.08);
        border: 1px solid rgba(245,158,11,.25);
        border-radius: 13px;
        padding: 14px 16px;
        color: #f6d69a;
    }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(57,214,194,.35);
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        color: #071923;
        background: linear-gradient(100deg, #39d6c2, #70e0cb);
        transition: all .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #8af2e4;
        color: #06131c;
        background: #78eadb;
        transform: translateY(-1px);
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background: #122238;
    }
    div[data-testid="stMetric"] {
        background: rgba(20,36,58,.72);
        border: 1px solid var(--cv-border);
        padding: 15px 17px;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { color: #9fb2c9; }
    div[data-testid="stMetricValue"] { color: #f2f7ff; }
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(107,167,255,.42);
        border-radius: 14px;
        background: rgba(17,34,55,.45);
        padding: 8px;
    }
    hr { border-color: var(--cv-border); }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONFIGURATION & SESSION STATE
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
# Resolve weights robustly: Streamlit Cloud may clone a repository where the
# checkpoint is at the root, under models/, or in another nested directory.
# CRACKVISION_MODEL_PATH can override automatic discovery.
def resolve_model_path() -> tuple[Path | None, list[str]]:
    import os

    candidates: list[Path] = []
    configured = os.getenv("CRACKVISION_MODEL_PATH")
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = BASE_DIR / configured_path
        candidates.append(configured_path)

    candidates.extend([
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "best.pt",
        BASE_DIR / "weights" / "best.pt",
    ])

    # Search recursively as a fallback; ignore virtual environments and git internals.
    try:
        candidates.extend(
            item for item in BASE_DIR.rglob("best.pt")
            if not any(part in {".git", ".venv", "venv", "site-packages"} for part in item.parts)
        )
    except OSError:
        pass

    seen: set[str] = set()
    unique: list[Path] = []
    for item in candidates:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(item)

    existing = [item for item in unique if item.is_file()]
    if existing:
        selected = existing[0]
        # Git LFS pointer files are small text placeholders, not model weights.
        try:
            with selected.open("rb") as handle:
                header = handle.read(200)
            if b"git-lfs.github.com/spec/v1" in header:
                return None, [f"{selected}: Git LFS pointer detected; actual model weights were not downloaded."]
        except OSError as exc:
            return None, [f"Could not read {selected}: {exc}"]
        if selected.stat().st_size < 100_000:
            return None, [f"{selected}: file is only {selected.stat().st_size:,} bytes; expected a real .pt checkpoint, not a pointer/placeholder."]
        return selected, [str(item) for item in existing]

    return None, [str(item) for item in unique[:8]]


MODEL_PATH, MODEL_DIAGNOSTICS = resolve_model_path()
APP_VERSION = "2.1 Premium"

DEFAULT_SETTINGS = {
    "confidence": 0.25,
    "iou": 0.45,
    "image_size": 640,
    "max_detections": 300,
}

if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "analysis_counter" not in st.session_state:
    st.session_state.analysis_counter = 0


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str, modified_time: float):
    """Load and cache the repository's trained YOLO weights."""
    del modified_time  # Cache invalidation when best.pt is replaced.
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics could not be imported. This is usually an OpenCV/cv2 "
            f"dependency problem. Import detail: {YOLO_IMPORT_ERROR}"
        )
    return YOLO(model_path)


def get_model():
    if YOLO is None:
        return None, (
            "Ultralytics import failed before model loading. "
            f"Underlying error: {YOLO_IMPORT_ERROR}"
        )
    if MODEL_PATH is None:
        return None, (
            "A usable best.pt checkpoint was not found. Checked these locations: "
            + "; ".join(MODEL_DIAGNOSTICS)
            + ". Confirm that the actual model file is committed to GitHub (not only a Git LFS pointer)."
        )
    try:
        model = load_yolo_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        return model, None
    except Exception as exc:
        return None, str(exc)


model, model_error = get_model()
model_online = model is not None


# =============================================================================
# HELPERS
# =============================================================================

def render_hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str = "") -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">{label}</div>
            <div class="panel-value">{value}</div>
            <div class="panel-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def safe_class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, f"Class {class_id}"))
        return str(names[class_id])
    except Exception:
        return f"Class {class_id}"


def analyze_image(
    source_image: Image.Image,
    confidence: float,
    iou: float,
    image_size: int,
    max_detections: int,
) -> dict[str, Any]:
    """Run YOLO inference and return display-ready outputs and metrics."""
    predictions = model.predict(
        source=source_image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        max_det=max_detections,
        verbose=False,
    )
    result = predictions[0]

    boxes = result.boxes
    count = len(boxes) if boxes is not None else 0

    if count:
        scores = boxes.conf.detach().cpu().numpy().astype(float)
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        avg_confidence = float(np.mean(scores) * 100)
        max_confidence = float(np.max(scores) * 100)
        min_confidence = float(np.min(scores) * 100)
    else:
        scores = np.array([], dtype=float)
        class_ids = np.array([], dtype=int)
        avg_confidence = max_confidence = min_confidence = 0.0

    class_counts: dict[str, int] = {}
    for class_id in class_ids:
        label = safe_class_name(result.names, int(class_id))
        class_counts[label] = class_counts.get(label, 0) + 1

    plotted = result.plot()  # Ultralytics returns a BGR NumPy array.
    annotated = Image.fromarray(plotted[:, :, ::-1].copy())

    detection_rows = []
    for index, score in enumerate(scores):
        detection_rows.append(
            {
                "Detection": index + 1,
                "Class": safe_class_name(result.names, int(class_ids[index])),
                "Confidence (%)": round(float(score) * 100, 2),
            }
        )

    return {
        "count": count,
        "average_confidence": avg_confidence,
        "max_confidence": max_confidence,
        "min_confidence": min_confidence,
        "class_counts": class_counts,
        "annotated": annotated,
        "detection_table": pd.DataFrame(
            detection_rows,
            columns=["Detection", "Class", "Confidence (%)"],
        ),
    }


def add_history_record(filename: str, metrics: dict[str, Any], fingerprint: str) -> None:
    st.session_state.analysis_counter += 1
    st.session_state.history.insert(
        0,
        {
            "ID": st.session_state.analysis_counter,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": filename,
            "Cracks / detections": metrics["count"],
            "Average confidence (%)": round(metrics["average_confidence"], 2),
            "Maximum confidence (%)": round(metrics["max_confidence"], 2),
            "Image fingerprint": fingerprint[:12],
        },
    )


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 4px 18px 4px;">
            <div style="font-size: 2rem;">🏗️</div>
            <div style="font-family: Manrope; font-size: 1.22rem; font-weight: 800;">
                CrackVision <span style="color:#39d6c2;">AI</span>
            </div>
            <div style="font-size:.76rem;color:#91a4bd;">CONCRETE INTELLIGENCE PLATFORM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "WORKSPACE",
        ["Overview", "AI Inspection", "Inspection History", "Project Information"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    if model_online:
        st.markdown('<span class="status-pill">● MODEL ONLINE</span>', unsafe_allow_html=True)
    else:
        st.error("● MODEL OFFLINE")
        with st.expander("Model diagnostic"):
            st.code(model_error or "Unknown model error")
    st.markdown("---")
    st.caption(f"CrackVision AI · {APP_VERSION}")
    st.caption("AI-assisted visual screening")


# =============================================================================
# OVERVIEW / DASHBOARD
# =============================================================================

if page == "Overview":
    render_hero(
        "STRUCTURAL VISUAL INTELLIGENCE",
        "See the crack. Understand the signal.",
        "An AI-assisted concrete inspection workspace powered by your trained "
        "YOLO segmentation model. Upload an image, review detections, and keep "
        "a clear record of every analysis.",
    )

    total_images = len(st.session_state.history)
    total_detections = sum(
        item["Cracks / detections"] for item in st.session_state.history
    )
    mean_conf = (
        np.mean([x["Average confidence (%)"] for x in st.session_state.history])
        if total_images else 0
    )

    a, b, c, d = st.columns(4)
    with a:
        metric_card("IMAGES ANALYZED", str(total_images), "Current session")
    with b:
        metric_card("DETECTIONS FOUND", str(total_detections), "Across this session")
    with c:
        metric_card("MEAN CONFIDENCE", f"{mean_conf:.1f}%", "Average of image-level scores")
    with d:
        metric_card("AI ENGINE", "YOLO", "Custom trained weights")

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("### Your inspection workspace")
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">01 / INPUT</div>
                <h3 style="margin-top:4px;">Upload concrete imagery</h3>
                <p class="muted">Use JPG, JPEG, or PNG images captured during a visual inspection.</p>
                <hr>
                <div class="section-kicker">02 / INFERENCE</div>
                <h3 style="margin-top:4px;">Run AI crack detection</h3>
                <p class="muted">Adjust confidence and inference settings to review model predictions.</p>
                <hr>
                <div class="section-kicker">03 / REVIEW</div>
                <h3 style="margin-top:4px;">Inspect results and export</h3>
                <p class="muted">Compare the original and annotated image, then download your results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Model readiness")
        if model_online:
            st.success("Your trained model is loaded and ready for inference.")
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-label">ACTIVE MODEL FILE</div>
                    <div style="font-weight:700;word-break:break-all;">{MODEL_PATH.name}</div>
                    <div class="panel-foot">{MODEL_PATH.relative_to(BASE_DIR) if MODEL_PATH.is_relative_to(BASE_DIR) else MODEL_PATH}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("The application is running, but the model is not available.")
            st.markdown(
                """
                <div class="warning-box">
                    Add your trained <b>best.pt</b> file to the
                    <code>models/</code> folder, then restart or rerun the app.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code("Crack-Vision-AI-Concrete-/\n├── app.py\n├── models/\n│   └── best.pt\n└── requirements.txt")
            with st.expander("Model file diagnostics", expanded=True):
                st.write("Repository root:", str(BASE_DIR))
                st.write("Expected model path: `models/best.pt` (also searches root and nested folders).")
                st.write("Search results:")
                for diagnostic in MODEL_DIAGNOSTICS:
                    st.code(diagnostic)
                st.caption("If best.pt is stored with Git LFS, configure Git LFS download for Streamlit Cloud or commit the actual checkpoint file.")

    st.markdown("")
    st.markdown(
        """
        <div class="info-box">
            <b>Important:</b> AI predictions are decision-support information only.
            They do not establish structural safety, crack severity, or repair requirements.
            A qualified professional must assess the structure on site.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# AI INSPECTION
# =============================================================================

elif page == "AI Inspection":
    render_hero(
        "AI-POWERED IMAGE ANALYSIS",
        "Concrete crack inspection",
        "Upload a concrete surface image and run your trained YOLO model. "
        "Review the annotated output, confidence scores, and detection-level details.",
    )

    if not model_online:
        st.error("Inference is unavailable because the trained model could not be loaded.")
        st.info("Verify that `models/best.pt` exists and that the dependencies in requirements.txt are installed.")
        st.stop()

    settings_col, upload_col = st.columns([0.78, 1.5], gap="large")

    with settings_col:
        st.markdown("### Inference controls")
        confidence = st.slider(
            "Confidence threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(DEFAULT_SETTINGS["confidence"]),
            step=0.05,
            help="Higher values reduce low-confidence predictions.",
        )
        iou = st.slider(
            "NMS IoU threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(DEFAULT_SETTINGS["iou"]),
            step=0.05,
            help="Controls overlap suppression for predicted boxes.",
        )
        image_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640, 768],
            value=640,
        )
        # Remove duplicate values while preserving order.
        image_size = int(image_size)
        max_detections = st.number_input(
            "Maximum detections",
            min_value=10,
            max_value=1000,
            value=300,
            step=10,
        )
        st.markdown(
            """
            <div class="info-box">
                <b>Tip</b><br>
                Start with a confidence threshold around 0.25. Tune it using
                validation images and your model's observed performance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with upload_col:
        st.markdown("### Image input")
        uploaded_file = st.file_uploader(
            "Choose a concrete surface image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG.",
        )

        if uploaded_file is not None:
            try:
                raw_bytes = uploaded_file.getvalue()
                fingerprint = hashlib.sha256(raw_bytes).hexdigest()
                source_image = ImageOps.exif_transpose(
                    Image.open(io.BytesIO(raw_bytes))
                ).convert("RGB")

                st.image(
                    source_image,
                    caption=f"Input image · {uploaded_file.name}",
                    use_container_width=True,
                )

                run_analysis = st.button(
                    "🚀 Run crack analysis",
                    type="primary",
                    use_container_width=True,
                )

                if run_analysis:
                    with st.spinner("Running model inference…"):
                        metrics = analyze_image(
                            source_image=source_image,
                            confidence=confidence,
                            iou=iou,
                            image_size=image_size,
                            max_detections=int(max_detections),
                        )

                    st.session_state.latest_result = {
                        "filename": uploaded_file.name,
                        "fingerprint": fingerprint,
                        "source": source_image,
                        "metrics": metrics,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    add_history_record(uploaded_file.name, metrics, fingerprint)
                    st.success("Analysis completed successfully.")

            except Exception as exc:
                st.error("The selected image could not be processed.")
                st.exception(exc)

    latest = st.session_state.latest_result
    if latest is not None:
        st.markdown("---")
        st.markdown("## Inspection results")
        st.caption(
            f"Latest analysis: {latest['filename']} · {latest['timestamp']}"
        )
        metrics = latest["metrics"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detections", metrics["count"])
        m2.metric("Average confidence", f"{metrics['average_confidence']:.2f}%")
        m3.metric("Highest confidence", f"{metrics['max_confidence']:.2f}%")
        m4.metric("Lowest confidence", f"{metrics['min_confidence']:.2f}%")

        original_col, output_col = st.columns(2, gap="large")
        with original_col:
            st.markdown("### Original image")
            st.image(latest["source"], use_container_width=True)
        with output_col:
            st.markdown("### AI annotated image")
            st.image(metrics["annotated"], use_container_width=True)

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "⬇️ Download annotated image",
                data=image_to_bytes(metrics["annotated"]),
                file_name=f"crackvision_annotated_{Path(latest['filename']).stem}.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_col2:
            report_payload = {
                "application": "CrackVision AI",
                "analysis_timestamp": latest["timestamp"],
                "image": latest["filename"],
                "image_fingerprint_sha256": latest["fingerprint"],
                "detection_count": metrics["count"],
                "average_confidence_percent": round(metrics["average_confidence"], 2),
                "maximum_confidence_percent": round(metrics["max_confidence"], 2),
                "minimum_confidence_percent": round(metrics["min_confidence"], 2),
                "class_counts": metrics["class_counts"],
                "disclaimer": "AI-assisted visual screening only; not a structural safety assessment.",
            }
            st.download_button(
                "⬇️ Download JSON report",
                data=json.dumps(report_payload, indent=2).encode("utf-8"),
                file_name=f"crackvision_report_{Path(latest['filename']).stem}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("### Detection details")
        if metrics["detection_table"].empty:
            st.info("No detections passed the selected confidence threshold.")
        else:
            st.dataframe(
                metrics["detection_table"],
                use_container_width=True,
                hide_index=True,
            )
            if metrics["class_counts"]:
                st.caption(
                    "Class distribution: "
                    + " · ".join(
                        f"{name}: {count}"
                        for name, count in metrics["class_counts"].items()
                    )
                )

        st.markdown(
            """
            <div class="warning-box">
                <b>Engineering note:</b> Detection count and model confidence are
                computer-vision outputs. They are not measurements of crack width,
                depth, cause, progression, or structural risk.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# INSPECTION HISTORY
# =============================================================================

elif page == "Inspection History":
    render_hero(
        "SESSION RECORDS",
        "Inspection history",
        "Review the analyses performed during this active application session "
        "and export a structured record for documentation.",
    )

    history = st.session_state.history
    if not history:
        st.info("No inspection records yet. Run an analysis from the AI Inspection page.")
    else:
        history_df = pd.DataFrame(history)
        total = len(history_df)
        detections = int(history_df["Cracks / detections"].sum())
        h1, h2, h3 = st.columns(3)
        h1.metric("Inspection records", total)
        h2.metric("Total detections", detections)
        h3.metric(
            "Average image confidence",
            f"{history_df['Average confidence (%)'].mean():.2f}%",
        )

        st.markdown("### Recent analyses")
        st.dataframe(
            history_df.drop(columns=["Image fingerprint"]),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = history_df.to_csv(index=False).encode("utf-8-sig")
        json_bytes = history_df.to_json(
            orient="records", indent=2, force_ascii=False
        ).encode("utf-8")

        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "⬇️ Export history as CSV",
                data=csv_bytes,
                file_name="crackvision_inspection_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            st.download_button(
                "⬇️ Export history as JSON",
                data=json_bytes,
                file_name="crackvision_inspection_history.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")
        confirm_clear = st.checkbox(
            "I understand that clearing history removes all records from this session."
        )
        if st.button(
            "Clear session history",
            disabled=not confirm_clear,
            use_container_width=False,
        ):
            st.session_state.history = []
            st.session_state.latest_result = None
            st.session_state.analysis_counter = 0
            st.rerun()

    st.caption(
        "History is stored in Streamlit session state. It is not a permanent database "
        "and may reset when the session or application restarts."
    )


# =============================================================================
# PROJECT INFORMATION
# =============================================================================

elif page == "Project Information":
    render_hero(
        "ABOUT THE PLATFORM",
        "Built for clearer concrete inspection workflows.",
        "CrackVision AI combines computer vision and a Streamlit interface to "
        "make trained-model predictions easier to review and document.",
    )

    info1, info2 = st.columns(2, gap="large")
    with info1:
        st.markdown("### Technology stack")
        st.markdown(
            """
            - **Python** — application logic
            - **Streamlit** — interactive web application
            - **Ultralytics YOLO** — model inference
            - **PyTorch** — deep-learning runtime
            - **Pillow** — image handling
            - **NumPy & Pandas** — metrics and reporting
            """
        )
    with info2:
        st.markdown("### Application workflow")
        st.markdown(
            """
            1. Prepare and annotate a concrete-crack dataset.
            2. Train and validate a YOLO segmentation/detection model.
            3. Place trained weights at `models/best.pt`.
            4. Upload a concrete surface image.
            5. Run inference and review annotated predictions.
            6. Export the annotated image and analysis report.
            """
        )

    st.markdown("### Expected project structure")
    st.code(
        "Crack-Vision-AI-Concrete-/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── models/\n"
        "    └── best.pt",
        language="text",
    )

    st.markdown("### Deployment")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run app.py",
        language="bash",
    )

    st.markdown(
        """
        <div class="warning-box">
            <b>Responsible use and limitations</b><br>
            CrackVision AI is a research and decision-support interface. Model
            performance depends on training data, image quality, lighting,
            camera angle, and the conditions represented in the dataset. A
            prediction must not be treated as a structural diagnosis or safety
            certification. Consult a qualified civil/structural engineer for
            professional inspection and decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
                color:#71869f;font-size:.76rem;padding:4px 0 14px 0;">
        <span>🏗️ <b style="color:#a9bfd5;">CrackVision AI</b> · Concrete Intelligence</span>
        <span>AI-assisted visual screening · Not an engineering certification</span>
    </div>
    """,
    unsafe_allow_html=True,
)
