import streamlit as st
import numpy as np
import pandas as pd
import cv2
import io

from PIL import Image
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CrackVision AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: #ffffff;
}

header {
    background-color: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
}

.metric-card {
    background-color: #1e293b;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: #38bdf8;
    font-size: 30px;
    font-weight: bold;
}

.section-title {
    color: #38bdf8;
    font-size: 23px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0369a1;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DEEP LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_paths = [
        Path("models/best.pt"),
        Path("results/crack_detection/weights/best.pt")
    ]

    for path in model_paths:

        if path.exists():
            return YOLO(str(path))

    return None


model = load_model()


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#38bdf8;'>🏗️ CrackVision AI</h2>",
        unsafe_allow_html=True
    )

    st.write("AI Concrete Inspection Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Crack Detection",
            "Analysis History",
            "About Project"
        ]
    )

    st.divider()

    st.write("### Model Status")

    if model is not None:
        st.success("Deep Learning Model Loaded")
    else:
        st.error("Model Not Found")

    st.caption("YOLOv8 Segmentation Architecture")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">CrackVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Concrete Crack Detection '
        'and Segmentation System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<div class="section-title">Project Dashboard</div>',
        unsafe_allow_html=True
    )

    total = len(st.session_state.history)

    detected = sum(
        item["Crack Detected"] == "Yes"
        for item in st.session_state.history
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Images Analyzed",
            total
        )

    with col2:

        st.metric(
            "Cracks Detected",
            detected
        )

    with col3:

        st.metric(
            "Model Architecture",
            "YOLOv8"
        )

    st.divider()

    st.subheader("System Overview")

    st.write("""
    CrackVision AI is a Deep Learning application
    designed to identify and segment cracks in
    concrete structures.

    The application uses Computer Vision and
    YOLOv8 segmentation to analyze uploaded
    concrete images.

    Upload an image to begin AI-powered inspection.
    """)

    if st.button("Start Crack Detection"):

        st.session_state.analysis = None
        st.info("Select Crack Detection from the sidebar.")


# ==========================================
# CRACK DETECTION
# ==========================================

elif page == "Crack Detection":

    st.markdown(
        '<div class="main-title">AI Crack Detection</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a concrete image for Deep Learning analysis."
    )

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:

        uploaded_file = st.file_uploader(
            "Upload Concrete Image",
            type=["jpg", "jpeg", "png"]
        )

    with col2:

        confidence = st.slider(
            "Detection Confidence",
            0.10,
            0.90,
            0.25,
            0.05
        )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🚀 Analyze Image"):

            if model is None:

                st.error(
                    "Model not found. "
                    "Place best.pt inside the models folder."
                )

            else:

                with st.spinner(
                    "AI is analyzing the concrete image..."
                ):

                    results = model.predict(
                        image_array,
                        conf=confidence,
                        verbose=False
                    )

                    result = results[0]

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    boxes = result.boxes

                    crack_detected = (
                        boxes is not None and len(boxes) > 0
                    )

                    scores = []

                    if crack_detected:

                        scores = (
                            boxes.conf.cpu().numpy()
                        )

                        confidence_score = (
                            float(np.max(scores)) * 100
                        )

                    else:

                        confidence_score = 0.0

                    # Segmentation mask

                    crack_coverage = 0.0

                    if result.masks is not None:

                        masks = (
                            result.masks.data.cpu().numpy()
                        )

                        combined_mask = np.zeros(
                            image_array.shape[:2],
                            dtype=np.uint8
                        )

                        for mask in masks:

                            resized_mask = cv2.resize(
                                mask,
                                (
                                    image_array.shape[1],
                                    image_array.shape[0]
                                )
                            )

                            combined_mask = np.maximum(
                                combined_mask,
                                (resized_mask > 0.5).astype(
                                    np.uint8
                                )
                            )

                        crack_pixels = np.count_nonzero(
                            combined_mask
                        )

                        total_pixels = combined_mask.size

                        crack_coverage = (
                            crack_pixels / total_pixels
                        ) * 100

                    # Save analysis

                    analysis = {
                        "image": image.copy(),
                        "annotated": annotated,
                        "detected": crack_detected,
                        "confidence": confidence_score,
                        "coverage": crack_coverage,
                        "time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )
                    }

                    st.session_state.analysis = analysis

                    st.session_state.history.append({
                        "Date": analysis["time"],
                        "Crack Detected": (
                            "Yes" if crack_detected else "No"
                        ),
                        "Confidence": round(
                            confidence_score, 2
                        ),
                        "Coverage": round(
                            crack_coverage, 2
                        )
                    })

    # Display results

    analysis = st.session_state.analysis

    if analysis is not None:

        st.divider()

        st.subheader("AI Detection Results")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                analysis["image"],
                caption="Original Image",
                use_container_width=True
            )

        with col2:

            st.image(
                analysis["annotated"],
                caption="AI Segmentation Result",
                use_container_width=True
            )

        st.divider()

        st.subheader("Analysis Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Crack Detection",
                "YES" if analysis["detected"] else "NO"
            )

        with col2:

            st.metric(
                "Maximum Confidence",
                f"{analysis['confidence']:.2f}%"
            )

        with col3:

            st.metric(
                "Crack Coverage",
                f"{analysis['coverage']:.2f}%"
            )

        if analysis["detected"]:

            st.warning(
                "Potential concrete cracks detected. "
                "Professional inspection is recommended."
            )

        else:

            st.success(
                "No cracks detected by the model."
            )

        st.caption(
            "Image-based AI analysis only; this result "
            "does not establish structural safety."
        )

        report = pd.DataFrame([{
            "Date": analysis["time"],
            "Crack Detected": analysis["detected"],
            "Maximum Confidence": analysis["confidence"],
            "Crack Coverage": analysis["coverage"]
        }])

        st.download_button(
            "📥 Download Analysis CSV",
            report.to_csv(index=False),
            file_name="crackvision_report.csv",
            mime="text/csv"
        )


# ==========================================
# ANALYSIS HISTORY
# ==========================================

elif page == "Analysis History":

    st.markdown(
        '<div class="main-title">Analysis History</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "Download History",
            history_df.to_csv(index=False),
            file_name="crackvision_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.session_state.analysis = None

            st.rerun()

    else:

        st.info("No image analysis available yet.")


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About CrackVision AI</div>',
        unsafe_allow_html=True
    )

    st.write("""
    CrackVision AI is an AI-powered concrete
    crack detection and segmentation application.

    ### Deep Learning Technologies

    - Convolutional Neural Networks
    - YOLOv8 Segmentation
    - Transfer Learning
    - Computer Vision
    - Image Processing

    ### Development Technologies

    - Python
    - PyTorch
    - OpenCV
    - NumPy
    - Streamlit

    ### Application Features

    - Concrete image analysis
    - Crack detection
    - Pixel-level segmentation
    - Confidence visualization
    - Crack coverage analysis
    - Downloadable reports

    ### Project Objective

    To demonstrate how Deep Learning and
    Computer Vision can assist in identifying
    visible cracks in concrete surfaces.
    """)

    st.info(
        "Developed as a Deep Learning and "
        "Computer Vision portfolio project."
    )
