import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from ultralytics import YOLO

from pathlib import Path
from datetime import datetime


# ==================================
# PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="CrackVision AI",
    page_icon="🏗️",
    layout="wide"
)


# ==================================
# CUSTOM CSS
# ==================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #f8fafc;
}

.metric-card {
    background: #172033;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #263449;
}

.status {
    color: #22c55e;
    font-weight: bold;
}

.stButton > button {
    background-color: #0d9488;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #0f766e;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==================================
# MODEL LOADING
# ==================================

MODEL_PATH = Path("models/best.pt")


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    return YOLO(str(MODEL_PATH))


try:

    model = load_model()

    model_status = model is not None

except Exception as e:

    model = None

    model_status = False

    model_error = str(e)


# ==================================
# SESSION STATE
# ==================================

if "history" not in st.session_state:

    st.session_state.history = []


# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("🏗️ CrackVision AI")

st.sidebar.caption(
    "AI Concrete Inspection Platform"
)

page = st.sidebar.radio(

    "NAVIGATION",

    [
        "Dashboard",
        "Crack Detection",
        "Analysis History",
        "About Project"
    ]

)

st.sidebar.divider()

if model_status:

    st.sidebar.success("AI Model Online")

else:

    st.sidebar.error("AI Model Offline")


# ==================================
# DASHBOARD
# ==================================

if page == "Dashboard":

    st.title("🏗️ CrackVision AI")

    st.write(
        "Intelligent Concrete Crack Detection "
        "and Segmentation System"
    )

    total = len(st.session_state.history)

    cracks = sum(
        x["Cracks Detected"]
        for x in st.session_state.history
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Images Analyzed",
        total
    )

    c2.metric(
        "Cracks Detected",
        cracks
    )

    c3.metric(
        "Model Architecture",
        "YOLOv8"
    )

    st.divider()

    st.subheader("System Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### AI Inspection

            Upload concrete images and analyze
            structural cracks using deep learning.

            - Image Upload
            - Crack Detection
            - Segmentation Masks
            - Confidence Scores
            """
        )

    with col2:

        st.markdown(
            """
            ### Model Information

            **Architecture:** YOLOv8n-Seg

            **Task:** Instance Segmentation

            **Class:** Concrete Crack

            **Framework:** PyTorch

            **Frontend:** Streamlit
            """
        )

    if model_status:

        st.success("YOLOv8 model loaded successfully!")

    else:

        st.warning(
            "Model not available. Add the trained "
            "models/best.pt file."
        )


# ==================================
# CRACK DETECTION
# ==================================

elif page == "Crack Detection":

    st.title("🔍 Concrete Crack Detection")

    st.write(
        "Upload an image to perform AI-powered "
        "concrete crack inspection."
    )

    uploaded_file = st.file_uploader(

        "Upload Concrete Image",

        type=["jpg", "jpeg", "png"]

    )

    confidence = st.slider(

        "Detection Confidence",

        0.10,

        0.90,

        0.25

    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🚀 Analyze Concrete Image"):

            if model is None:

                st.error(
                    "AI model is not loaded. "
                    "Check models/best.pt."
                )

            else:

                with st.spinner(
                    "AI is analyzing your image..."
                ):

                    results = model.predict(

                        source=image,

                        conf=confidence,

                        imgsz=640,

                        verbose=False

                    )

                    result = results[0]

                    # Segmentation count
                    crack_count = (
                        len(result.masks.data)
                        if result.masks is not None
                        else 0
                    )

                    # Confidence
                    if (
                        result.boxes is not None
                        and len(result.boxes) > 0
                    ):

                        avg_conf = float(
                            result.boxes.conf.mean()
                        ) * 100

                    else:

                        avg_conf = 0.0

                    # Prediction image
                    annotated = result.plot()

                    annotated = annotated[:, :, ::-1]

                    st.success(
                        "Image analysis completed!"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.subheader("Original Image")

                        st.image(image)

                    with col2:

                        st.subheader("AI Prediction")

                        st.image(annotated)

                    st.divider()

                    c1, c2 = st.columns(2)

                    c1.metric(
                        "Cracks Detected",
                        crack_count
                    )

                    c2.metric(
                        "Average Confidence",
                        f"{avg_conf:.2f}%"
                    )

                    st.session_state.history.append({

                        "Date": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                        "Image": uploaded_file.name,

                        "Cracks Detected": crack_count,

                        "Confidence": round(avg_conf, 2)

                    })


# ==================================
# ANALYSIS HISTORY
# ==================================

elif page == "Analysis History":

    st.title("📊 Analysis History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(

            "Download Analysis Report",

            data=csv,

            file_name="crack_analysis.csv",

            mime="text/csv"

        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No inspection records available yet."
        )


# ==================================
# ABOUT PROJECT
# ==================================

elif page == "About Project":

    st.title("ℹ️ About CrackVision AI")

    st.markdown(
        """
        ## Intelligent Concrete Inspection Platform

        CrackVision AI is a deep learning application
        designed to identify and segment cracks in
        concrete structures.

        ### Technologies Used

        - Python
        - YOLOv8 Segmentation
        - PyTorch
        - OpenCV
        - Streamlit
        - Computer Vision

        ### Project Workflow

        1. Dataset Preparation
        2. Image Annotation
        3. YOLOv8 Model Training
        4. Model Evaluation
        5. Image Prediction
        6. Crack Segmentation
        7. Web Application Deployment

        ### Objective

        To support automated concrete crack inspection
        through computer vision and deep learning.
        """
    )
