import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Traffic AI — Indian Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --navy-900: #050b1c;
        --navy-800: #08122b;
        --navy-700: #0d1b3a;
        --cyan: #22d3ee;
        --blue: #3b82f6;
        --text: #e8f1ff;
        --muted: #93a4c4;
        --glass: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(120, 190, 255, 0.18);
    }

    .stApp {
        background:
            radial-gradient(1100px 600px at 12% -8%, rgba(59, 130, 246, 0.22), transparent 60%),
            radial-gradient(900px 520px at 92% 4%, rgba(34, 211, 238, 0.16), transparent 60%),
            linear-gradient(180deg, var(--navy-900) 0%, var(--navy-800) 45%, #060d20 100%);
        color: var(--text);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    h1, h2, h3, h4, p, span, label, li {color: var(--text);}

    /* ---------- animations ---------- */
    @keyframes fadeIn {from {opacity: 0;} to {opacity: 1;}}
    @keyframes slideUp {
        from {opacity: 0; transform: translateY(26px);}
        to {opacity: 1; transform: translateY(0);}
    }
    @keyframes floaty {
        0%, 100% {transform: translateY(0);}
        50% {transform: translateY(-12px);}
    }
    @keyframes glowPulse {
        0%, 100% {box-shadow: 0 0 22px rgba(34, 211, 238, 0.18);}
        50% {box-shadow: 0 0 42px rgba(34, 211, 238, 0.38);}
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    @keyframes growBar {from {width: 0%;}}
    @keyframes spin {to {transform: rotate(360deg);}}

    .fade-in {animation: fadeIn 0.9s ease both;}
    .slide-up {animation: slideUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;}
    .d1 {animation-delay: 0.1s;}
    .d2 {animation-delay: 0.22s;}
    .d3 {animation-delay: 0.34s;}
    .d4 {animation-delay: 0.46s;}

    /* ---------- navigation ---------- */
    .nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding: 14px 22px;
        margin-bottom: 26px;
        border-radius: 18px;
        background: rgba(9, 18, 40, 0.72);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(14px);
    }
    .nav-logo {
        font-weight: 800;
        font-size: 1.12rem;
        letter-spacing: 0.4px;
        background: linear-gradient(90deg, #22d3ee, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .nav-links {display: flex; gap: 8px; flex-wrap: wrap;}
    .nav-links a {
        color: var(--muted);
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 8px 14px;
        border-radius: 10px;
        transition: all 0.25s ease;
    }
    .nav-links a:hover {
        color: #fff;
        background: rgba(34, 211, 238, 0.14);
        transform: translateY(-2px);
    }

    /* ---------- hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        text-align: center;
        padding: 58px 26px 52px;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(23, 42, 82, 0.6), rgba(8, 18, 43, 0.6));
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(16px);
    }
    .hero-badge {
        display: inline-block;
        font-size: 0.78rem;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--cyan);
        border: 1px solid rgba(34, 211, 238, 0.35);
        background: rgba(34, 211, 238, 0.08);
        padding: 6px 16px;
        border-radius: 999px;
        margin-bottom: 20px;
    }
    .hero h1 {
        font-size: clamp(2rem, 5.2vw, 3.5rem);
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 14px;
        background: linear-gradient(90deg, #ffffff, #22d3ee, #6366f1, #ffffff);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 9s ease infinite;
    }
    .hero p {
        color: var(--muted);
        font-size: clamp(0.98rem, 1.7vw, 1.18rem);
        max-width: 640px;
        margin: 0 auto;
    }
    .hero-visual {
        font-size: clamp(3rem, 8vw, 4.6rem);
        margin-bottom: 10px;
        animation: floaty 4.5s ease-in-out infinite;
        filter: drop-shadow(0 12px 26px rgba(34, 211, 238, 0.35));
    }
    .hero-orb {
        position: absolute;
        border-radius: 50%;
        filter: blur(60px);
        opacity: 0.5;
        animation: floaty 8s ease-in-out infinite;
    }
    .orb-a {width: 220px; height: 220px; background: #1d4ed8; top: -70px; left: -60px;}
    .orb-b {width: 200px; height: 200px; background: #0e7490; bottom: -80px; right: -50px; animation-delay: 1.6s;}
    .hero-stats {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 14px;
        margin-top: 30px;
    }
    .hero-chip {
        padding: 10px 18px;
        border-radius: 14px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        font-size: 0.86rem;
        color: var(--muted);
        transition: all 0.25s ease;
    }
    .hero-chip:hover {
        transform: translateY(-4px);
        color: #fff;
        border-color: rgba(34, 211, 238, 0.5);
    }
    .hero-chip b {color: var(--cyan); font-weight: 700;}

    /* ---------- section headings ---------- */
    .section-title {
        text-align: center;
        font-size: clamp(1.35rem, 3vw, 1.9rem);
        font-weight: 700;
        margin: 52px 0 6px;
    }
    .section-sub {
        text-align: center;
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 26px;
    }

    /* ---------- glass card ---------- */
    .glass {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        backdrop-filter: blur(14px);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass:hover {
        transform: translateY(-6px);
        border-color: rgba(34, 211, 238, 0.45);
        box-shadow: 0 18px 44px rgba(3, 10, 28, 0.6);
    }

    /* ---------- upload ---------- */
    .upload-box {
        text-align: center;
        padding: 34px 22px 26px;
        border-radius: 22px;
        border: 2px dashed rgba(34, 211, 238, 0.42);
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.07), rgba(99, 102, 241, 0.07));
        animation: glowPulse 4s ease-in-out infinite;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: var(--cyan);
        transform: translateY(-4px);
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.14), rgba(99, 102, 241, 0.12));
    }
    .upload-icon {
        font-size: 2.8rem;
        animation: floaty 3.6s ease-in-out infinite;
    }
    .upload-box h3 {margin: 10px 0 4px; font-size: 1.25rem;}
    .upload-box p {color: var(--muted); margin: 0; font-size: 0.93rem;}
    .formats {
        display: inline-block;
        margin-top: 12px;
        font-size: 0.78rem;
        letter-spacing: 1px;
        color: var(--cyan);
        border: 1px solid rgba(34, 211, 238, 0.32);
        border-radius: 999px;
        padding: 5px 14px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(9, 18, 40, 0.7);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(34, 211, 238, 0.55);
        box-shadow: 0 0 26px rgba(34, 211, 238, 0.16);
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] div {color: var(--text) !important;}
    [data-testid="stFileUploader"] button {
        background: linear-gradient(90deg, #22d3ee, #3b82f6) !important;
        color: #041024 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 10px 24px rgba(34, 211, 238, 0.4);
    }

    /* ---------- image preview ---------- */
    [data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 16px 40px rgba(2, 8, 24, 0.65);
        animation: slideUp 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    /* ---------- result card ---------- */
    .result-card {
        border-radius: 24px;
        padding: 30px;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.12), rgba(99, 102, 241, 0.12));
        border: 1px solid rgba(34, 211, 238, 0.32);
        backdrop-filter: blur(14px);
    }
    .result-label {
        font-size: 0.76rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--muted);
    }
    .result-value {
        font-size: clamp(1.5rem, 3.4vw, 2.15rem);
        font-weight: 800;
        line-height: 1.2;
        margin: 4px 0 18px;
        background: linear-gradient(90deg, #ffffff, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .confidence-value {
        font-size: clamp(1.6rem, 3.6vw, 2.3rem);
        font-weight: 800;
        color: var(--cyan);
    }

    .bar-track {
        width: 100%;
        height: 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.09);
        overflow: hidden;
        margin-top: 10px;
    }
    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #22d3ee, #6366f1);
        animation: growBar 1.4s cubic-bezier(0.22, 1, 0.36, 1) both;
        box-shadow: 0 0 16px rgba(34, 211, 238, 0.55);
    }

    /* ---------- rank cards ---------- */
    .rank-card {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-radius: 16px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        transition: all 0.28s ease;
    }
    .rank-card:hover {
        transform: translateX(6px);
        border-color: rgba(34, 211, 238, 0.5);
        background: rgba(34, 211, 238, 0.08);
    }
    .rank-medal {font-size: 1.5rem; min-width: 38px; text-align: center;}
    .rank-body {flex: 1; min-width: 0;}
    .rank-name {
        font-weight: 600;
        font-size: 0.98rem;
        margin-bottom: 7px;
        word-break: break-word;
    }
    .rank-pct {
        font-weight: 700;
        color: var(--cyan);
        min-width: 78px;
        text-align: right;
        font-size: 1rem;
    }

    /* ---------- info cards ---------- */
    .info-card {
        text-align: center;
        padding: 22px 14px;
        border-radius: 18px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        transition: all 0.3s ease;
        height: 100%;
    }
    .info-card:hover {
        transform: translateY(-7px);
        border-color: rgba(34, 211, 238, 0.5);
        box-shadow: 0 16px 38px rgba(2, 8, 24, 0.55);
    }
    .info-icon {font-size: 1.7rem;}
    .info-label {
        font-size: 0.74rem;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 8px;
    }
    .info-value {font-size: 1.02rem; font-weight: 700; margin-top: 4px;}

    /* ---------- steps ---------- */
    .step-card {
        text-align: center;
        padding: 24px 16px;
        border-radius: 18px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        transition: all 0.3s ease;
        height: 100%;
    }
    .step-card:hover {
        transform: translateY(-7px);
        border-color: rgba(34, 211, 238, 0.5);
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        font-size: 0.85rem;
        font-weight: 700;
        color: #041024;
        background: linear-gradient(90deg, #22d3ee, #3b82f6);
        margin-bottom: 10px;
    }
    .step-icon {font-size: 2rem; animation: floaty 5s ease-in-out infinite;}
    .step-title {font-weight: 700; margin-top: 8px;}
    .step-desc {color: var(--muted); font-size: 0.86rem; margin-top: 4px;}
    .step-arrow {
        text-align: center;
        color: var(--cyan);
        font-size: 1.4rem;
        animation: floaty 3s ease-in-out infinite;
    }

    /* ---------- analyzing ---------- */
    .analyzing {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 22px;
        border-radius: 18px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        font-weight: 600;
    }
    .spinner {
        width: 26px;
        height: 26px;
        border-radius: 50%;
        border: 3px solid rgba(34, 211, 238, 0.2);
        border-top-color: var(--cyan);
        animation: spin 0.9s linear infinite;
    }

    /* ---------- about + footer ---------- */
    .about {
        padding: 30px;
        border-radius: 22px;
        background: var(--glass);
        border: 1px solid var(--glass-border);
        line-height: 1.75;
        color: var(--muted);
    }
    .about b {color: var(--text);}

    .footer {
        margin-top: 56px;
        padding: 30px 20px;
        text-align: center;
        border-top: 1px solid var(--glass-border);
        color: var(--muted);
    }
    .footer-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 6px;
    }
    .footer-dev {
        margin-top: 10px;
        color: var(--cyan);
        font-weight: 600;
    }

    /* ---------- streamlit widget theming ---------- */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {color: var(--text) !important;}
    .stDataFrame {border-radius: 14px; overflow: hidden;}

    /* ---------- responsive ---------- */
    @media (max-width: 768px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .nav {justify-content: center; text-align: center;}
        .hero {padding: 40px 16px;}
        .glass, .result-card, .about {padding: 18px;}
        .rank-card {flex-wrap: wrap; gap: 10px;}
        .rank-pct {text-align: left;}
        .step-arrow {transform: rotate(90deg);}
    }
    </style>
    """,
    unsafe_allow_html=True
)


def progress_bar(pct: float, delay: float = 0.0) -> str:
    """Return an animated gradient progress bar for the given percentage."""
    return (
        f'<div class="bar-track"><div class="bar-fill" '
        f'style="width:{max(pct, 0.6):.2f}%;animation-delay:{delay}s"></div></div>'
    )


# -------------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------------

MODEL_URL = "https://huggingface.co/ChigurlaAparna/vgg16-traffic-sign-classification/resolve/main/vgg16_best.keras"
MODEL_PATH = "vgg16_best.keras"

# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading VGG16 Model from Hugging Face..."):

            response = requests.get(MODEL_URL)

            if response.status_code != 200:
                st.error("❌ Failed to download model from Hugging Face.")
                st.stop()

            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()

# -------------------------------------------------------
# CLASS NAMES
# -------------------------------------------------------

df = pd.read_csv("class_names.csv")

class_names = dict(zip(df["ClassId"], df["Name"]))

# -------------------------------------------------------
# HEADER / NAVIGATION
# -------------------------------------------------------

st.markdown(
    """
    <div class="nav fade-in" id="home">
        <div class="nav-logo">🚦 Traffic AI</div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#model">Model</a>
            <a href="#about">About</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# HERO
# -------------------------------------------------------

st.markdown(
    """
    <div class="hero slide-up">
        <div class="hero-orb orb-a"></div>
        <div class="hero-orb orb-b"></div>
        <div class="hero-visual">🚦</div>
        <div class="hero-badge">Deep Learning · Computer Vision</div>
        <h1>AI-Powered Traffic Sign Recognition</h1>
        <p>Identify Indian traffic signs instantly using Deep Learning.</p>
        <div class="hero-stats">
            <div class="hero-chip"><b>VGG16</b> Transfer Learning</div>
            <div class="hero-chip"><b>59</b> Sign Classes</div>
            <div class="hero-chip"><b>224 × 224</b> Input</div>
            <div class="hero-chip"><b>Top-5</b> Predictions</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# UPLOAD
# -------------------------------------------------------

st.markdown('<div class="section-title slide-up d1">Upload Traffic Sign Image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub slide-up d1">Let the model analyze your traffic sign in seconds.</div>',
    unsafe_allow_html=True
)

upload_left, upload_mid, upload_right = st.columns([1, 3, 1])

with upload_mid:
    st.markdown(
        """
        <div class="upload-box slide-up d2">
            <div class="upload-icon">📤</div>
            <h3>Upload Traffic Sign Image</h3>
            <p>Drag &amp; drop your image here</p>
            <div class="formats">SUPPORTED FORMATS · JPG · JPEG · PNG</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a Traffic Sign Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# -------------------------------------------------------
# PREDICTION PIPELINE
# -------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # ---------------------------------------------------
    # IMAGE PREVIEW
    # ---------------------------------------------------

    st.markdown('<div class="section-title slide-up">Image Preview</div>', unsafe_allow_html=True)

    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
        st.markdown(
            f"""
            <div class="glass slide-up" style="margin-top:14px;text-align:center;">
                <div class="result-label">Image Details</div>
                <div style="margin-top:8px;font-weight:600;">
                    {image.width} × {image.height} px &nbsp;·&nbsp; {image.mode} &nbsp;·&nbsp; {uploaded_file.name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------
    # PREDICTION (ML logic unchanged)
    # ---------------------------------------------------

    with result_col:
        status = st.empty()
        status.markdown(
            """
            <div class="analyzing fade-in">
                <div class="spinner"></div>
                <div>AI is analyzing the traffic sign...</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Preprocess Image
        img = image.resize((224, 224))
        img = np.array(img)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Prediction
        prediction = model.predict(img, verbose=0)

        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0])) * 100

        time.sleep(0.4)
        status.empty()

        # -----------------------------------------------
        # RESULT
        # -----------------------------------------------

        st.markdown(
            f"""
            <div class="result-card slide-up">
                <div class="result-label">Traffic Sign</div>
                <div class="result-value">{class_names[predicted_class]}</div>
                <div class="result-label">Confidence</div>
                <div class="confidence-value">{confidence:.2f}%</div>
                {progress_bar(confidence, 0.2)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------
    # TOP 5 PREDICTIONS
    # ---------------------------------------------------

    st.markdown('<div class="section-title">Top 5 Predictions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ranked class probabilities produced by the model.</div>',
        unsafe_allow_html=True
    )

    top5 = np.argsort(prediction[0])[::-1][:5]

    result = pd.DataFrame({
        "Traffic Sign": [class_names[i] for i in top5],
        "Confidence (%)": [round(prediction[0][i] * 100, 2) for i in top5]
    })

    medals = ["🥇", "🥈", "🥉", "4.", "5."]

    rank_cards = ""
    for rank, (name, pct) in enumerate(
        zip(result["Traffic Sign"], result["Confidence (%)"])
    ):
        rank_cards += f"""
        <div class="rank-card slide-up" style="animation-delay:{0.08 * rank:.2f}s">
            <div class="rank-medal">{medals[rank]}</div>
            <div class="rank-body">
                <div class="rank-name">{name}</div>
                {progress_bar(float(pct), 0.2 + 0.08 * rank)}
            </div>
            <div class="rank-pct">{pct:.2f}%</div>
        </div>
        """

    st.markdown(rank_cards, unsafe_allow_html=True)

    # ---------------------------------------------------
    # CONFIDENCE CHART
    # ---------------------------------------------------

    st.markdown('<div class="section-title">Prediction Confidence</div>', unsafe_allow_html=True)

    chart = result.set_index("Traffic Sign")

    st.bar_chart(chart, color="#22d3ee")

    with st.expander("View detailed prediction table"):
        st.dataframe(
            result,
            use_container_width=True
        )

# -------------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------------

st.markdown('<div class="section-title" id="model">Model Information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Architecture and deployment details behind the classifier.</div>',
    unsafe_allow_html=True
)

model_info = [
    ("🧠", "Model", "VGG16 Transfer Learning"),
    ("⚙️", "Framework", "TensorFlow / Keras"),
    ("🖼️", "Input", "224 × 224"),
    ("🏷️", "Classes", "59"),
    ("🚀", "Deployment", "Streamlit + Hugging Face"),
]

info_cols = st.columns(len(model_info))

for col, (icon, label, value) in zip(info_cols, model_info):
    with col:
        st.markdown(
            f"""
            <div class="info-card slide-up">
                <div class="info-icon">{icon}</div>
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------------------------------
# HOW IT WORKS
# -------------------------------------------------------

st.markdown('<div class="section-title" id="how-it-works">How It Works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">A four-step pipeline from raw image to recognised traffic sign.</div>',
    unsafe_allow_html=True
)

steps = [
    ("📤", "Upload Image", "JPG, JPEG or PNG traffic sign photo."),
    ("🧪", "Image Preprocessing", "RGB conversion, resize to 224 × 224, VGG16 preprocessing."),
    ("🧠", "VGG16 Prediction", "Fine-tuned VGG16 computes class probabilities."),
    ("🚦", "Traffic Sign Result", "Predicted sign, confidence and top-5 ranking."),
]

step_cols = st.columns([3, 1, 3, 1, 3, 1, 3])

for index, (icon, title, desc) in enumerate(steps):
    with step_cols[index * 2]:
        st.markdown(
            f"""
            <div class="step-card slide-up" style="animation-delay:{0.1 * index:.2f}s">
                <div class="step-num">{index + 1}</div>
                <div class="step-icon">{icon}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    if index < len(steps) - 1:
        with step_cols[index * 2 + 1]:
            st.markdown('<div class="step-arrow">➜</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# ABOUT
# -------------------------------------------------------

st.markdown('<div class="section-title" id="about">About the Project</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="about slide-up">
        This application classifies Indian traffic signs using a fine-tuned
        <b>VGG16 deep learning model</b>. The network was trained with transfer learning on a
        dataset of Indian road signage and is hosted on <b>Hugging Face</b>, then served through a
        <b>Streamlit</b> interface.
        <br><br>
        Every uploaded image is converted to RGB, resized to <b>224 × 224</b> and passed through the
        official VGG16 preprocessing pipeline before inference. The app reports the predicted sign,
        its confidence score, the <b>top-5 most likely classes</b> and a confidence chart, so the
        model's reasoning stays transparent.
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <div class="footer-title">🚦 Indian Traffic Sign Classification</div>
        <div>Built with Python, TensorFlow, VGG16 and Streamlit</div>
        <div class="footer-dev">Developed by Aparna Chigurla</div>
    </div>
    """,
    unsafe_allow_html=True
)
