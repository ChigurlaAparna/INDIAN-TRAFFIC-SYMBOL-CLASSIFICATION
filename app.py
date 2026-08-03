import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Indian Traffic Sign Classification",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "vgg16_best.keras",
        compile=False
    )
    return model

model = load_model()

# -------------------------------------------------------
# LOAD CLASS NAMES
# -------------------------------------------------------

df = pd.read_csv("class_names.csv")

class_names = dict(zip(df["ClassId"], df["Name"]))

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------

st.title("🚦 Indian Traffic Sign Classification")
st.markdown("### Deep Learning using VGG16")

st.write("---")

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.header("Model Information")

st.sidebar.write("**Model:** VGG16")

st.sidebar.write("**Number of Classes:** 58")

st.sidebar.write("**Input Size:** 224 × 224")

st.sidebar.write("**Framework:** TensorFlow / Keras")

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    img = image.resize((224,224))

    img = np.array(img)

    img = img.astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))*100

    with col2:

        st.success("Prediction Completed")

        st.markdown(
            f"## 🚦 {class_names[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(confidence/100)

    st.write("---")

    st.subheader("Top 5 Predictions")

    top5 = np.argsort(prediction[0])[::-1][:5]

    result = pd.DataFrame({
        "Traffic Sign":[class_names[i] for i in top5],
        "Confidence (%)":[round(prediction[0][i]*100,2) for i in top5]
    })

    st.dataframe(
        result,
        use_container_width=True
    )

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------

st.write("---")

st.markdown(
"""
### About

This application classifies **Indian Traffic Signs**
using a **fine-tuned VGG16 Deep Learning Model**.

**Technology Stack**

- TensorFlow
- Keras
- Streamlit
- VGG16
"""
)