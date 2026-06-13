import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: #0d0d0d; }

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00f5c4, #0099ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-top: 0.3rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
}

.result-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid #00f5c420;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}

.label-chip {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.chip-tumor { background: #ff4d4d22; color: #ff6b6b; border: 1px solid #ff4d4d55; }
.chip-notumor { background: #00f5c422; color: #00f5c4; border: 1px solid #00f5c455; }

.conf-bar-bg {
    background: #1e1e2e;
    border-radius: 8px;
    height: 10px;
    margin: 4px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #00f5c4, #0099ff);
}

.info-box {
    background: #111827;
    border-left: 3px solid #00f5c4;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    color: #aaa;
    margin: 1rem 0;
}

div[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #00f5c440;
    border-radius: 12px;
    padding: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5c4, #0099ff);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

hr { border-color: #1e1e2e; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)
MODEL_PATH = "best_brain_tumor_model.keras"   # ← put your saved model here
LAST_CONV_LAYER = "Conv_1_bn"                 # MobileNetV2 last conv BN layer

CLASS_INFO = {
    "glioma":     ("Glioma", "🔴", "chip-tumor",   "Tumor arising from glial cells. Most common primary brain tumor."),
    "meningioma": ("Meningioma", "🟠", "chip-tumor", "Tumor from meninges (brain lining). Often benign but needs monitoring."),
    "notumor":    ("No Tumor", "🟢", "chip-notumor", "No tumor detected in the scan."),
    "pituitary":  ("Pituitary Tumor", "🟡", "chip-tumor", "Tumor in the pituitary gland. Affects hormone regulation."),
}

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(uploaded_file):
    """Handle JPG/PNG uploads → (224,224,3) float32 array normalised to [0,1]."""
    img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def preprocess_npy(uploaded_file):
    """Handle .npy uploads (already 224×224×3)."""
    arr = np.load(io.BytesIO(uploaded_file.read())).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.shape != (224, 224, 3):
        st.error(f"Expected shape (224,224,3), got {arr.shape}")
        return None
    return arr

# ─── GradCAM ──────────────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(255 * img) if img.max() <= 1.0 else np.uint8(img)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

# ─── UI ────────────────────────────────────────────────────────────────────────
def render_confidence_bars(probs):
    for cls, p in zip(CLASSES, probs):
        label, icon, _, _ = CLASS_INFO[cls]
        bar_html = f"""
        <div style="margin:6px 0">
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;color:#ccc;">
                <span>{icon} {label}</span><span>{p*100:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{p*100:.1f}%"></div>
            </div>
        </div>"""
        st.markdown(bar_html, unsafe_allow_html=True)

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🧠 About")
        st.markdown("""
        **Brain Tumor MRI Classifier**  
        Built with MobileNetV2 + GradCAM  
        
        **Classes detected:**
        - 🔴 Glioma
        - 🟠 Meningioma
        - 🟢 No Tumor
        - 🟡 Pituitary Tumor
        """)
        st.markdown("---")
        st.markdown("**Model file:** `best_brain_tumor_model.keras`")
        st.markdown("**Input:** 224 × 224 × 3 RGB")
        st.markdown("**Backbone:** MobileNetV2")
        show_gradcam = st.checkbox("Show GradCAM heatmap", value=True)
        alpha = st.slider("GradCAM overlay strength", 0.2, 0.8, 0.4, 0.05)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="hero-title">Brain Tumor Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload an MRI scan — get an instant AI prediction with explainability.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model check ──────────────────────────────────────────────────────────
    model = load_keras_model()
    if model is None:
        st.error(f"⚠️ Model file `{MODEL_PATH}` not found. Place it in the same folder as `app.py`.")
        st.info("Run your training notebook first, then copy `best_brain_tumor_model.keras` here.")
        st.stop()

    # ── Upload ───────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop an MRI image here",
        type=["jpg", "jpeg", "png", "npy"],
        help="Supports JPG, PNG, or .npy (224×224×3)"
    )

    if uploaded is None:
        st.markdown("""
        <div class="info-box">
        👆 Upload an MRI scan above to get started. Supported formats: <b>JPG / PNG / .npy</b>
        </div>""", unsafe_allow_html=True)
        return

    # ── Preprocess ───────────────────────────────────────────────────────────
    with st.spinner("Processing image…"):
        if uploaded.name.endswith(".npy"):
            img = preprocess_npy(uploaded)
        else:
            img = preprocess_image(uploaded)

    if img is None:
        return

    # ── Predict ──────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        inp = img[np.newaxis, ...]
        probs = model.predict(inp, verbose=0)[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]

    label, icon, chip_cls, description = CLASS_INFO[pred_class]

    # ── GradCAM ──────────────────────────────────────────────────────────────
    if show_gradcam:
        with st.spinner("Generating GradCAM…"):
            try:
                heatmap = make_gradcam_heatmap(inp, model, LAST_CONV_LAYER, pred_index=pred_idx)
                gradcam_img = overlay_gradcam(img, heatmap, alpha=alpha)
            except Exception as e:
                show_gradcam = False
                st.warning(f"GradCAM failed: {e}")

    # ── Layout ───────────────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Original MRI")
        display_img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        st.image(display_img, use_container_width=True, caption="Input scan")

        if show_gradcam:
            st.markdown("#### GradCAM Heatmap")
            st.image(gradcam_img, use_container_width=True, caption="Regions influencing prediction")

    with col2:
        st.markdown("#### Prediction Result")
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;margin-bottom:0.5rem">{icon}</div>
            <div style="font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;color:#f0f0f0">
                {label}
            </div>
            <span class="label-chip {chip_cls}" style="margin:0.5rem 0;display:inline-block">
                {pred_class.upper()}
            </span>
            <div style="font-size:2rem;font-weight:600;color:#00f5c4;margin:0.8rem 0">
                {confidence*100:.1f}% confidence
            </div>
            <div style="color:#888;font-size:0.9rem;line-height:1.5">{description}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Class Probabilities")
        render_confidence_bars(probs)

        # Download GradCAM
        if show_gradcam:
            buf = io.BytesIO()
            plt.imsave(buf, gradcam_img, format="png")
            st.download_button(
                "⬇ Download GradCAM image",
                data=buf.getvalue(),
                file_name="gradcam_result.png",
                mime="image/png"
            )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#444;font-size:0.8rem'>"
        "⚠️ For research purposes only — not a medical diagnostic tool."
        "</div>", unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
