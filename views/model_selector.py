import os
import glob
import streamlit as st
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

def get_model_info(model_path):
    size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    last_modified = os.path.getmtime(model_path)
    return {"size": f"{size:.2f} MB", "last_modified": last_modified}

def select_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")) + glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    model_names = [os.path.basename(p) for p in model_files]

    if not model_names:
        st.sidebar.warning("⚠️ Nessun modello trovato in 'models/'!")
        return {}

    # ✅ Manteniamo la selezione in session_state per evitare refresh
    if "selected_models" not in st.session_state:
        st.session_state["selected_models"] = model_names[:1]

    selected_models = st.sidebar.multiselect("📌 Seleziona i modelli", model_names, default=st.session_state["selected_models"])
    st.session_state["selected_models"] = selected_models

    if selected_models:
        st.sidebar.subheader("📊 Dettagli Modelli Selezionati")
        for model in selected_models:
            model_path = os.path.join(MODELS_DIR, model)
            info = get_model_info(model_path)
            st.sidebar.text(f"{model} - {info['size']}")

    return {model: YOLO(os.path.join(MODELS_DIR, model)) for model in selected_models}
