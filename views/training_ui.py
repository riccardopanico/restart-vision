import os
import glob
import streamlit as st
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")

def select_pretrained_model():
    """Permette la selezione di un modello YOLO pre-addestrato."""
    pt_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    if not pt_files:
        st.warning("⚠️ Nessun modello trovato nella cartella 'models'!")
        return None

    model_names = [os.path.basename(p) for p in pt_files]
    selected_model = st.sidebar.selectbox("📌 Seleziona il modello YOLO", model_names, index=0)
    return os.path.join(MODELS_DIR, selected_model)

def select_training_dataset():
    """Permette la selezione di un dataset YOLO esistente."""
    yaml_files = sorted(glob.glob(os.path.join(DATASETS_DIR, "**", "*.yaml"), recursive=True))
    if not yaml_files:
        st.warning("⚠️ Nessun dataset trovato nella cartella 'datasets'!")
        return None

    dataset_names = [os.path.relpath(f, DATASETS_DIR) for f in yaml_files]
    selected_dataset = st.sidebar.selectbox("📂 Seleziona dataset di allenamento", dataset_names, index=0)
    return os.path.join(DATASETS_DIR, selected_dataset)

def training_interface():
    """Interfaccia Streamlit per avviare il training YOLOv8."""
    st.subheader("🎯 Training YOLOv8")

    model_path = select_pretrained_model()
    dataset_path = select_training_dataset()

    if not model_path or not dataset_path:
        return

    epochs = st.sidebar.slider("Epochs", 1, 500, 50)
    batch_size = st.sidebar.slider("Batch Size", 1, 128, 16)
    learning_rate = st.sidebar.number_input("Learning Rate", 1e-6, 1.0, 0.01, step=0.001)

    if st.button("🚀 Avvia Training"):
        st.success("Training avviato! Controlla la console per i dettagli.")
        model = YOLO(model_path)
        model.train(data=dataset_path, epochs=epochs, batch=batch_size, lr0=learning_rate)
