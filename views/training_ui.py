import os
import glob
import subprocess
import time
import streamlit as st
from ultralytics import YOLO
import io
import sys
from contextlib import redirect_stdout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATASETS_DIR = os.path.join(BASE_DIR, "..", "datasets")

def select_pretrained_model():
    pt_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    if not pt_files:
        st.warning("⚠️ Nessun modello trovato nella cartella 'models'!")
        return None
    model_names = [os.path.basename(p) for p in pt_files]
    selected_model = st.sidebar.selectbox("📌 Seleziona il modello YOLO", model_names, index=0)
    return os.path.join(MODELS_DIR, selected_model)

def select_training_dataset():
    yaml_files = sorted(glob.glob(os.path.join(DATASETS_DIR, "**", "*.yaml"), recursive=True))
    if not yaml_files:
        st.warning("⚠️ Nessun dataset trovato nella cartella 'datasets'!")
        return None
    dataset_names = [os.path.relpath(f, DATASETS_DIR) for f in yaml_files]
    selected_dataset = st.sidebar.selectbox("📂 Seleziona dataset di allenamento", dataset_names, index=0)
    return os.path.join(DATASETS_DIR, selected_dataset)

def training_interface():
    st.subheader("🎯 Training YOLOv8")
    model_path = select_pretrained_model()
    dataset_path = select_training_dataset()
    if not model_path or not dataset_path:
        return

    # Parametri di base
    epochs = st.sidebar.slider("Epochs", 1, 500, 50)
    batch_size = st.sidebar.slider("Batch Size", 1, 128, 16)
    learning_rate = st.sidebar.number_input("Learning Rate (lr0)", 1e-6, 1.0, 0.01, step=0.001)
    optimizer = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"], index=1)
    momentum = None
    if optimizer == "SGD":
        momentum = st.sidebar.slider("Momentum (solo per SGD)", 0.0, 1.0, 0.937, 0.001)
    use_early_stopping = st.sidebar.checkbox("Usa Early Stopping", value=False)
    patience = None
    if use_early_stopping:
        patience = st.sidebar.slider("Patience Early Stopping (epoch)", 1, 200, 50)
    single_cls = st.sidebar.checkbox("Single Class Detection", value=False)

    # Selezione se riprendere da checkpoint (resume) oppure no
    use_resume = st.sidebar.checkbox("Resume training (se possibile)", value=False)

    model = YOLO(model_path)

    # --- Cattura della summary (model.info()) in un buffer per Streamlit
    with io.StringIO() as buf, redirect_stdout(buf):
        model.info()
        summary_str = buf.getvalue()

    # Mostra la summary in un expander
    with st.expander("Mostra struttura modello YOLO"):
        st.text(summary_str)

    st.write("---")
    
    freeze_value = st.sidebar.slider("Numero di layer da congelare (freeze)", 0, 23, 0)

    start_tensorboard = st.sidebar.checkbox("Lancia TensorBoard", value=False)
    tb_process = None

    if st.button("🚀 Avvia Training"):
        st.success("Training avviato! Controlla la console per i dettagli.")

        if start_tensorboard:
            log_dir = "runs/train"
            try:
                tb_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
                time.sleep(2)
                st.info("TensorBoard avviato su http://localhost:6006")
            except FileNotFoundError:
                st.warning("TensorBoard non trovato! Assicurati di averlo installato: pip install tensorboard")

        model = YOLO(model_path)
        train_args = {
            "optimizer" : optimizer,
            "data": dataset_path,
            "epochs": epochs,
            "batch": batch_size,
            "lr0": learning_rate,
            "single_cls": single_cls,
            "freeze": freeze_value 
        }
        if optimizer == "SGD" and momentum is not None:
            train_args["momentum"] = momentum
        if use_early_stopping and patience is not None:
            train_args["patience"] = patience

        if use_resume:
            train_args["resume"] = True
            train_args["data"] = dataset_path
        
        model.train(**train_args)

        st.success("✅ Training completato!")

        if tb_process:
            tb_process.terminate()
            st.info("TensorBoard arrestato.")
