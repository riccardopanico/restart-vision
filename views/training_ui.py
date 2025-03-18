import os
import glob
import subprocess
import time
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

    # Scelta del modello e del dataset YOLO
    model_path = select_pretrained_model()
    dataset_path = select_training_dataset()

    if not model_path or not dataset_path:
        return

    # Parametri di training base
    epochs = st.sidebar.slider("Epochs", 1, 500, 50)
    batch_size = st.sidebar.slider("Batch Size", 1, 128, 16)
    learning_rate = st.sidebar.number_input("Learning Rate (lr0)", 1e-6, 1.0, 0.01, step=0.001)

    # Parametri di ottimizzazione avanzati
    optimizer = st.sidebar.selectbox("Optimizer", ["SGD", "Adam"], index=1)
    momentum = st.sidebar.slider("Momentum (valido solo per SGD)", 0.0, 1.0, 0.937, 0.001)

    # Early Stopping
    use_early_stopping = st.sidebar.checkbox("Usa Early Stopping", value=False)
    patience = 50
    if use_early_stopping:
        patience = st.sidebar.slider("Patience Early Stopping (epoch)", 1, 200, 50)

    # Avvio TensorBoard
    start_tensorboard = st.sidebar.checkbox("Lancia TensorBoard", value=False)
    tb_process = None

    if st.button("🚀 Avvia Training"):
        st.success("Training avviato! Controlla la console per i dettagli.")

        # Se l'utente vuole lanciare TensorBoard
        if start_tensorboard:
            log_dir = "runs/train"  # Oppure un percorso specifico
            try:
                tb_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", "6006"])
                time.sleep(2)  # Attendi che TensorBoard si avvii
                st.info("TensorBoard avviato su http://localhost:6006")
            except FileNotFoundError:
                st.warning("TensorBoard non trovato! Assicurati di averlo installato: pip install tensorboard")

        # Carica il modello YOLO
        model = YOLO(model_path)

        # Costruisci i parametri da passare a model.train()
        # YOLOv8 supporta alcuni parametri come `optimizer`, `momentum`, `epochs`, `patience` etc.
        train_args = {
            "data": dataset_path,
            "epochs": epochs,
            "batch": batch_size,
            "lr0": learning_rate,
            "optimizer": optimizer.lower(),  # YOLOv8 vuole "sgd" o "adam"
        }

        # Momentum vale solo per SGD
        if optimizer == "SGD":
            train_args["momentum"] = momentum

        # Se l'utente ha attivato Early Stopping
        if use_early_stopping:
            train_args["patience"] = patience  # Numero di epoch senza miglioramenti

        # Avvia l'allenamento
        model.train(**train_args)

        st.success("Training completato!")

        # Se vuoi terminare automaticamente TensorBoard dopo il training, decommenta:
        # if tb_process:
        #     tb_process.terminate()
        #     st.info("TensorBoard arrestato.")
