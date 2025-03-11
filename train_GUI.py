import os
import glob
import subprocess
from datetime import datetime

import streamlit as st
from ultralytics import YOLO

# Imposta base_dir e models_dir
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
datasets_dir = os.path.join(base_dir, "datasets")

def select_pretrained_model():
    """
    Restituisce il percorso completo del modello selezionato,
    cercandolo nella cartella models_dir.
    """
    pt_files = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    if not pt_files:
        st.warning("Nessun file .pt trovato nella cartella 'models'!")
        return None
    
    # Nome dei file .pt, senza path
    model_names = [os.path.basename(p) for p in pt_files]
    # Menu a tendina, simile al multiselect (ma qui facciamo single select)
    selected_model = st.sidebar.selectbox(
        "Seleziona il modello pre-addestrato",
        model_names,
        index=0
    )
    # Ritorna il path completo
    return os.path.join(models_dir, selected_model)

def select_dataset():
    """
    Cerca nelle sottocartelle di datasets_dir i file .yaml e 
    propone un menu per selezionare il dataset YOLO da allenare.
    Restituisce il path completo del file .yaml scelto.
    """
    dataset_options = []
    for root, dirs, files in os.walk(datasets_dir):
        for f in files:
            if f.endswith(".yaml"):
                dataset_options.append(os.path.join(root, f))

    if not dataset_options:
        st.warning("Nessun file .yaml trovato nella cartella 'datasets'!")
        return None

    # Ordina alfabeticamente i .yaml trovati
    dataset_options.sort()
    # Estrai solo la parte finale del path per il menu
    dataset_labels = [os.path.relpath(d, datasets_dir) for d in dataset_options]
    selected_dataset = st.sidebar.selectbox(
        "Seleziona il dataset di allenamento",
        dataset_labels,
        index=0
    )
    # Trova il path completo effettivo
    full_path = os.path.join(datasets_dir, selected_dataset)
    return full_path

def start_tensorboard(logdir, port=6006):
    """
    Avvia TensorBoard come processo separato, se desiderato.
    """
    tb_cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
    process = subprocess.Popen(tb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main():
    st.set_page_config(page_title="YOLO Training", layout="wide")
    st.title("Pannello di Training YOLO")

    st.sidebar.header("Configurazione Allenamento")

    # --- Selezione modello pre-addestrato ---
    model_path = select_pretrained_model()
    
    # --- Selezione dataset YOLO ---
    dataset_path = select_dataset()

    # --- Parametri di training (riprendi quelli del tuo train.py) ---
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=10000, value=50)
    batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=1024, value=16)
    learning_rate = st.sidebar.number_input("Learning rate (lr0)", min_value=1e-6, max_value=1.0, value=0.01, step=0.001)
    momentum = st.sidebar.slider("Momentum", 0.0, 1.0, 0.937, 0.001)
    optimizer = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "auto"], index=0)
    
    project_name = st.sidebar.text_input("Project Name", value="runs/train")
    default_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_name = st.sidebar.text_input("Nome Esperimento", value=default_name)
    
    single_class = st.sidebar.checkbox("Modalità single_class?", value=False)
    validate = st.sidebar.checkbox("Effettua validazione?", value=False)
    resume = st.sidebar.checkbox("Resume training?", value=False)
    
    # Bottone di avvio
    if st.sidebar.button("Avvia Training"):
        if not model_path or not os.path.exists(model_path):
            st.error("Percorso modello .pt non valido o non selezionato!")
            st.stop()
        if not dataset_path or not os.path.exists(dataset_path):
            st.error("File .yaml per il dataset non trovato o non selezionato!")
            st.stop()

        st.success("Inizializzo training...")

        # Avvio TensorBoard (opzionale)
        tb_process = start_tensorboard(logdir=project_name, port=6006)
        st.write("**TensorBoard avviato**. Visita [localhost:6006](http://localhost:6006) per i log.")

        # Carica il modello e avvia addestramento
        with st.spinner("Training in corso..."):
            try:
                model = YOLO(model_path)
                model.train(
                    data=dataset_path,
                    epochs=epochs,
                    batch=batch_size,
                    lr0=learning_rate,
                    momentum=momentum,
                    optimizer=optimizer,
                    project=project_name,
                    name=experiment_name,
                    resume=resume,
                    val=validate,
                    single_cls=single_class
                )
                st.success(f"Training completato! Risultati in {project_name}/{experiment_name}")
            except Exception as e:
                st.error(f"Errore durante il training: {e}")
            finally:
                # Se vuoi terminare automaticamente TensorBoard alla fine:
                # tb_process.terminate()
                pass

if __name__ == "__main__":
    main()
