import streamlit as st
from views.model_selector import select_models
from views.inference_params import set_inference_parameters
from views.output_params import select_source, upload_file
from inference.inference import InferenceEngine
from datetime import datetime

def run_app():
    """Gestisce la UI Streamlit e avvia l'inferenza."""
    st.title("YOLOv8 Model Inference")
    st.sidebar.header("Configurazione")

    # Selezione modelli e sorgente
    models = select_models()
    source_type = select_source()
    source = upload_file(source_type)

    if source is None:
        st.warning("Seleziona una sorgente valida prima di avviare l'inferenza.")
        return

    # Parametri inferenza
    params = set_inference_parameters(source_type)
    
    # Numero colonne nella griglia
    num_columns = st.sidebar.slider("Numero di colonne", 1, 12, 3)

    # Creazione istanza dell'engine
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if params["save_output"] else None
    engine = InferenceEngine(models, source_type, source, session_id, **params)

    if st.sidebar.button("Avvia Inferenza"):
        engine.run(num_columns)
