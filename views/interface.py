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

    # Input manuale delle classi
    st.sidebar.subheader("Inserisci le Classi")
    class_input = st.sidebar.text_area(
        "Inserisci le classi (una per riga):",
        placeholder="Es: persona\nauto\nbicicletta"
    )

    # Parsing delle classi inserite
    class_list = [cls.strip() for cls in class_input.split("\n") if cls.strip()]

    selected_class = None
    if class_list:
        st.sidebar.subheader("Seleziona la Classe di Riferimento")
        selected_class = st.sidebar.radio("Classe da usare come ID fisso:", class_list)

    # Creazione istanza dell'engine con la classe fissa selezionata
    engine = InferenceEngine(models, source_type, source, session_id, static_class_name=selected_class, **params)
    
    # Avvio inferenza con classi selezionate e ID fisso
    if st.sidebar.button("Avvia Inferenza") and selected_class:
        engine.run(num_columns)
