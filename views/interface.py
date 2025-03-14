import streamlit as st
from views.model_selector import select_models
from views.inference_params import set_inference_parameters
from views.output_params import select_source, upload_file
from views.dataset_manager import dataset_management_ui
from inference.inference import InferenceEngine
from views.training_ui import training_interface
from datetime import datetime

def run_app():
    """Gestisce la UI Streamlit e avvia l'inferenza."""
    st.title("YOLOv8 Model Inference")
    st.sidebar.header("Configurazione")
    
    st.sidebar.header("Gestione Dataset")
    dataset_management_ui()
    
    st.sidebar.header("Training YOLOv8")
    training_interface()

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

    st.sidebar.subheader("Inserisci le Classi")
    default_classes = "class_1\nclass_2\nclass_3"

    class_input = st.sidebar.text_area(
        "Inserisci le classi (una per riga):",
        value=st.session_state.get("class_input", default_classes),  
        key="class_textarea"  
    )

    # ✅ Salviamo il valore in session_state per poterlo riutilizzare
    st.session_state["class_input"] = class_input

    # ✅ Creiamo una lista delle classi per il merge
    class_list = [cls.strip() for cls in class_input.split("\n") if cls.strip()]

    selected_class = None
    selected_class_id = None
    if class_list:
        st.sidebar.subheader("Seleziona la Classe di Riferimento")
        selected_class = st.sidebar.radio("Classe da usare come ID fisso:", class_list)

        # ✅ Troviamo l'ID numerico della classe selezionata
        selected_class_id = class_list.index(selected_class)

    # Creazione istanza dell'engine con la classe fissa selezionata
    engine = InferenceEngine(models, source_type, source, session_id, static_class_id=selected_class_id, **params)
    
    # Avvio inferenza con classi selezionate e ID fisso
    if st.sidebar.button("Avvia Inferenza") and selected_class:
        engine.run(num_columns)
