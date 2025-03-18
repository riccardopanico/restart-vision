import streamlit as st
from views.model_selector import select_models
from views.inference_params import set_inference_parameters
from views.output_params import select_source, upload_file, list_available_cameras
from views.dataset_manager import dataset_management_ui
from inference.inference import InferenceEngine
from views.training_ui import training_interface
from datetime import datetime

def run_app():
    """Gestisce la UI Streamlit e avvia l'inferenza."""
    st.title("YOLOv8 Model Inference")
    
    # **GESTIONE DATASET & TRAINING**
    st.sidebar.header("Gestione Dataset")
    dataset_management_ui()
    
    st.sidebar.header("Training YOLOv8")
    training_interface()

    # **SELEZIONE MODELLI**
    models = select_models()
    
    # **SELEZIONE DELLA SORGENTE**
    source_type = select_source()
    
    # **GESTIONE DELLA SORGENTE**
    source = None

    if source_type == "webcam":
        available_cameras = list_available_cameras()
        if available_cameras:
            # Esempio: "Webcam 0", "Webcam 1"
            selected_camera = st.sidebar.selectbox("Seleziona una webcam:", available_cameras, key="webcam_select")
            try:
                # "Webcam 0" -> split()[-1] = "0"
                source = int(selected_camera.split()[-1])
                st.sidebar.success(f"📸 Webcam selezionata con indice: {source}")
            except ValueError:
                st.sidebar.error("⚠️ Errore nella selezione della webcam.")
                return
        else:
            st.sidebar.warning("⚠️ Nessuna webcam disponibile!")
            return
    else:
        source = upload_file(source_type)

    if source is None:
        st.warning("⚠️ Seleziona una sorgente valida prima di avviare l'inferenza.")
        return

    # **PARAMETRI INFERENZA**
    params = set_inference_parameters(source_type)
        # **GESTIONE CLASSI PERSONALIZZATE**
    st.sidebar.subheader("Inserisci le Classi")
    default_classes = "class_1\nclass_2\nclass_3"

    if "class_input" not in st.session_state:
        st.session_state["class_input"] = default_classes  

    class_input = st.sidebar.text_area(
        "Inserisci le classi (una per riga):",
        value=st.session_state["class_input"],  
        key="class_textarea"
    )

    # **Aggiorniamo il valore in session_state solo se è cambiato**
    if class_input != st.session_state["class_input"]:
        st.session_state["class_input"] = class_input

    # **Conversione classi in lista**
    class_list = [cls.strip() for cls in class_input.split("\n") if cls.strip()]

    # **SELEZIONE CLASSE DI RIFERIMENTO (RADIO BUTTON)**
    selected_class = None
    selected_class_id = None

    if class_list:
        st.sidebar.subheader("Seleziona la Classe di Riferimento")
        selected_class = st.sidebar.radio("Classe da usare come ID fisso:", class_list, key="class_radio")

        if selected_class:
            selected_class_id = class_list.index(selected_class)

    # **CONFIGURAZIONE GRIGLIA**
    num_columns = st.sidebar.slider("Numero di colonne", 1, 12, 3)

    # **CREAZIONE MOTORE DI INFERENZA**
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if params["save_output"] else "temp_session"
    engine = InferenceEngine(models, source_type, source, session_id, **params)
    
    # **AVVIO INFERENZA**
    if st.sidebar.button("Avvia Inferenza"):
        engine.run(num_columns)
