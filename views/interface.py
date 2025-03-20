import streamlit as st
from views.model_selector import select_models
from views.inference_params import set_inference_parameters
from views.source_manager import select_source, upload_file, select_webcam
from views.dataset_manager import dataset_management_ui
from inference.inference import InferenceEngine
from views.training_ui import training_interface
from datetime import datetime

def run_app():
    st.title("YOLO Model Inference")

    dataset_management_ui()

    st.sidebar.header("Training YOLO")
    training_interface()

    models = select_models()

    source_type = select_source()
    source = select_webcam() if source_type == "webcam" else upload_file(source_type)

    if source is None:
        st.warning("⚠️ Seleziona una sorgente valida prima di avviare l'inferenza.")
        return

    params = set_inference_parameters(source_type)

    st.sidebar.subheader("Inserisci le Classi")
    default_classes = "model_1\nmodel_2\nmodel_3"

    class_input = st.sidebar.text_area("Inserisci le classi (una per riga):",
                                       value=st.session_state.get("class_input", default_classes),
                                       key="class_textarea")

    class_list = [cls.strip() for cls in class_input.split("\n") if cls.strip()]
    st.session_state["class_input"] = class_input

    selected_class = None
    selected_class_id = None

    if class_list:
        st.sidebar.subheader("Seleziona la Classe di Riferimento")
        selected_class = st.sidebar.radio("Classe da usare come ID fisso:", class_list, key="class_radio")

        if selected_class:
            selected_class_id = class_list.index(selected_class)

    num_columns = st.sidebar.slider("Numero di colonne", 1, 12, 3)

    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if params["save_output"] else "temp_session"
    engine = InferenceEngine(models, source_type, source, session_id,
                             static_class_id=selected_class_id, **params)

    if st.sidebar.button("Avvia Inferenza"):
        engine.run(num_columns)
