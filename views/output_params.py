import os
import cv2
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def list_available_cameras(max_cameras=5):
    """Trova e restituisce un elenco delle webcam disponibili, es. ["Webcam 0", "Webcam 1", ...]."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Webcam {i}")
            cap.release()
    return available_cameras

def select_source():
    """Seleziona la sorgente dell'inferenza."""
    source_options = ["webcam", "video", "image"]

    if "source_type" not in st.session_state:
        st.session_state["source_type"] = source_options[0]  # Imposta il default come "webcam"

    return st.sidebar.selectbox(
        "📡 Seleziona la sorgente",
        source_options,
        index=source_options.index(st.session_state["source_type"]),
        key="source_select"
    )

def upload_file(source_type):
    """Gestisce il caricamento del file video o immagine."""
    if source_type in ["video", "image"]:
        file_types = ["mp4", "mov", "avi", "mkv"] if source_type == "video" else ["jpg", "jpeg", "png"]
        media_file = st.sidebar.file_uploader("📤 Carica un file", type=file_types, key="file_upload")

        if media_file:
            temp_dir = os.path.join(BASE_DIR, "..", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, media_file.name)

            with open(temp_file, "wb") as f:
                f.write(media_file.read())

            return temp_file

    elif source_type == "webcam":
        available_cameras = list_available_cameras()
        if available_cameras:
            selected_camera = st.sidebar.selectbox("📸 Seleziona una webcam:", available_cameras, key="webcam_select")
            try:
                return int(selected_camera.split()[-1][:-1])  # ✅ Converte "Webcam X (Index Y)" in indice Y
            except ValueError:
                st.sidebar.error("⚠️ Errore nella selezione della webcam.")
                return None
        else:
            st.sidebar.warning("⚠️ Nessuna webcam disponibile!")
            return None

    return None
