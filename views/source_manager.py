import os
import time
import cv2
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def list_available_cameras(max_cameras=10):
    """Trova e restituisce un elenco delle webcam disponibili."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Webcam {i} (Index {i})")
            cap.release()
    return available_cameras

def select_source():
    """Gestisce la selezione della sorgente di inferenza (webcam, video o immagine)."""
    source_options = ["webcam", "video", "image"]

    if "source_type" not in st.session_state:
        st.session_state["source_type"] = source_options[0]

    selected_source = st.sidebar.selectbox("📡 Seleziona la sorgente", source_options, 
                                           index=source_options.index(st.session_state["source_type"]),
                                           key="source_select")

    st.session_state["source_type"] = selected_source

    return selected_source

def upload_file(source_type):
    """Gestisce il caricamento di file immagine o video."""
    file_types = {
        "video": ["mp4", "mov", "avi", "mkv", "webm"],
        "image": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    }.get(source_type, [])

    media_file = st.sidebar.file_uploader("📤 Carica un file", type=file_types, key="file_upload")

    if media_file:
        temp_dir = os.path.join(BASE_DIR, "..", "temp")
        os.makedirs(temp_dir, exist_ok=True)

        unique_filename = f"{int(time.time())}_{media_file.name}"
        temp_file = os.path.join(temp_dir, unique_filename)

        with open(temp_file, "wb") as f:
            f.write(media_file.read())

        return temp_file

    return None

def select_webcam():
    """Gestisce la selezione della webcam disponibile."""
    available_cameras = list_available_cameras()
    if available_cameras:
        selected_camera = st.sidebar.selectbox("📸 Seleziona una webcam:", available_cameras, key="webcam_select")

        try:
            return int(selected_camera.split("Index ")[-1].strip(")"))
        except (ValueError, IndexError):
            st.sidebar.error("⚠️ Errore nella selezione della webcam.")
            return None
    else:
        st.sidebar.warning("⚠️ Nessuna webcam disponibile!")
        return None
