import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def select_source():
    """Seleziona la sorgente dell'inferenza."""
    return st.sidebar.selectbox("Seleziona la sorgente", ["webcam", "video", "image"])

def upload_file(source_type):
    """Gestisce il caricamento del file video o immagine."""
    if source_type in ["video", "image"]:
        file_types = ["mp4", "mov", "avi", "mkv"] if source_type == "video" else ["jpg", "jpeg", "png"]
        media_file = st.sidebar.file_uploader("Carica un file", type=file_types)

        if media_file:
            temp_dir = os.path.join(BASE_DIR, "..", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, media_file.name)

            with open(temp_file, "wb") as f:
                f.write(media_file.read())

            return temp_file
    elif source_type == "webcam":
        return 0
    
    return None
