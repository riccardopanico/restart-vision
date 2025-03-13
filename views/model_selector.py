import os
import glob
import streamlit as st
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def select_models():
    """Permette di selezionare i modelli disponibili."""
    models_dir = os.path.join(BASE_DIR, "..", "models")
    pt_files = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    model_names = [os.path.basename(p) for p in pt_files]
    
    selected_models = st.sidebar.multiselect("Seleziona i modelli", model_names, default=model_names[:1])
    
    return {model: YOLO(os.path.join(models_dir, model)) for model in selected_models}
