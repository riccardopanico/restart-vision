import torch
import streamlit as st

def set_inference_parameters(source_type):
    """Configura i parametri di inferenza per YOLOv8."""
    st.sidebar.subheader("📌 Parametri Inferenza")
    
    # ✅ Evitiamo il refresh impostando i valori di default solo al primo avvio
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    
    device = "cuda" if torch.cuda.is_available() and st.sidebar.checkbox("⚡ Usa CUDA se disponibile", value=True) else "cpu"
    
    save_output = st.sidebar.checkbox("💾 Salva output inferenza", value=False)
    save_video = save_frames = save_annotated_frames = save_labels = save_crop_boxes = False
    images_folder = "images"

    if save_output:
        if source_type != "image":
            save_video = st.sidebar.checkbox("🎥 Salva video", value=True)
        save_frames = st.sidebar.checkbox("🖼️ Salva frames", value=False)
        
        if save_frames:
            save_labels = st.sidebar.checkbox("📝 Salva labels YOLO", value=False)
            save_crop_boxes = st.sidebar.checkbox("✂️ Salva crop dei bounding box", value=False)
            images_folder = st.sidebar.text_input("📂 Nome cartella immagini", value="images")
            save_annotated_frames = st.sidebar.checkbox("📍 Salva frames con box", value=True)

    save_output = save_output and (save_video or save_frames)

    # ✅ Frame skipping solo se inferenza su video/webcam
    frame_skip = 1 if source_type not in ["video", "webcam"] else st.sidebar.slider("🎞️ Inferenza ogni N frame", 1, 30, 1)

    st.sidebar.subheader("📏 Risoluzione Output")
    resolution_options = {
        "Usa risoluzione originale": None,
        "YOLO Default (640x640)": (640, 640),
        "1280x720 (HD)": (1280, 720),
        "1920x1080 (Full HD)": (1920, 1080),
        "256x256 (Bassa Risoluzione)": (256, 256)
    }

    output_resolution_label = st.sidebar.selectbox("📏 Seleziona risoluzione output", list(resolution_options.keys()), index=0)
    output_resolution = resolution_options[output_resolution_label]

    return {
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "device": device,
        "save_output": save_output,
        "save_video": save_video,
        "save_frames": save_frames,
        "save_annotated_frames": save_annotated_frames,
        "save_labels": save_labels,
        "save_crop_boxes": save_crop_boxes,  # ✅ Salvataggio crop dei box abilitato
        "images_folder": images_folder,
        "frame_skip": frame_skip,
        "output_resolution": output_resolution
    }
