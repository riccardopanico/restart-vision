import torch
import streamlit as st

def set_inference_parameters(source_type):
    st.sidebar.subheader("📌 Parametri Inferenza")

    if "inference_params" not in st.session_state:
        st.session_state["inference_params"] = {
            "confidence": 0.5,
            "iou_threshold": 0.45,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_output": False,
            "save_video": False,
            "save_frames": False,
            "save_annotated_frames": False,
            "save_labels": False,
            "save_crop_boxes": False,
            "images_folder": "images",
            "frame_skip": 1,
            "output_resolution": None,
            "num_workers": 4  # Default: usa 4 worker
        }

    params = st.session_state["inference_params"]

    params["confidence"] = st.sidebar.slider("🎯 Confidence Threshold", 0.0, 1.0, params["confidence"], 0.01)
    params["iou_threshold"] = st.sidebar.slider("📏 IoU Threshold", 0.0, 1.0, params["iou_threshold"], 0.01)

    params["device"] = st.sidebar.selectbox("💻 Seleziona dispositivo", ["cuda", "mps", "cpu"], index=["cuda", "mps", "cpu"].index(params["device"]))

    params["save_output"] = st.sidebar.checkbox("💾 Salva output inferenza", value=params["save_output"])
    if params["save_output"]:
        if source_type != "image":
            params["save_video"] = st.sidebar.checkbox("🎥 Salva video", value=params["save_video"])
        params["save_frames"] = st.sidebar.checkbox("🖼️ Salva frames", value=params["save_frames"])

        if params["save_frames"]:
            params["save_labels"] = st.sidebar.checkbox("📝 Salva labels YOLO", value=params["save_labels"])
            params["save_crop_boxes"] = st.sidebar.checkbox("✂️ Salva crop dei bounding box", value=params["save_crop_boxes"])
            params["images_folder"] = st.sidebar.text_input("📂 Cartella immagini", value=params["images_folder"])
            params["save_annotated_frames"] = st.sidebar.checkbox("📍 Salva frames con box", value=params["save_annotated_frames"])

    params["frame_skip"] = st.sidebar.slider("🎞️ Inferenza ogni N frame", 1, 30, params["frame_skip"])
    params["num_workers"] = st.sidebar.slider("🛠️ Worker inferenza", 1, 8, params["num_workers"])

    st.sidebar.subheader("📏 Risoluzione Output")
    resolution_options = {
        "Usa risoluzione originale": None,
        "YOLO Default (640x640)": (640, 640),
        "1280x720 (HD)": (1280, 720),
        "1920x1080 (Full HD)": (1920, 1080),
        "256x256 (Bassa Risoluzione)": (256, 256)
    }

    output_resolution_label = st.sidebar.selectbox("📏 Seleziona risoluzione output", list(resolution_options.keys()), index=list(resolution_options.values()).index(params["output_resolution"]))
    params["output_resolution"] = resolution_options[output_resolution_label]

    st.session_state["inference_params"] = params
    return params
