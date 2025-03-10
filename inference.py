import os
import cv2
import glob
import streamlit as st
import torch
from datetime import datetime
from ultralytics import YOLO

base_dir = os.path.dirname(os.path.abspath(__file__))

def create_output_dir(model_name, session_id, subfolder):
    output_dir = os.path.join(base_dir, "output", model_name, session_id, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def select_models():
    models_dir = os.path.join(base_dir, "models")
    pt_files = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    model_names = [os.path.basename(p) for p in pt_files]
    selected_models = st.sidebar.multiselect("Seleziona i modelli", model_names, default=model_names[:1])
    return {model: YOLO(os.path.join(models_dir, model)) for model in selected_models}

def select_source():
    return st.sidebar.selectbox("Seleziona la sorgente", ["webcam", "video", "image"])

def upload_file(source_type):
    if source_type in ["video", "image"]:
        file_types = ["mp4", "mov", "avi", "mkv"] if source_type == "video" else ["jpg", "jpeg", "png"]
        media_file = st.sidebar.file_uploader("Carica un file", type=file_types)
        if media_file:
            temp_dir = os.path.join(base_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, media_file.name)
            with open(temp_file, "wb") as f:
                f.write(media_file.read())
            return temp_file
    elif source_type == "webcam":
        return 0
    return None

def set_inference_parameters(source_type):
    st.sidebar.subheader("Parametri Inferenza")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)
    device = "cuda" if st.sidebar.checkbox("Usa CUDA se disponibile", value=True) and torch.cuda.is_available() else "cpu"
    save_output = st.sidebar.checkbox("Salva output inferenza")
    save_video, save_frames, save_annotated_frames, save_labels = False, False, False, False
    images_folder = "images"
    
    if save_output:
        if source_type != "image":  # Il checkbox "Salva video" viene mostrato solo per video/webcam
            save_video = st.sidebar.checkbox("Salva video", value=True)
        save_frames = st.sidebar.checkbox("Salva frames", value=False)
        if save_frames:
            save_labels = st.sidebar.checkbox("Salva labels YOLO", value=False)
            images_folder = st.sidebar.text_input("Nome cartella immagini (default: images)", value="images") or "images"
            save_annotated_frames = st.sidebar.checkbox("Salva frames con box", value=True)

    save_output = save_output and (save_video or save_frames)
    frame_skip = 1  # Default: inferenza su tutti i frame
    if source_type in ["video", "webcam"]:
        frame_skip = st.sidebar.slider("Inferenza ogni N frame", 1, 30, 1)
        
    st.sidebar.subheader("Risoluzione Output")

    # Definizione delle risoluzioni disponibili
    resolution_options = {
        "Usa risoluzione originale": None,  # Nessun ridimensionamento se selezionato
        "YOLO Default (640x640)": (640, 640),
        "1280x720 (HD)": (1280, 720),
        "1920x1080 (Full HD)": (1920, 1080),
        "256x256 (Bassa Risoluzione)": (256, 256)
    }

    # L'utente può scegliere la risoluzione, di default mantiene l'originale
    output_resolution_label = st.sidebar.selectbox("Seleziona risoluzione output", list(resolution_options.keys()), index=0)
    output_resolution = resolution_options[output_resolution_label]

    return confidence, iou_threshold, device, save_output, save_video, save_frames, save_annotated_frames, save_labels, images_folder, frame_skip, output_resolution


def save_yolo_labels(output_dir, model_name, session_id, frame_counter, results, original_resolution, output_resolution):
    labels_dir = os.path.join(base_dir, "output", model_name, session_id, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")
    with open(label_file, "w") as f:
        for box in results[0].boxes:
            class_id = int(box.cls)
            if output_resolution is not None:
                img_width, img_height = output_resolution
                orig_width, orig_height = original_resolution
                x_center, y_center, width, height = box.xywh.tolist()[0]  # Coordinate assolute
                
                # Scala le coordinate rispetto alla nuova risoluzione
                x_center = (x_center / orig_width) * img_width
                y_center = (y_center / orig_height) * img_height
                width = (width / orig_width) * img_width
                height = (height / orig_height) * img_height
                
                # Normalizza le coordinate per YOLO
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
            else:
                x_center, y_center, width, height = box.xywhn.tolist()[0]  # Già normalizzato


            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def perform_inference(models, source_type, source, num_columns, confidence, iou_threshold, device, save_output, save_video, save_frames, save_annotated_frames, save_labels, images_folder, session_id, frame_skip, output_resolution):
    grid = [st.columns(num_columns) for _ in range((len(models) + num_columns - 1) // num_columns)]
    cap = None

    if source_type == "image":
        frame = cv2.imread(source)
        if frame is None:
            st.error("Errore nel caricamento dell'immagine.")
            return
        original_resolution = (frame.shape[1], frame.shape[0])  # (width, height)

        for model_name, model in models.items():
            output_dir = create_output_dir(model_name, session_id, images_folder) if save_output else None
            row_idx, col_idx = divmod(list(models.keys()).index(model_name), num_columns)
            model.to(device)
            results = model(frame, conf=confidence, iou=iou_threshold)
            annotated_frame = results[0].plot()
            if output_resolution is not None:
                annotated_frame = cv2.resize(annotated_frame, output_resolution)
            grid[row_idx][col_idx].image(annotated_frame, channels="BGR")

            if save_output:
                frame_to_save = annotated_frame if save_annotated_frames else frame
                cv2.imwrite(os.path.join(output_dir, f"frame_0.jpg"), frame_to_save)
                if save_labels:
                    save_yolo_labels(output_dir, model_name, session_id, 0, results, original_resolution, output_resolution)

    elif source_type in ["video", "webcam"]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return
        stop_button = st.sidebar.button("Stop Inferenza")
        frame_holders = [col.empty() for row in grid for col in row]

        writers = {model_name: cv2.VideoWriter(
            os.path.join(create_output_dir(model_name, session_id, "videos"), "output.avi"),
            cv2.VideoWriter_fourcc(*'XVID'), 20.0, output_resolution
        ) if save_output and save_video else None for model_name in models.keys()}

        frame_counter = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success or stop_button:
                break
            frame_counter += 1
            original_resolution = (frame.shape[1], frame.shape[0])  # (width, height)

            for i, (model_name, model) in enumerate(models.items()):
                model.to(device)
                results = model(frame, conf=confidence, iou=iou_threshold)
                annotated_frame = results[0].plot()
                frame_holders[i].image(annotated_frame, channels="BGR")

                if save_output and frame_counter % frame_skip == 0:  # Condiziona solo il salvataggio
                    if save_frames and len(results[0].boxes) > 0:
                        output_dir = create_output_dir(model_name, session_id, images_folder)
                        frame_to_save = annotated_frame if save_annotated_frames else frame
                        if output_resolution is not None:
                            frame_to_save = cv2.resize(frame_to_save, output_resolution)
                        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_counter}.jpg"), frame_to_save)
                        if save_labels:
                            save_yolo_labels(output_dir, model_name, session_id, frame_counter, results, original_resolution, output_resolution)

                    if save_video and writers[model_name]:
                        writers[model_name].write(annotated_frame)

        cap.release()
        for writer in writers.values():
            if writer:
                writer.release()
        cv2.destroyAllWindows()

def main():
    st.set_page_config(page_title="Model Inference", layout="wide")
    st.title("Modello di Inferenza Generale")
    st.sidebar.header("Configurazione")
    models = select_models()
    source_type = select_source()
    source = upload_file(source_type)
    confidence, iou_threshold, device, save_output, save_video, save_frames, save_annotated_frames, save_labels, images_folder, frame_skip, output_resolution = set_inference_parameters(source_type)
    num_columns = st.sidebar.slider("Numero di colonne", 1, 12, 3)
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if save_output else None
    if st.sidebar.button("Avvia Inferenza") and source is not None:
        perform_inference(models, source_type, source, num_columns, confidence, iou_threshold, device, save_output, save_video, save_frames, save_annotated_frames, save_labels, images_folder, session_id, frame_skip, output_resolution)

if __name__ == "__main__":
    main()
