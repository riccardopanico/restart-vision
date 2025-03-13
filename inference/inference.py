import os
import cv2
import streamlit as st
from ultralytics import YOLO

class InferenceEngine:
    """Classe che esegue l'inferenza YOLOv8 separata dalla UI."""

    def __init__(self, models, source_type, source, session_id, **params):
        self.models = models
        self.source_type = source_type
        self.source = source
        self.session_id = session_id
        self.params = params

    def run(self, num_columns):
        """Esegue inferenza e visualizza i risultati."""
        cap = None
        grid = [st.columns(num_columns) for _ in range((len(self.models) + num_columns - 1) // num_columns)]

        if self.source_type == "image":
            self._process_image(grid)
        elif self.source_type in ["video", "webcam"]:
            self._process_video(grid)

    def _create_output_dir(self, model_name, subfolder):
        """Crea la cartella di output per il modello e sessione attuale."""
        output_dir = os.path.join("output", model_name, self.session_id, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_yolo_labels(self, output_dir, model_name, frame_counter, results, original_resolution, output_resolution):
        """Salva le etichette YOLO in un file di testo."""
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")

        with open(label_file, "w") as f:
            for box in results[0].boxes:
                class_id = int(box.cls)
                x_center, y_center, width, height = box.xywhn.tolist()[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    def _process_image(self, grid):
        """Inferenza su immagine."""
        frame = cv2.imread(self.source)
        if frame is None:
            st.error("Errore nel caricamento dell'immagine.")
            return

        original_resolution = (frame.shape[1], frame.shape[0])

        for model_name, model in self.models.items():
            output_dir = self._create_output_dir(model_name, self.params["images_folder"]) if self.params["save_output"] else None
            row_idx, col_idx = divmod(list(self.models.keys()).index(model_name), len(grid[0]))

            model.to(self.params["device"])
            results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
            annotated_frame = results[0].plot()

            if self.params["output_resolution"]:
                annotated_frame = cv2.resize(annotated_frame, self.params["output_resolution"])

            grid[row_idx][col_idx].image(annotated_frame, channels="BGR")

            if self.params["save_output"]:
                frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                cv2.imwrite(os.path.join(output_dir, "frame_0.jpg"), frame_to_save)

                if self.params["save_labels"]:
                    self._save_yolo_labels(output_dir, model_name, 0, results, original_resolution, self.params["output_resolution"])
                    
    def _process_video(self, grid):
        """Inferenza su video o webcam."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return

        stop_button = st.sidebar.button("Stop Inferenza")
        frame_holders = [col.empty() for row in grid for col in row]

        # Assicuriamoci che frameSize sia sempre una tupla valida
        if self.params["output_resolution"]:
            frame_size = self.params["output_resolution"]
        else:
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        writers = {
            model_name: cv2.VideoWriter(
                os.path.join(self._create_output_dir(model_name, "videos"), "output.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                frame_size  # Ora è sempre una tupla valida (larghezza, altezza)
            ) if self.params["save_output"] and self.params["save_video"] else None
            for model_name in self.models.keys()
        }

        frame_counter = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success or stop_button:
                break

            frame_counter += 1
            original_resolution = (frame.shape[1], frame.shape[0])

            for i, (model_name, model) in enumerate(self.models.items()):
                model.to(self.params["device"])
                results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
                annotated_frame = results[0].plot()
                frame_holders[i].image(annotated_frame, channels="BGR")

                if self.params["save_output"] and frame_counter % self.params["frame_skip"] == 0:
                    output_dir = self._create_output_dir(model_name, self.params["images_folder"])
                    frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                    if self.params["output_resolution"]:
                        frame_to_save = cv2.resize(frame_to_save, self.params["output_resolution"])
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_counter}.jpg"), frame_to_save)

                    if self.params["save_labels"]:
                        self._save_yolo_labels(output_dir, model_name, frame_counter, results, original_resolution, self.params["output_resolution"])

                    if self.params["save_video"] and writers[model_name]:
                        writers[model_name].write(annotated_frame)

        cap.release()
        for writer in writers.values():
            if writer:
                writer.release()
        cv2.destroyAllWindows()
