import os
import cv2
import streamlit as st
import yaml  # Per generare il file data.yaml
from ultralytics import YOLO

class InferenceEngine:
    """Classe per eseguire inferenza YOLOv8 separata dalla UI."""

    def __init__(self, models, source_type, source, session_id, **params):
        self.models = models
        self.source_type = source_type
        self.source = source
        self.session_id = session_id
        self.params = params
        self.detected_classes = set()  # Traccia le classi rilevate

    def run(self, num_columns):
        """Esegue inferenza e visualizza i risultati."""
        cap = None
        grid = [st.columns(num_columns) for _ in range((len(self.models) + num_columns - 1) // num_columns)]

        if self.source_type == "image":
            self._process_image(grid)
        elif self.source_type in ["video", "webcam"]:
            self._process_video(grid)

    def _create_output_dir(self, model_name):
        """Crea la struttura di cartelle per il modello e la sessione."""
        base_output_dir = os.path.join("output", model_name, self.session_id)
        os.makedirs(base_output_dir, exist_ok=True)

        # Creazione delle cartelle train, val, test dentro images e labels
        images_dir = os.path.join(base_output_dir, "images")
        labels_dir = os.path.join(base_output_dir, "labels")
        videos_dir = os.path.join(base_output_dir, "videos")

        for subfolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(images_dir, subfolder), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, subfolder), exist_ok=True)

        os.makedirs(videos_dir, exist_ok=True)

        return {
            "base": base_output_dir,
            "images": images_dir,
            "labels": labels_dir,
            "videos": videos_dir
        }

    def _save_yolo_labels(self, labels_dir, model_name, frame_counter, results):
        """Salva le etichette YOLO in un file di testo e aggiorna le classi rilevate."""
        label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")

        print(f"🔍 DEBUG: Salvataggio labels in {label_file}")

        with open(label_file, "w") as f:
            for box in results[0].boxes:
                class_id = int(box.cls)
                self.detected_classes.add(class_id)  # Registra la classe trovata

                print(f"✅ DEBUG: Classe rilevata e aggiunta: {class_id}")  # Nuovo debug

                x_center, y_center, width, height = box.xywhn.tolist()[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                
    def _generate_data_yaml(self, output_dir):
        """Genera il file data.yaml in base alle classi trovate."""

        # DEBUG: Stampa le classi rilevate
        print(f"🔍 DEBUG: Classi rilevate prima di creare data.yaml: {self.detected_classes}")

        # Forziamo la creazione della cartella se non esiste
        os.makedirs(output_dir, exist_ok=True)

        yaml_path = os.path.join(output_dir, "data.yaml")

        # Se non ci sono classi rilevate, inseriamo almeno una classe di default
        if not self.detected_classes:
            print("⚠️ DEBUG: Nessuna classe rilevata, aggiungo una classe di default.")
            self.detected_classes = {0}  # Forziamo almeno una classe

        data_yaml = {
            "train": "dataset/images/train",
            "val": "dataset/images/val",
            "test": "dataset/images/test",
            "nc": len(self.detected_classes),
            "names": [f"model_{cls}" for cls in sorted(self.detected_classes)]
        }

        # Scriviamo il file YAML
        try:
            with open(yaml_path, "w") as file:
                yaml.dump(data_yaml, file, default_flow_style=False)
            
            print(f"✅ DEBUG: File data.yaml salvato in: {yaml_path}")

        except Exception as e:
            print(f"❌ ERRORE: Impossibile scrivere il file data.yaml! Errore: {e}")

    def _process_image(self, grid):
        """Inferenza su immagine."""
        frame = cv2.imread(self.source)
        if frame is None:
            st.error("Errore nel caricamento dell'immagine.")
            return

        for model_name, model in self.models.items():
            output_dirs = self._create_output_dir(model_name)
            row_idx, col_idx = divmod(list(self.models.keys()).index(model_name), len(grid[0]))

            model.to(self.params["device"])
            results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
            annotated_frame = results[0].plot()

            if self.params["output_resolution"]:
                annotated_frame = cv2.resize(annotated_frame, self.params["output_resolution"])

            grid[row_idx][col_idx].image(annotated_frame, channels="BGR")

            if self.params["save_output"]:
                frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                cv2.imwrite(os.path.join(output_dirs["images"], "train", "frame_0.jpg"), frame_to_save)

                if self.params["save_labels"]:
                    self._save_yolo_labels(os.path.join(output_dirs["labels"], "train"), model_name, 0, results)

        if self.params["save_labels"]:
            for model_name in self.models.keys():
                output_dir = self._create_output_dir(model_name)["base"]
                self._generate_data_yaml(output_dir)

    def _process_video(self, grid):
        """Inferenza su video o webcam."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return

        stop_button = st.sidebar.button("Stop Inferenza")
        frame_holders = [col.empty() for row in grid for col in row]

        if self.params["output_resolution"]:
            frame_size = self.params["output_resolution"]
        else:
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        output_dirs = {
            model_name: self._create_output_dir(model_name)
            for model_name in self.models.keys()
        }

        writers = {
            model_name: cv2.VideoWriter(
                os.path.join(output_dirs[model_name]["videos"], "output.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                frame_size
            ) if self.params["save_output"] and self.params["save_video"] else None
            for model_name in self.models.keys()
        }

        frame_counter = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success or stop_button:
                    break

                frame_counter += 1

                for i, (model_name, model) in enumerate(self.models.items()):
                    model.to(self.params["device"])
                    results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
                    annotated_frame = results[0].plot()
                    frame_holders[i].image(annotated_frame, channels="BGR")

                    if self.params["save_output"] and frame_counter % self.params["frame_skip"] == 0:
                        output_dir = output_dirs[model_name]
                        frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                        if self.params["output_resolution"]:
                            frame_to_save = cv2.resize(frame_to_save, self.params["output_resolution"])
                        cv2.imwrite(os.path.join(output_dir["images"], "train", f"frame_{frame_counter}.jpg"), frame_to_save)

                        if self.params["save_labels"]:
                            self._save_yolo_labels(os.path.join(output_dir["labels"], "train"), model_name, frame_counter, results)

                        if self.params["save_video"] and writers[model_name]:
                            writers[model_name].write(annotated_frame)

            print("🔍 DEBUG: Uscita dal ciclo di inferenza.")

        finally:
            # Assicuriamoci di chiudere sempre tutto
            cap.release()
            for writer in writers.values():
                if writer:
                    writer.release()
            cv2.destroyAllWindows()

            # Forziamo la creazione del file data.yaml anche in caso di errore
            if self.params["save_labels"]:
                for model_name in self.models.keys():
                    output_dir = output_dirs[model_name]["base"]

                    print(f"🔍 DEBUG: Chiamata finale a _generate_data_yaml per {output_dir}")  
                    self._generate_data_yaml(output_dir)

            print("✅ DEBUG: Fine inferenza, dovrebbe essere stato creato data.yaml.")