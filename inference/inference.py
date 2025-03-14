import os
import cv2
import streamlit as st
import yaml  # Per generare il file data.yaml
from ultralytics import YOLO

class InferenceEngine:
    """Classe per eseguire inferenza YOLOv8 separata dalla UI."""

    def __init__(self, models, source_type, source, session_id, static_class_id=None, **params):
        self.models = models
        self.source_type = source_type
        self.source = source
        self.session_id = session_id if session_id is not None else "default_session"
        self.static_class_id = static_class_id  # ✅ Memorizziamo l'ID fisso scelto dall'utente
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
        if self.session_id is None:
            self.session_id = "default_session"  # Usa un valore di fallback

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
        """Salva tutte le etichette YOLO in un file di testo, assegnando sempre la classe selezionata nel radio button."""
        label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")

        print(f"🔍 DEBUG: Tentativo di salvataggio labels in {label_file} per modello {model_name}")

        # Se non ci sono detections, non salvare nulla
        if not results[0].boxes or len(results[0].boxes) == 0:
            print(f"⚠️ ATTENZIONE: Nessuna detections per {model_name} nel frame {frame_counter}.")
            return

        print(f"✅ DEBUG: {len(results[0].boxes)} oggetti rilevati per {model_name}")

        with open(label_file, "w") as f:
            for box in results[0].boxes:
                original_class_id = int(box.cls)

                # ✅ Imposta sempre la classe selezionata dal radio button
                new_class_id = self.static_class_id  # <-- Usa il valore scelto dall'utente

                # ✅ Debug dettagliato
                x_center, y_center, width, height = box.xywhn.tolist()[0]
                print(f"✅ DEBUG: {model_name} → Classe originale {original_class_id}, assegnata nuova classe {new_class_id} (scelta dall'utente), BBox: ({x_center}, {y_center}, {width}, {height})") 

                # ✅ Scriviamo nel file `.txt`
                f.write(f"{new_class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ DEBUG: File labels salvato correttamente in {label_file} per {model_name}")

    def _generate_data_yaml(self, output_dir, static_class_name):
        """Genera il file data.yaml con un solo ID fisso per tutte le classi selezionate."""

        yaml_path = os.path.join(output_dir, "data.yaml")

        if not static_class_name:
            print("⚠️ DEBUG: Nessuna classe selezionata come ID fisso, il file data.yaml non verrà creato.")
            return

        data_yaml = {
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": 1,  # Un'unica classe di riferimento
            "names": [static_class_name]
        }

        # Scrittura del file data.yaml
        try:
            with open(yaml_path, "w") as file:
                yaml.dump(data_yaml, file, default_flow_style=False)
            
            print(f"✅ DEBUG: File data.yaml creato con classe fissa '{static_class_name}' in: {yaml_path}")

        except Exception as e:
            print(f"❌ ERRORE: Impossibile scrivere il file data.yaml! Errore: {e}")

    def _process_image(self, grid):
        """Inferenza su immagine per tutti i modelli selezionati, con gestione della risoluzione."""
        frame = cv2.imread(self.source)
        if frame is None:
            st.error("Errore nel caricamento dell'immagine.")
            return

        # ✅ Applichiamo la risoluzione selezionata se impostata
        if self.params["output_resolution"]:
            frame = cv2.resize(frame, self.params["output_resolution"])
            print(f"🔍 DEBUG: Immagine ridimensionata a {self.params['output_resolution']}")

        for model_name, model in self.models.items():
            print(f"🔍 DEBUG: Elaborazione immagine con modello {model_name}")

            output_dirs = self._create_output_dir(model_name)
            row_idx, col_idx = divmod(list(self.models.keys()).index(model_name), len(grid[0]))

            model.to(self.params["device"])
            results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])

            print(f"✅ DEBUG: {model_name} ha rilevato {len(results[0].boxes)} oggetti")

            annotated_frame = results[0].plot()

            if self.params["output_resolution"]:
                annotated_frame = cv2.resize(annotated_frame, self.params["output_resolution"])

            grid[row_idx][col_idx].image(annotated_frame, channels="BGR")

            if self.params["save_output"]:
                frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                image_path = os.path.join(output_dirs["images"], "train", "frame_0.jpg")
                cv2.imwrite(image_path, frame_to_save)
                print(f"✅ DEBUG: Immagine salvata in {image_path} per modello {model_name}")

                if self.params["save_labels"]:
                    self._save_yolo_labels(os.path.join(output_dirs["labels"], "train"), model_name, 0, results)

    def _process_video(self, grid):
        """Inferenza su video per tutti i modelli selezionati, con gestione della risoluzione."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return

        stop_button = st.sidebar.button("Stop Inferenza")
        frame_holders = [col.empty() for row in grid for col in row]

        output_dirs = {
            model_name: self._create_output_dir(model_name)
            for model_name in self.models.keys()
        }

        frame_counter = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success or stop_button:
                    break

                frame_counter += 1

                # ✅ Applichiamo la risoluzione selezionata se impostata
                if self.params["output_resolution"]:
                    frame = cv2.resize(frame, self.params["output_resolution"])
                    print(f"🔍 DEBUG: Frame {frame_counter} ridimensionato a {self.params['output_resolution']}")

                for i, (model_name, model) in enumerate(self.models.items()):
                    print(f"🔍 DEBUG: Elaborazione frame {frame_counter} con modello {model_name}")

                    model.to(self.params["device"])
                    results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])

                    print(f"✅ DEBUG: {model_name} ha rilevato {len(results[0].boxes)} oggetti")

                    annotated_frame = results[0].plot()
                    frame_holders[i].image(annotated_frame, channels="BGR")

                    # ✅ Ora salviamo le immagini dei frame processati, rispettando la risoluzione
                    if self.params["save_output"]:
                        frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
                        image_path = os.path.join(output_dirs[model_name]["images"], "train", f"frame_{frame_counter}.jpg")

                        try:
                            cv2.imwrite(image_path, frame_to_save)
                            print(f"✅ DEBUG: Immagine salvata in {image_path} per modello {model_name}")
                        except Exception as e:
                            print(f"❌ ERRORE: Impossibile salvare {image_path}. Errore: {e}")

                    # ✅ Ora salviamo le labels per ogni frame elaborato
                    if self.params["save_labels"]:
                        self._save_yolo_labels(os.path.join(output_dirs[model_name]["labels"], "train"), model_name, frame_counter, results)

        finally:
            cap.release()
            cv2.destroyAllWindows()
