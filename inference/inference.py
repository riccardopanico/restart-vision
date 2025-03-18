import os
import cv2
import streamlit as st
import yaml
from ultralytics import YOLO

class InferenceEngine:
    """Classe per eseguire inferenza YOLOv8 separata dalla UI."""

    def __init__(self, models, source_type, source, session_id, static_class_id=None, **params):
        self.models = models
        self.source_type = source_type
        self.source = source
        self.session_id = session_id if session_id is not None else "default_session"
        self.static_class_id = static_class_id  # ✅ ID fisso scelto dall'utente
        self.params = params
        self.detected_classes = set()  # Traccia le classi rilevate

    def run(self, num_columns):
        """Esegue inferenza e visualizza i risultati."""
        grid = [st.columns(num_columns) for _ in range((len(self.models) + num_columns - 1) // num_columns)]
        self._process_frames(grid)

    def _create_output_dir(self, model_name):
        """Crea la struttura di cartelle per il modello e la sessione."""
        base_output_dir = os.path.join("output", model_name, self.session_id)
        os.makedirs(base_output_dir, exist_ok=True)

        # Creazione delle cartelle train, val, test dentro images e labels
        images_dir = os.path.join(base_output_dir, "images")
        labels_dir = os.path.join(base_output_dir, "labels")
        crops_dir = os.path.join(base_output_dir, "crops")  # ✅ Cartella per i crop

        for subfolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(images_dir, subfolder), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, subfolder), exist_ok=True)

        os.makedirs(crops_dir, exist_ok=True)  # ✅ Creazione cartella crops

        return {
            "base": base_output_dir,
            "images": images_dir,
            "labels": labels_dir,
            "crops": crops_dir
        }

    def _process_frames(self, grid):
        """Inferenza su video/webcam con gestione della risoluzione e pulsante STOP."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return

        stop_button = st.sidebar.button("⏹️ Stop Inferenza")  # ✅ Pulsante di STOP
        frame_holders = [col.empty() for row in grid for col in row]
        output_dirs = {model_name: self._create_output_dir(model_name) for model_name in self.models.keys()}
        frame_counter = 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success or stop_button:  # ✅ Interrompiamo il ciclo se viene premuto STOP
                    break

                frame_counter += 1
                if self.params["output_resolution"]:
                    frame = cv2.resize(frame, self.params["output_resolution"])

                for i, (model_name, model) in enumerate(self.models.items()):
                    model.to(self.params["device"])
                    results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
                    annotated_frame = results[0].plot()
                    frame_holders[i].image(annotated_frame, channels="BGR")

                    # ✅ Salvataggio immagini e labels
                    self._save_frame_and_labels(frame, annotated_frame, output_dirs[model_name], model_name, frame_counter, results)

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _process_single_frame(self, frame, grid, frame_holders, output_dirs, frame_counter):
        """Elabora un singolo frame per tutti i modelli selezionati e gestisce il salvataggio."""
        if self.params["output_resolution"]:
            frame = cv2.resize(frame, self.params["output_resolution"])

        for i, (model_name, model) in enumerate(self.models.items()):
            model.to(self.params["device"])
            results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
            annotated_frame = results[0].plot()
            
            if frame_holders:
                frame_holders[i].image(annotated_frame, channels="BGR")

            if self.params["save_output"]:
                self._save_frame_and_labels(frame, annotated_frame, output_dirs[model_name], model_name, frame_counter, results)

    def _save_frame_and_labels(self, frame, annotated_frame, output_dir, model_name, frame_counter, results):
        """Salva il frame e le etichette YOLO se attivate nelle impostazioni."""
        frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
        image_path = os.path.join(output_dir["images"], "train", f"frame_{frame_counter}.jpg")
        cv2.imwrite(image_path, frame_to_save)

        if self.params["save_labels"]:
            self._save_yolo_labels(os.path.join(output_dir["labels"], "train"), model_name, frame_counter, results)

        if self.params.get("save_crop_boxes", False):
            self._save_cropped_boxes(frame, output_dir["crops"], frame_counter, results)

    def _save_cropped_boxes(self, frame, crops_dir, frame_counter, results):
        """Salva i crop dei bounding box rilevati."""
        os.makedirs(crops_dir, exist_ok=True)

        for idx, box in enumerate(results[0].boxes):
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                crop_path = os.path.join(crops_dir, f"crop_{frame_counter}_{idx}.jpg")
                cv2.imwrite(crop_path, crop)
            except ValueError:
                print(f"⚠️ ERRORE: Il bounding box per il frame {frame_counter} ha un numero di valori errato.")
                
    def _save_yolo_labels(self, labels_dir, model_name, frame_counter, results):
        """Salva le etichette YOLO in un file di testo nel formato corretto."""
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")

        print(f"🔍 DEBUG: Tentativo di salvataggio labels in {label_file} per modello {model_name}")

        # Se non ci sono detections, non salvare nulla
        if not results[0].boxes or len(results[0].boxes) == 0:
            print(f"⚠️ ATTENZIONE: Nessuna detections per {model_name} nel frame {frame_counter}.")
            return

        print(f"✅ DEBUG: {len(results[0].boxes)} oggetti rilevati per {model_name}")

        with open(label_file, "w") as f:
            for box in results[0].boxes:
                if len(box.xywhn.tolist()[0]) < 4:  # Controlla che ci siano abbastanza valori
                    print(f"❌ ERRORE: Bounding box con valori insufficienti nel frame {frame_counter}.")
                    continue

                original_class_id = int(box.cls)

                # ✅ Imposta sempre la classe selezionata nel radio button
                new_class_id = self.static_class_id if self.static_class_id is not None else original_class_id

                x_center, y_center, width, height = box.xywhn.tolist()[0]

                print(f"✅ DEBUG: {model_name} → Classe originale {original_class_id}, nuova classe {new_class_id} (scelta dall'utente), BBox: ({x_center}, {y_center}, {width}, {height})") 

                # ✅ Scriviamo nel file `.txt`
                f.write(f"{new_class_id} {x_center} {y_center} {width} {height}\n")

        print(f"✅ DEBUG: File labels salvato correttamente in {label_file} per {model_name}")
