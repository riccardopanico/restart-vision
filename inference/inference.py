import os
import cv2
import streamlit as st
import yaml
from ultralytics import YOLO

class InferenceEngine:
    def __init__(self, models, source_type, source, session_id, static_class_id=None, **params):
        self.models = models
        self.source_type = source_type
        self.source = source
        self.session_id = session_id if session_id else "default_session"
        self.static_class_id = static_class_id
        self.params = params
        self.detected_classes = set()

    def run(self, num_columns):
        grid = [st.columns(num_columns) for _ in range((len(self.models) + num_columns - 1) // num_columns)]
        self._process_frames(grid)

    def _create_output_dir(self, model_name):
        base_output_dir = os.path.join("output", model_name, self.session_id)
        os.makedirs(base_output_dir, exist_ok=True)

        images_dir = os.path.join(base_output_dir, "images")
        labels_dir = os.path.join(base_output_dir, "labels")
        crops_dir = os.path.join(base_output_dir, "crops")
        video_dir = os.path.join(base_output_dir, "videos")

        for subfolder in ["train", "val", "test"]:
            os.makedirs(os.path.join(images_dir, subfolder), exist_ok=True)
            os.makedirs(os.path.join(labels_dir, subfolder), exist_ok=True)

        os.makedirs(crops_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        return {"base": base_output_dir, "images": images_dir, "labels": labels_dir, "crops": crops_dir, "videos": video_dir}

    def _process_frames(self, grid):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            st.error("Errore nell'apertura del video o webcam.")
            return

        stop_button = st.sidebar.button("⏹️ Stop Inferenza")
        frame_holders = [col.empty() for row in grid for col in row]
        output_dirs = {model_name: self._create_output_dir(model_name) for model_name in self.models.keys()}
        frame_counter = 0
        video_writers = {}

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        output_resolution = self.params["output_resolution"] or (width, height)

        if self.params["save_video"]:
            for model_name in self.models.keys():
                video_path = os.path.join(output_dirs[model_name]["videos"], f"inference_{self.session_id}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writers[model_name] = cv2.VideoWriter(video_path, fourcc, fps, output_resolution)

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success or stop_button:
                    break

                frame_counter += 1
                frame = cv2.resize(frame, output_resolution) if output_resolution else frame

                for i, (model_name, model) in enumerate(self.models.items()):
                    model.to(self.params["device"])
                    results = model(frame, conf=self.params["confidence"], iou=self.params["iou_threshold"])
                    annotated_frame = results[0].plot()
                    frame_holders[i].image(annotated_frame, channels="BGR")

                    if self.params["save_output"]:
                        self._save_frame_and_labels(frame, annotated_frame, output_dirs[model_name], model_name, frame_counter, results)
                        if self.params["save_video"]:
                            video_writers[model_name].write(annotated_frame)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            for writer in video_writers.values():
                writer.release()

    def _save_frame_and_labels(self, frame, annotated_frame, output_dir, model_name, frame_counter, results):
        frame_to_save = annotated_frame if self.params["save_annotated_frames"] else frame
        image_path = os.path.join(output_dir["images"], "train", f"frame_{frame_counter}.jpg")
        cv2.imwrite(image_path, frame_to_save)

        if self.params["save_labels"]:
            self._save_yolo_labels(os.path.join(output_dir["labels"], "train"), model_name, frame_counter, results)

        if self.params.get("save_crop_boxes", False):
            self._save_cropped_boxes(frame, output_dir["crops"], frame_counter, results)

    def _save_cropped_boxes(self, frame, crops_dir, frame_counter, results):
        os.makedirs(crops_dir, exist_ok=True)
        for idx, box in enumerate(results[0].boxes):
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                crop_path = os.path.join(crops_dir, f"crop_{frame_counter}_{idx}.jpg")
                cv2.imwrite(crop_path, crop)
            except ValueError:
                print(f"⚠️ ERRORE: Bounding box errato nel frame {frame_counter}.")

    def _save_yolo_labels(self, labels_dir, model_name, frame_counter, results):
        os.makedirs(labels_dir, exist_ok=True)
        label_file = os.path.join(labels_dir, f"frame_{frame_counter}.txt")

        if not results[0].boxes or len(results[0].boxes) == 0:
            return

        with open(label_file, "w") as f:
            for box in results[0].boxes:
                if len(box.xywhn.tolist()[0]) < 4:
                    continue

                original_class_id = int(box.cls)
                new_class_id = self.static_class_id if self.static_class_id is not None else original_class_id
                x_center, y_center, width, height = box.xywhn.tolist()[0]

                f.write(f"{new_class_id} {x_center} {y_center} {width} {height}\n")
