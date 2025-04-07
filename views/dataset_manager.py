import os
import shutil
import yaml
import time
import streamlit as st
from datetime import datetime

DATASETS_DIR = "datasets"
OUTPUT_DIR = "output"
os.makedirs(DATASETS_DIR, exist_ok=True)

def list_datasets():
    datasets = []
    if not os.path.exists(OUTPUT_DIR):
        return datasets

    for model_name in os.listdir(OUTPUT_DIR):
        model_path = os.path.join(OUTPUT_DIR, model_name)
        if os.path.isdir(model_path):
            for session in os.listdir(model_path):
                session_path = os.path.join(model_path, session)
                if os.path.isdir(session_path):
                    for split in ["train", "val", "test"]:
                        images_path = os.path.join(session_path, split, "images")
                        labels_path = os.path.join(session_path, split, "labels")
                        if os.path.exists(images_path) and os.path.exists(labels_path):
                            datasets.append({
                                "modello": model_name,
                                "data": session,
                                "path": session_path,
                                "split": split
                            })
    return datasets

def merge_datasets(selected_datasets, target_name="merged_dataset", class_list=[]):
    target_path = os.path.abspath(os.path.join(DATASETS_DIR, target_name))
    os.makedirs(target_path, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_path, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_path, split, "labels"), exist_ok=True)

    for dataset in selected_datasets:
        model_name = dataset["modello"]
        session_name = dataset["data"]
        split = dataset["split"]
        dataset_path = os.path.abspath(dataset["path"])

        for data_type in ["images", "labels"]:
            source_folder = os.path.join(dataset_path, split, data_type)
            target_folder = os.path.join(target_path, split, data_type)

            if os.path.exists(source_folder) and os.listdir(source_folder):
                for file in os.listdir(source_folder):
                    source_file = os.path.join(source_folder, file)
                    file_ext = file.split(".")[-1]
                    file_base = file.replace(f".{file_ext}", "")
                    new_file_name = f"{model_name}_{session_name}_{file_base}.{file_ext}"
                    shutil.copy(source_file, os.path.join(target_folder, new_file_name))

    yaml_path = os.path.join(target_path, "data.yaml")
    data_yaml = {
        # "train": os.path.abspath(os.path.join(target_path, "train", "images")),
        # "val": os.path.abspath(os.path.join(target_path, "val", "images")),
        # "test": os.path.abspath(os.path.join(target_path, "test", "images")),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_list),
        "names": class_list
    }

    with open(yaml_path, "w") as file:
        yaml.dump(data_yaml, file, default_flow_style=False, allow_unicode=True)

    return target_path

def delete_dataset(dataset_path):
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path, ignore_errors=True)
        time.sleep(0.5)
        if not os.path.exists(dataset_path):
            return True
    return False

def dataset_management_ui():
    datasets = list_datasets()

    if not datasets:
        st.sidebar.warning("⚠️ Nessun dataset trovato in `output/`.")
        return

    if st.button("🔄 Aggiorna lista"):
        st.rerun()

    selected_datasets = []
    for dataset in datasets:
        col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])

        with col1:
            st.text(f"📂 {dataset['modello']}")
        with col2:
            st.text(f"📅 {dataset['data']}")
        with col3:
            st.text(f"🔹 {dataset['split']}")
        with col4:
            selected = st.checkbox("Seleziona", key=f"{dataset['modello']}_{dataset['data']}_{dataset['split']}")
            if selected:
                selected_datasets.append(dataset)
        with col5:
            if st.button("🗑️", key=f"delete_{dataset['modello']}_{dataset['data']}_{dataset['split']}"):
                if delete_dataset(dataset["path"]):
                    st.success(f"✅ Dataset `{dataset['modello']}` eliminato con successo.")
                    st.rerun()
                else:
                    st.error(f"❌ Errore nell'eliminazione di `{dataset['modello']}`.")

    if not selected_datasets:
        st.warning("⚠️ Seleziona almeno un dataset per procedere con il merge.")
        return

    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset_name = st.text_input("Nome per il dataset unito:", st.session_state.dataset_name, key="dataset_name_input")
    st.session_state.dataset_name = dataset_name

    class_list = st.session_state.get("class_input", "model_1\nmodel_2\nmodel_3").split("\n")
    class_list = [cls.strip() for cls in class_list if cls.strip()]

    if st.button("Conferma/Merge"):
        if dataset_name.strip():
            target_path = merge_datasets(selected_datasets, dataset_name, class_list=class_list)
            st.success(f"✅ Merge completato! Dataset salvato in `{target_path}`")
        else:
            st.warning("⚠️ Devi inserire un nome per il dataset unito!")