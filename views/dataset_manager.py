import os
import streamlit as st
import shutil
import yaml
from datetime import datetime

datasets_dir = "datasets"
os.makedirs(datasets_dir, exist_ok=True)

def list_datasets(output_dir="output"):
    """Restituisce una lista di dataset trovati nella cartella output."""
    datasets = []

    if not os.path.exists(output_dir):
        return datasets

    for model_name in os.listdir(output_dir):
        model_path = os.path.join(output_dir, model_name)
        if os.path.isdir(model_path):
            for session in os.listdir(model_path):
                session_path = os.path.join(model_path, session)
                if os.path.isdir(session_path):
                    datasets.append({
                        "modello": model_name,
                        "data": session,
                        "path": session_path
                    })
    
    return datasets

def merge_datasets(selected_datasets, target_name="merged_dataset", datasets_dir="datasets", class_list=[]):
    """Unisce più dataset in un'unica cartella, rinomina i file e mantiene la struttura YOLO (train, val, test)."""

    # Creazione della cartella target con struttura YOLO
    target_path = os.path.join(datasets_dir, target_name)
    for folder in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        os.makedirs(os.path.join(target_path, folder), exist_ok=True)

    for dataset in selected_datasets:
        model_name = dataset["modello"]
        session_name = dataset["data"]
        dataset_path = dataset["path"]

        for data_type in ["images", "labels"]:  
            for split in ["train", "val", "test"]:  # Assicuriamoci che train, val, test siano gestiti
                source_folder = os.path.join(dataset_path, f"{data_type}/{split}")
                target_folder = os.path.join(target_path, f"{data_type}/{split}")

                print(f"🔍 DEBUG: Controllando {source_folder} -> {target_folder}")

                if os.path.exists(source_folder) and os.listdir(source_folder):  # Verifica che la cartella non sia vuota
                    print(f"📂 Trovati file in {source_folder}, procedo con la copia.")

                    for file in os.listdir(source_folder):
                        source_file = os.path.join(source_folder, file)
                        
                        # ✅ Creiamo un nome univoco per evitare sovrascritture
                        file_extension = file.split(".")[-1]
                        file_base_name = file.replace(f".{file_extension}", "")
                        new_file_name = f"{model_name}_{session_name}_{file_base_name}.{file_extension}"
                        target_file = os.path.join(target_folder, new_file_name)

                        try:
                            shutil.copy(source_file, target_file)
                            print(f"✅ Copiato {source_file} -> {target_file}")
                        except Exception as e:
                            print(f"❌ ERRORE: Non sono riuscito a copiare {source_file}. Errore: {e}")

                else:
                    print(f"⚠️ ATTENZIONE: Nessun file trovato in {source_folder} o cartella vuota.")

    # ✅ Creiamo il file `data.yaml`
    yaml_path = os.path.join(target_path, "data.yaml")
    data_yaml = {
        "train": os.path.join(target_path, "images/train"),
        "val": os.path.join(target_path, "images/val"),
        "test": os.path.join(target_path, "images/test"),
        "nc": len(class_list),
        "names": class_list
    }

    with open(yaml_path, "w") as file:
        yaml.dump(data_yaml, file, default_flow_style=False)

    print(f"✅ Merge completato in {target_path} con data.yaml generato!")
    return target_path

def delete_dataset(dataset_path):
    """Forza la rimozione completa di un dataset e della cartella, anche se rimane vuota o bloccata."""
    try:
        if os.path.exists(dataset_path):  # ✅ Verifica se la cartella esiste
            
            # ✅ Rimuoviamo tutto il contenuto della cartella
            shutil.rmtree(dataset_path, ignore_errors=True)
            time.sleep(0.5)  # 🔄 Attende un attimo per evitare blocchi

            # ✅ Controlliamo se la cartella è ancora presente
            if os.path.exists(dataset_path):
                print(f"⚠️ ATTENZIONE: La cartella {dataset_path} non si è eliminata subito. Forziamo la rimozione.")

                # ✅ Tentiamo di rimuovere eventuali file nascosti
                for root, dirs, files in os.walk(dataset_path, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            print(f"🗑️ DEBUG: Eliminato file residuo {file_path}")
                        except Exception as e:
                            print(f"❌ ERRORE: Non posso eliminare {file_path}. Errore: {e}")

                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                            print(f"🗑️ DEBUG: Eliminata cartella vuota {dir_path}")
                        except Exception as e:
                            print(f"❌ ERRORE: Non posso eliminare {dir_path}. Errore: {e}")

                # ✅ Ultimo tentativo di eliminazione della cartella
                try:
                    os.rmdir(dataset_path)
                    print(f"✅ DEBUG: Cartella {dataset_path} rimossa con successo dopo pulizia.")
                except Exception as e:
                    print(f"❌ ERRORE: La cartella {dataset_path} è ancora presente. Errore: {e}")
                    return False

            print(f"✅ DEBUG: Dataset eliminato {dataset_path}")
            return True
        else:
            print(f"⚠️ ATTENZIONE: Il dataset {dataset_path} non esiste già.")
            return False
    except Exception as e:
        print(f"❌ ERRORE: Impossibile eliminare {dataset_path}. Errore: {e}")
        return False

def dataset_management_ui():
    """Interfaccia Streamlit per la gestione dei dataset generati dall'inferenza."""
    st.sidebar.subheader("Gestione Dataset")

    # Recuperiamo la lista dei dataset
    datasets = list_datasets()

    if not datasets:
        st.sidebar.warning("⚠️ Nessun dataset trovato in `output/`.")
        return

    # Mostriamo i dataset in una tabella interattiva
    st.subheader("Seleziona i dataset da unire o eliminare")

    selected_datasets = []
    for dataset in datasets:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])  # Aggiunto un nuovo bottone per eliminare

        with col1:
            st.text(f"📂 {dataset['modello']}")
        with col2:
            st.text(f"📅 {dataset['data']}")
        with col3:
            selected = st.checkbox("Seleziona", key=f"{dataset['modello']}_{dataset['data']}")
            if selected:
                selected_datasets.append(dataset)
        with col4:
            delete_button = st.button("🗑️", key=f"delete_{dataset['modello']}_{dataset['data']}")
            if delete_button:
                if delete_dataset(dataset["path"]):
                    st.success(f"✅ Dataset `{dataset['modello']}` eliminato con successo.")
                    st.rerun()  # ✅ Sostituito st.experimental_rerun() con st.rerun()
                else:
                    st.error(f"❌ Errore nell'eliminazione di `{dataset['modello']}`.")

    # ✅ Impediamo il merge se nessun dataset è selezionato
    if not selected_datasets:
        st.warning("⚠️ Seleziona almeno un dataset per procedere con il merge.")
        return

    # ✅ Salviamo il nome del dataset nel session state per evitare reset
    if "dataset_name" not in st.session_state:
        st.session_state.dataset_name = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset_name = st.text_input(
        "Nome per il dataset unito:",
        st.session_state.dataset_name,
        key="dataset_name_input"
    )

    # ✅ Aggiorniamo il session_state quando l'utente cambia il nome
    st.session_state.dataset_name = dataset_name

    # ✅ Recuperiamo le classi dal session_state e facciamo pulizia
    class_list = st.session_state.get("class_input", "class_1\nclass_2\nclass_3").split("\n")
    class_list = [cls.strip() for cls in class_list if cls.strip()]  

    # ✅ Pulsante per eseguire il merge
    if st.button("Conferma/Merge"):
        if dataset_name.strip():
            target_path = merge_datasets(selected_datasets, dataset_name, class_list=class_list)
            st.success(f"✅ Merge completato! Dataset salvato in `{target_path}`")
        else:
            st.warning("⚠️ Devi inserire un nome per il dataset unito!")
