from datetime import datetime
from ultralytics import YOLO

# Parametri configurabili
model_path = 'best_236ep_v8.pt'  # Percorso del modello pre-addestrato
dataset_path = '/home/airestart/Scrivania/ultralytics/ds_gianel_2/data.yaml'  # Percorso del file YAML del dataset
epochs = 50
batch_size = 16
learning_rate = 0.01
momentum = 0.937
optimizer = 'Adam'  # Specificato per mantenere gli iperparametri
project_name = 'runs/train'
experiment_name = f"exp_gianel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # Nome esperimento con data e ora
resume_training = False
validate = False  # Disattiva la fase di validazione
single_class = False  # Disattiva la modalità single class

# Carica il modello pre-addestrato
model = YOLO(model_path)

# Avvia l'allenamento
model.train(
    data=dataset_path,
    epochs=epochs,
    batch=batch_size,
    lr0=learning_rate,
    momentum=momentum,
    optimizer=optimizer,
    project=project_name,
    name=experiment_name,
    resume=resume_training,
    val=validate,
    single_cls=single_class
)

print(f"🟢 Training avviato senza validazione su dataset: {dataset_path}")
