# YOLOv8 Streamlit Inference

🚀 **YOLOv8 Streamlit Inference** è un'applicazione basata su **Streamlit** che permette di eseguire inferenza su immagini, video e webcam utilizzando modelli **YOLOv8**.

## 📂 Struttura del Progetto

```
📂 restart-vision
│── app.py                     # Avvio dell'applicazione Streamlit
│── 📂 views                    # Gestisce l'interfaccia utente
│   │── interface.py            # UI principale di Streamlit
│   │── model_selector.py       # Selezione del modello YOLO
│   │── inference_params.py     # Configurazione parametri di inferenza
│   │── output_params.py        # Selezione della sorgente (video/webcam/immagine)
│── 📂 inference                # Contiene la logica di inferenza YOLO
│   │── __init__.py             # Rende la cartella un modulo Python
│   │── inference.py            # Classe `InferenceEngine` per l'inferenza YOLO
│── 📂 models                   # Cartella per i modelli YOLO (.pt)
│── 📂 output                   # Cartella per salvare output (video, labels, immagini)
│── 📂 temp                     # File caricati temporaneamente da Streamlit
│── README.md                   # Documentazione del progetto
```

---

## 🚀 **Installazione**

### 1. **Clona il repository**
```bash
git clone https://github.com/tuo-utente/restart-vision.git
cd restart-vision
```

### 2. **Crea e attiva l'ambiente virtuale**
```bash
python3 -m venv restart_venv
source restart_venv/bin/activate  # Mac/Linux
restart_venv\Scripts\activate      # Windows
```

### 3. **Installa le dipendenze**
```bash
pip install -r requirements.txt
```

### 4. **Avvia l'applicazione**
```bash
streamlit run app.py
```

---

## 🔧 **Configurazione**
L'applicazione permette di:
- **Selezionare modelli YOLO** (.pt) dalla cartella `/models`
- **Caricare immagini o video** per l'inferenza
- **Eseguire inferenza in tempo reale da webcam**
- **Regolare i parametri di inferenza**, tra cui:
  - Confidence Threshold
  - IoU Threshold
  - Risoluzione Output
  - Frequenza di inferenza sui frame (`frame_skip`)
  - Salvataggio di video, frame e labels YOLO

---

## 🎨 **Come usare l'interfaccia**
1. **Carica un modello YOLO**  
   - Scegli un file `.pt` dalla cartella `models/`
  
2. **Seleziona la sorgente**  
   - **Webcam**
   - **File video** (`.mp4`, `.avi`, ecc.)
   - **Immagine** (`.jpg`, `.png`, ecc.)

3. **Configura i parametri**  
   - Confidence Threshold
   - IoU Threshold
   - Inferenza ogni N frame (per video)
   - Salvataggio video, immagini e labels

4. **Avvia l'inferenza**  
   - Premi **"Avvia Inferenza"** e visualizza i risultati in tempo reale!

---

## 🛠 **Output**
L'output dell'inferenza viene salvato nella cartella `/output/` con la seguente struttura:
```
📂 output/
│── 📂 nome-modello/
│    │── 📂 sessione_YYYY-MM-DD_HH-MM-SS/
│    │    │── 📂 images/          # Frames salvati
│    │    │── 📂 labels/          # File YOLO .txt
│    │    │── 📂 videos/          # Video output
```

---

## 🔧 **Tecnologie Utilizzate**
- **Python 3.10**
- **Ultralytics YOLOv8** → Modelli di rilevamento
- **Streamlit** → Interfaccia Web
- **OpenCV** → Gestione immagini e video
- **Torch** → Backend YOLO

---

## 📈 **TODO / Miglioramenti Futuri**
✅ **Gestione multi-thread per inferenza video**  
✅ **Migliorare interfaccia utente (es. visualizzazione bounding box in overlay)**  
✅ **Aggiungere supporto per più modelli YOLO contemporaneamente**  

---

## 📞 **Supporto**
Hai bisogno di aiuto? Apri un **Issue** su GitHub o contattami via email.  
Se il progetto ti è stato utile, lascia una **Star ⭐ su GitHub!**

