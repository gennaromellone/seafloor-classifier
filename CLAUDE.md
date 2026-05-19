# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progetto

Classificatore di **Posidonia Oceanica** da video subacquei acquisiti dal drone marino **Argo** (USV, Università di Napoli Parthenope). Il sistema analizza video frame per frame, divide ogni frame in 4 quadranti (2×2), classifica ciascuno con una CNN MobileNet e produce un video annotato + CSV georeferenziabile.

Il contesto scientifico è il monitoraggio di habitat bentonici nell'Area Marina Protetta di Punta Licosa (Cilento), con l'obiettivo di mappare la distribuzione della Posidonia Oceanica integrando AI con survey geoacustici.

## Ambiente

Il venv attivo è **locale alla cartella `classifier/`**:

```bash
source venv/bin/activate
```

Python 3.13, TensorFlow 2.21.0, Keras 3.14.1, tf-keras 2.21.0, OpenCV, Apple Silicon.

> I modelli `TM1.h5` / `TM2` sono stati salvati con Keras 2.x. Per caricarli con Keras 3
> lo script usa `import tf_keras as tf` (shim di compatibilità). **Non cambiare questo import**
> con `tensorflow.keras` — Keras 3 non ricostruisce correttamente l'architettura Sequential+MobileNet.

## Comandi principali

```bash
# Inferenza su video (dalla cartella classifier/)
python classify_posidonia.py -i video/GOPR0701.MP4

# Con soglia e sampling personalizzati
python classify_posidonia.py -i video/GOPR0701.MP4 -t 0.85 -dt 30

# Visualizzazione in tempo reale (premi q per uscire)
python classify_posidonia.py -i video/GOPR0701.MP4 --show

# Training di un nuovo modello
python train_posidonia.py
```

## Architettura

### Inferenza — `classify_posidonia.py`

Pipeline per frame:
1. Legge ogni N-esimo frame (`-dt`, default 10) dal video con `cv2.VideoCapture`
2. Divide il frame in 4 quadranti (2×2)
3. Ogni quadrante viene ridimensionato a **224×224**, normalizzato in [0,1] e passato al modello
4. Il modello restituisce `[conf_posidonia, conf_pianoro]` — index 0 è sempre Posidonia
5. Soglia (`-t`, default 0.80) applicata a `conf_posidonia` → True/False per quadrante
6. `frame_posidonia = True` se almeno un quadrante supera la soglia

Output scritti in `output/`:
- `<video>_<model>.mp4` — frame annotati con etichetta e confidenza per quadrante (verde = Posidonia, rosso = Pianoro)
- `output_<video>_<model>.txt` — CSV con header `timestamp;[lat;lon;]Q1_conf;Q1_pos;...;frame_posidonia`
  - Le colonne `lat;lon` sono presenti solo se la georeferenziazione è attiva (vedi `config.py`)

### Training — `train_posidonia.py`

Transfer learning su **MobileNet** (imagenet, frozen) con testa `GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(2, softmax)`. Dataset letto da `dataset/` con struttura per sottocartelle (una per classe). Salva il best checkpoint in `models/TM_new.h5`.

Costanti configurabili in testa al file: `IMG_W/H`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `VAL_SPLIT`, `DATASET_DIR`, `OUTPUT_MODEL`.

### Modello attivo

`models/TM1.h5` — esportato da **Google Teachable Machine** (MobileNet backbone), addestrato su Dataset5 (567 immagini: 339 Posidonia + 228 Seafloor), accuracy 0.98. Classi: `0 = posidonia`, `1 = pianoro` (vedi `labels.txt`).

> **Preprocessing obbligatorio per TM1/TM2**: Teachable Machine usa `(pixel / 127.5) - 1.0` → range `[-1, 1]`.
> **Non usare** `/ 255.0` (range `[0, 1]`): il modello produce output casuali con normalizzazione errata.

### Dataset

```
dataset/
├── posidonia/   336 immagini JPG (frame GoPro, campagna Punta Licosa giu 2022)
└── pianoro/     205 immagini JPG (fondale senza Posidonia)
```

Sbilanciamento 1.6:1 — tenere presente se si riaddestrа il modello; considerare class weights o oversampling della classe minoritaria.

## Soglia

La soglia ottimale dipende dall'obiettivo:
- **0.80** (default) — buon equilibrio tra precisione e recall
- **0.85–0.95** — maggiore precisione, più falsi negativi; utile per mapping conservativo
- Dall'esperimento sul campo (poster NEPTUNE 2022): TM1 a soglia 0.95 → 15/20 campioni corretti

## Georeferenziazione GPS

Il join GPS↔frame è gestito da `config.py` + tre funzioni in `classify_posidonia.py`.

**Per abilitarlo**, impostare in `config.py`:
```python
GPS_CSV = "../Posidonia/Punta-Licosa-2022-06-14.csv"
```

**Meccanismo:**
1. `load_gps()` carica il CSV GPS (formato `$GGLV`, sep `;`) in una lista ordinata `(secondi_dalla_mezzanotte, lat, lon)`
2. `get_video_start_utc()` legge il tag `creation_time` dal video con `ffprobe` e applica l'offset di fuso
3. `lookup_gps()` fa ricerca binaria (`bisect`) per trovare la riga GPS più vicina al timestamp assoluto di ogni frame

**Calibrazione fuso orario (campagna Punta Licosa 2022):**
- La Yi Camera salva l'ora locale CEST (UTC+2) come se fosse UTC nel tag MP4
- `CAMERA_TZ_OFFSET_HOURS = -2` corregge questo: `creation_time - 2h` → UTC reale
- Scarto residuo osservato vs GPS log: **+2 secondi** (creazione header file)
- A velocità drone ~1 m/s → errore posizione stimato **≤ 3 metri**

**Parametri configurabili in `config.py`:**
| Parametro | Default | Descrizione |
|---|---|---|
| `GPS_CSV` | `None` | Path CSV GPS; `None` disabilita la georeferenziazione |
| `CAMERA_TZ_OFFSET_HOURS` | `-2` | Correzione fuso camera → UTC |
| `MAX_GPS_GAP_SECS` | `5` | Gap massimo accettabile; oltre questa soglia lat/lon vengono lasciati vuoti |

## Note operative

- Gli script vanno eseguiti **dalla cartella `classifier/`** — i path sono relativi (`models/`, `output/`, `dataset/`)
- I video da analizzare vanno copiati in `video/` (non inclusi nel repository per dimensione)
- I video GoPro della campagna sono in `../Posidonia/goproLicosa/` (GOPR0700–GOPR0710.MP4, ~784 MB totali)
- `ffprobe` deve essere installato e disponibile nel PATH (parte di `ffmpeg`)
