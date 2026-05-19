# Posidonia Oceanica Classifier

Frame-by-frame classifier for underwater video from the marine drone **Argo** (USV, Università di Napoli Parthenope). Detects the presence of *Posidonia oceanica* seagrass using a MobileNet CNN, with optional GPS georeferencing.

## Requirements

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install tf-keras  # required for loading Keras 2.x models (TM1.h5)
```

`ffprobe` (part of ffmpeg) must be available in PATH for GPS georeferencing.

### Dataset
To download the dataset:
```bash
wget https://drive.google.com/file/d/1lmzykjJpDG5IwRnIM3zqojRLCizJ7yPY/view?usp=sharing -O dataset.zip
unzip dataset.zip
```

## Usage

```bash
# Basic inference
python classify_posidonia.py -i video/GOPR0701.MP4

# Custom threshold and frame sampling
python classify_posidonia.py -i video/GOPR0701.MP4 -t 0.85 -dt 30

# Show video during processing
python classify_posidonia.py -i video/GOPR0701.MP4 --show
```

| Argument | Default | Description |
|---|---|---|
| `-i` | required | Input video path |
| `-m` | `models/TM1.h5` | Model path |
| `-t` | `0.80` | Confidence threshold (0.0–1.0) |
| `-dt` | `10` | Process 1 frame every N |
| `--show` | off | Display video in real time |

## Output

Results are written to `output/`:
- `<video>_<model>.mp4` — annotated video (green = Posidonia, red = seafloor)
- `output_<video>_<model>.txt` — CSV with per-frame classification and optional GPS coordinates

## GPS Georeferencing

Edit `config.py` to enable:

```python
GPS_CSV = "../Posidonia/Punta-Licosa-2022-06-14.csv"
```

When active, each CSV row includes `lat;lon` columns derived by matching the frame timestamp to the nearest GPS fix. The camera clock offset (`CAMERA_TZ_OFFSET_HOURS`) and maximum allowed GPS gap (`MAX_GPS_GAP_SECS`) are also configurable in `config.py`.

## Model

`models/TM1.h5` is a MobileNet exported from Google Teachable Machine, trained on Dataset5 (567 images, Punta Licosa campaign June 2022). Input normalization: `(pixel / 127.5) - 1.0`.

Each frame is split into 4 quadrants (2×2); a frame is classified as Posidonia if at least one quadrant exceeds the threshold.

## Training

```bash
python train_posidonia.py
```

Place images under `dataset/posidonia/` and `dataset/pianoro/`. The best checkpoint is saved to `models/TM_new.h5`.
