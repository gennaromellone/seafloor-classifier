import bisect
import csv
import subprocess
import numpy as np
import os
import cv2
import tf_keras as tf  # compatibility shim: TM1/TM2 were saved with Keras 2.x
import datetime

from argparse import ArgumentParser

import config


# ---------------------------------------------------------------------------
# GPS helpers
# ---------------------------------------------------------------------------

def load_gps(csv_path):
    """Parse GPS log into a sorted list of (seconds_since_midnight, lat, lon)."""
    entries = []
    with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0 or len(row) < 17:
                continue
            time_str = row[1].strip()
            lat_str  = row[15].strip().replace(',', '.')
            lon_str  = row[16].strip().replace(',', '.')
            if not time_str or not lat_str or not lon_str:
                continue
            try:
                h, m, s = map(int, time_str.split(':'))
                entries.append((h * 3600 + m * 60 + s, float(lat_str), float(lon_str)))
            except (ValueError, IndexError):
                continue
    entries.sort(key=lambda x: x[0])
    return entries


def get_video_start_utc(video_path, tz_offset_hours):
    """Read creation_time from video metadata and apply timezone correction."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet',
         '-show_entries', 'format_tags=creation_time',
         '-of', 'default=noprint_wrappers=1:nokey=1',
         video_path],
        capture_output=True, text=True
    )
    raw = result.stdout.strip().rstrip('Z').split('.')[0]
    if not raw:
        return None
    try:
        dt = datetime.datetime.strptime(raw, '%Y-%m-%dT%H:%M:%S')
        return dt + datetime.timedelta(hours=tz_offset_hours)
    except ValueError:
        return None


def lookup_gps(gps_entries, abs_dt, max_gap_secs):
    """Return (lat, lon) for the GPS entry nearest to abs_dt, or (None, None)."""
    target = abs_dt.hour * 3600 + abs_dt.minute * 60 + abs_dt.second
    times = [e[0] for e in gps_entries]
    idx = bisect.bisect_left(times, target)
    best = None
    for i in (idx - 1, idx):
        if 0 <= i < len(gps_entries):
            gap = abs(gps_entries[i][0] - target)
            if best is None or gap < best[0]:
                best = (gap, i)
    if best is None or best[0] > max_gap_secs:
        return None, None
    _, lat, lon = gps_entries[best[1]]
    return lat, lon

IMG_W = 224
IMG_H = 224


def normalizeFrame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (IMG_W, IMG_H))
    frame = frame.astype(np.float32)
    frame = (frame / 127.5) - 1.0  # Teachable Machine MobileNet: range [-1, 1]
    frame = np.expand_dims(frame, axis=0)
    return frame


def label_prediction(ris, threshold):
    """Returns (confidence_float, is_posidonia).
    Model output: index 0 = posidonia, index 1 = pianoro.
    """
    pos_conf = float(ris[0][0])
    return round(pos_conf, 4), pos_conf >= threshold


def main(args):
    threshold = float(args.threshold)
    delta = int(args.deltaTime)

    print(f"Loading model: {args.classModel}")
    print(f"Threshold: {threshold:.2f} | Delta frames: {delta}")
    model = tf.models.load_model(args.classModel)

    stream = cv2.VideoCapture(args.input)
    FPS = stream.get(cv2.CAP_PROP_FPS) or 30.0

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    model_name = os.path.splitext(os.path.basename(args.classModel))[0]
    out_stem = f"{video_name}_{model_name}"

    # --- GPS setup ---
    gps_entries = []
    video_start_utc = None
    if config.GPS_CSV:
        gps_entries = load_gps(config.GPS_CSV)
        video_start_utc = get_video_start_utc(args.input, config.CAMERA_TZ_OFFSET_HOURS)
        if video_start_utc:
            print(f"GPS: {len(gps_entries)} entries | video start UTC: {video_start_utc}")
        else:
            print("GPS CSV caricato ma creation_time non leggibile — georeferenziazione disabilitata")
    use_gps = bool(gps_entries and video_start_utc)

    os.makedirs("output", exist_ok=True)
    out_txt = open(f"output/output_{out_stem}.txt", "w")
    gps_header = "lat;lon;" if use_gps else ""
    out_txt.write(f"timestamp;{gps_header}Q1_conf;Q1_pos;Q2_conf;Q2_pos;Q3_conf;Q3_pos;Q4_conf;Q4_pos;frame_posidonia\n")

    writer = None
    W = H = None
    frame_idx = 0

    while True:
        grabbed, frame = stream.read()
        if not grabbed:
            break
        frame_idx += 1
        if frame_idx % delta != 0:
            continue

        if W is None:
            H, W = frame.shape[:2]

        output = frame.copy()
        output = cv2.line(output, (0, H // 2), (W, H // 2), (0, 0, 0), 2)
        output = cv2.line(output, (W // 2, 0), (W // 2, H), (0, 0, 0), 2)

        quadrants = [
            (frame[0:H//2, 0:W//2],     (35,         50)),        # alto sx
            (frame[0:H//2, W//2:W],     (W//2 + 35,  50)),        # alto dx
            (frame[H//2:H, 0:W//2],     (35,         H//2 + 50)), # basso sx
            (frame[H//2:H, W//2:W],     (W//2 + 35,  H//2 + 50)), # basso dx
        ]

        results = []
        for crop, text_pos in quadrants:
            ris = model.predict(normalizeFrame(crop), verbose=0)
            conf_str, is_pos = label_prediction(ris, threshold)
            color = (0, 200, 0) if is_pos else (0, 0, 200)
            label = "Posidonia" if is_pos else "Pianoro"
            cv2.putText(output, f"{label} {conf_str}", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            results.append((conf_str, is_pos))

        frame_posidonia = any(r[1] for r in results)

        if writer is None:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            out_fps = max(1, FPS / delta)
            writer = cv2.VideoWriter(f"output/{out_stem}.mp4", fourcc, out_fps, (W, H))

        writer.write(output)

        frame_secs = int(stream.get(cv2.CAP_PROP_POS_FRAMES) / FPS)
        timestamp = str(datetime.timedelta(seconds=frame_secs))

        gps_str = ""
        if use_gps:
            abs_dt = video_start_utc + datetime.timedelta(seconds=frame_secs)
            lat, lon = lookup_gps(gps_entries, abs_dt, config.MAX_GPS_GAP_SECS)
            gps_str = f"{lat:.6f};{lon:.6f};" if lat is not None else ";"

        row = f"{timestamp};{gps_str}" + ";".join(f"{r[0]};{int(r[1])}" for r in results) + f";{int(frame_posidonia)}\n"
        out_txt.write(row)

        if args.showVideo:
            cv2.imshow("Posidonia Classifier", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    out_txt.close()
    if writer:
        writer.release()
    stream.release()
    print(f"Output: output/{out_stem}.mp4  |  output/output_{out_stem}.txt")


if __name__ == "__main__":
    parser = ArgumentParser(description="Posidonia Oceanica classifier — video analysis")
    parser.add_argument("-i", "--input", required=True, dest="input",
                        help="Input video path", metavar="FILE")
    parser.add_argument("-m", "--model", default="models/TM1.h5",
                        dest="classModel",
                        help="Model path (default: models/TM1.h5)")
    parser.add_argument("-t", "--threshold", default=0.80,
                        dest="threshold",
                        help="Posidonia confidence threshold 0.0–1.0 (default: 0.80)")
    parser.add_argument("-dt", "--delta", default=10, type=int,
                        dest="deltaTime",
                        help="Process 1 frame every N (default: 10)")
    parser.add_argument("--show", default=False, action="store_true",
                        dest="showVideo",
                        help="Show video during processing")
    main(parser.parse_args())
