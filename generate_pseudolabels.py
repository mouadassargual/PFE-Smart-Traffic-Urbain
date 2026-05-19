"""
generate_pseudolabels.py
Utilise bellevue_best.onnx pour annoter automatiquement
les vidéos Bellevue → dataset pour fine-tuning yolo26n
"""
import cv2
import numpy as np
import onnxruntime as ort
import os
import glob
from pathlib import Path

# Config
MODEL_PATH  = "models_rpi/custom/bellevue_best.onnx"
VIDEOS_DIR  = "data/videos/ne8th"
OUTPUT_DIR  = "data/pseudolabels"
CONF_THRESH = 0.50   # Seuil confiance pour garder annotation
MAX_FRAMES  = 500    # Nombre de frames à extraire
INTERVAL    = 30     # 1 frame toutes les 30 frames

# Classes Bellevue → nos 6 classes
BELLEVUE_CLASSES = [
    "pedestrian","people","bicycle","car",
    "van","truck","tricycle","awning-tricycle",
    "bus","motor"
]
BELLEVUE_MAPPING = {
    "pedestrian":"person",   "people":"person",
    "bicycle":"motorcycle",  "car":"car",
    "van":"car",             "truck":"truck",
    "tricycle":"motorcycle", "awning-tricycle":"motorcycle",
    "bus":"bus",             "motor":"motorcycle"
}
OUR_CLASSES = ["car","bus","truck","motorcycle","person","emergency_vehicle"]

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

# Charger modèle
session    = ort.InferenceSession(MODEL_PATH,
               providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"Modèle chargé : {MODEL_PATH}")

def preprocess(frame):
    h, w   = frame.shape[:2]
    scale  = 640 / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    blob   = np.full((640,640,3), 114, dtype=np.uint8)
    ph, pw = (640-nh)//2, (640-nw)//2
    blob[ph:ph+nh, pw:pw+nw] = cv2.resize(frame, (nw,nh))
    blob   = blob.astype(np.float32)/255.0
    blob   = blob.transpose(2,0,1)[np.newaxis,...]
    return blob, scale, (pw, ph)

def postprocess(outputs, scale, pad, orig_shape):
    orig_h, orig_w = orig_shape[:2]
    pw, ph = pad
    dets   = []
    pred   = outputs[0]
    if pred.ndim == 3:
        pred = pred[0].T if pred.shape[1] < pred.shape[2] else pred[0]
    n_cls = pred.shape[1] - 4
    for det in pred:
        cx,cy,bw,bh = det[0],det[1],det[2],det[3]
        scores  = det[4:]
        raw_cid = int(np.argmax(scores))
        conf    = float(scores[raw_cid])
        if conf < CONF_THRESH:
            continue
        if n_cls == 10:
            raw_label  = BELLEVUE_CLASSES[raw_cid]
            class_name = BELLEVUE_MAPPING.get(raw_label,"car")
            class_id   = OUR_CLASSES.index(class_name) \
                         if class_name in OUR_CLASSES else 0
        else:
            class_id = raw_cid % len(OUR_CLASSES)
        x1 = float(np.clip((cx-bw/2-pw)/scale, 0, orig_w))
        y1 = float(np.clip((cy-bh/2-ph)/scale, 0, orig_h))
        x2 = float(np.clip((cx+bw/2-pw)/scale, 0, orig_w))
        y2 = float(np.clip((cy+bh/2-ph)/scale, 0, orig_h))
        if x2-x1 < 10 or y2-y1 < 10:
            continue
        dets.append([x1,y1,x2,y2,conf,class_id])
    return dets

# Trouver vidéos
videos = []
for ext in ["*.mp4","*.avi","*.mov"]:
    videos.extend(glob.glob(f"{VIDEOS_DIR}/{ext}"))
videos.extend(glob.glob("data/videos/116th/*.mp4"))

print(f"Vidéos trouvées : {len(videos)}")

frame_count = 0
saved_count = 0

for video_path in videos:
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vname = Path(video_path).stem
    idx   = 0
    print(f"\nTraitement : {vname} ({total} frames)")

    while frame_count < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob, scale, pad = preprocess(frame)
        outputs = session.run(None, {input_name: blob})
        dets    = postprocess(outputs, scale, pad, frame.shape)

        if len(dets) > 0:
            # Sauvegarder image
            img_name = f"{vname}_f{idx:06d}.jpg"
            img_path = f"{OUTPUT_DIR}/images/{img_name}"
            cv2.imwrite(img_path, frame)

            # Sauvegarder labels YOLO format
            lbl_name = f"{vname}_f{idx:06d}.txt"
            lbl_path = f"{OUTPUT_DIR}/labels/{lbl_name}"
            with open(lbl_path, "w") as f:
                for det in dets:
                    x1,y1,x2,y2,conf,cls = det
                    # YOLO format : class cx cy w h (normalisé)
                    cx_n = ((x1+x2)/2) / w
                    cy_n = ((y1+y2)/2) / h
                    bw_n = (x2-x1) / w
                    bh_n = (y2-y1) / h
                    f.write(f"{cls} {cx_n:.6f} {cy_n:.6f} "
                            f"{bw_n:.6f} {bh_n:.6f}\n")

            saved_count += 1
            frame_count += 1
            if saved_count % 50 == 0:
                print(f"  {saved_count} frames annotées...")

        idx += INTERVAL
        if idx >= total:
            break

    cap.release()
    if frame_count >= MAX_FRAMES:
        break

print(f"\n{'='*50}")
print(f"Total frames annotées : {saved_count}")
print(f"Images : {OUTPUT_DIR}/images/")
print(f"Labels : {OUTPUT_DIR}/labels/")

# Créer data.yaml pour Kaggle
yaml_content = f"""path: .
train: images
val: images

nc: 6
names: ['car', 'bus', 'truck', 'motorcycle', 'person', 'emergency_vehicle']
"""
with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
    f.write(yaml_content)
print(f"YAML  : {OUTPUT_DIR}/data.yaml")
