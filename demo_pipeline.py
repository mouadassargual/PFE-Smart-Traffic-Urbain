"""
demo_pipeline.py
================
Pipeline complet — PFE Smart Traffic Agadir
M1 → M2 → M4 → M3 → M5 → M6

Auteur  : Mouad ASSARGUAL
PFE M2  : Intelligence Artificielle Embarquee
FSA Ait Melloul — Universite Ibn Zohr, Agadir
Annee   : 2025/2026

Usage :
  python3 demo_pipeline.py --video data/videos/ne8th --profile ne8th
  python3 demo_pipeline.py --video data/videos/ne8th --model models_rpi/custom/bellevue_best.onnx --profile ne8th --save
  python3 demo_pipeline.py --video data/videos/116th --profile 116th --start 18000
  python3 demo_pipeline.py --video data/videos/ne8th --profile ne8th --lights  (afficher feux)
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import json
import time
import glob
from pathlib import Path

try:
    from mqtt_sender import MQTTSender
    MQTT_MODULE_AVAILABLE = True
except ImportError:
    MQTT_MODULE_AVAILABLE = False


# ════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════

MODEL_PATH = os.path.expanduser(
    "~/Desktop/smart-traffic-agadir/models_rpi/custom/bellevue_best.onnx")

INPUT_SIZE      = 640
CONF_THRESH_VEH = 0.30
CONF_THRESH_PED = 0.20
CONF_THRESH     = 0.25
IOU_THRESH      = 0.45
BLUR_KERNEL     = (51, 51)

# Nos 6 classes PFE
CLASSES = ["car", "bus", "truck", "motorcycle", "person", "emergency_vehicle"]

# Classes modele Bellevue (10 classes — VisDrone)
CLASSES_BELLEVUE = [
    "pedestrian", "people", "bicycle", "car",
    "van", "truck", "tricycle", "awning-tricycle",
    "bus", "motor"
]

# Mapping Bellevue → nos 6 classes
BELLEVUE_MAPPING = {
    "pedestrian":      "person",
    "people":          "person",
    "bicycle":         "motorcycle",
    "car":             "car",
    "van":             "car",
    "truck":           "truck",
    "tricycle":        "motorcycle",
    "awning-tricycle": "motorcycle",
    "bus":             "bus",
    "motor":           "motorcycle",
}

CLASS_COLORS = {
    "car":               (  0, 255,   0),
    "bus":               (  0, 165, 255),
    "truck":             (255, 165,   0),
    "motorcycle":        (  0, 255, 255),
    "person":            (255,   0, 255),
    "emergency_vehicle": (  0,   0, 255),
}

# Poids MDP
W_VEH     = 1.0
W_PED     = 1.5
THETA_H   = 10
THETA_M   = 5
DUR_LONG  = 45
DUR_MED   = 30
DUR_SHORT = 15

# Profils de zones calibres par intersection
ZONE_PROFILES = {
    "116th": {
        "N": np.array([[447,25],[716,146],[452,202],[406,38],[439,25]], np.int32),
        "E": np.array([[791,157],[1032,311],[1104,136],[991,77],[788,159]], np.int32),
        "S": np.array([[1087,404],[763,716],[1263,704],[1269,614],[1089,403]], np.int32),
        "W": np.array([[400,250],[503,681],[12,707],[7,414],[396,250]], np.int32),
    },
    "ne8th": {
        "N": np.array([[594,46],[538,47],[481,241],[868,227],[598,46]], np.int32),
        "E": np.array([[873,228],[1118,508],[1275,291],[1235,250],[873,227]], np.int32),
        "S": np.array([[1118,509],[542,678],[913,695],[1268,694],[1119,510]], np.int32),
        "W": np.array([[3,355],[479,240],[418,592],[3,531],[3,357]], np.int32),
    },
}

ZONE_POLYGONS_1280x720 = dict(ZONE_PROFILES["116th"])

ZONE_COLORS = {
    "N": (255, 100, 100),
    "S": (100, 255, 100),
    "E": (100, 100, 255),
    "W": (255, 255,   0),
}

LIGHT_POS = {
    "N": (640,  40),
    "S": (640, 680),
    "E": (1230, 360),
    "W": (50,  360),
}

COLOR_GREEN = (  0, 220,   0)
COLOR_RED   = (  0,   0, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (  0,   0,   0)
COLOR_BLUR  = (255,   0, 255)


# ════════════════════════════════════════════════════════
# M2 — CHARGEMENT MODELE
# ════════════════════════════════════════════════════════

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modele introuvable : {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path, sess_options=opts,
        providers=["CPUExecutionProvider"])
    print(f"  Modele  : {Path(model_path).name}  OK")
    return session


# ════════════════════════════════════════════════════════
# M2 — PREPROCESSING
# ════════════════════════════════════════════════════════

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    blob = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    ph, pw = (INPUT_SIZE - nh) // 2, (INPUT_SIZE - nw) // 2
    blob[ph:ph+nh, pw:pw+nw] = resized
    blob = blob.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, scale, (pw, ph)


# ════════════════════════════════════════════════════════
# M2 — MAPPING CLASSES
# ════════════════════════════════════════════════════════

def map_class(raw_cid, n_cls):
    """
    Mappe class_id brut vers nos 6 classes PFE
    Supporte : 6 classes (custom/yolov8n), 10 classes (Bellevue/VisDrone)
    Returns  : class_id (0-5) ou -1 si a ignorer
    """
    if n_cls == 10:
        if raw_cid >= len(CLASSES_BELLEVUE):
            return -1
        raw_label = CLASSES_BELLEVUE[raw_cid]
        mapped    = BELLEVUE_MAPPING.get(raw_label, "car")
        return CLASSES.index(mapped) if mapped in CLASSES else 0
    elif n_cls == 6:
        return raw_cid if raw_cid < 6 else -1
    else:
        return raw_cid % len(CLASSES)


# ════════════════════════════════════════════════════════
# M2 — POSTPROCESSING (supporte tous formats YOLO)
# ════════════════════════════════════════════════════════

def postprocess(outputs, scale, pad, orig_shape):
    """
    Decode sorties YOLO — 3 formats supportes :
      Format A : (1, 300, 6) NMS integre [x1,y1,x2,y2,conf,cls]
      Format B : (1, N, 8400) cxcywh + scores
      Format C : (1, 8400, N) transpose de B
    """
    orig_h, orig_w = orig_shape[:2]
    pw, ph = pad
    dets   = []
    pred   = outputs[0]

    # Format A : NMS integre Ultralytics (1, 300, 6)
    if pred.ndim == 3 and pred.shape[1] == 300 and pred.shape[2] == 6:
        pred = pred[0]  # (300, 6)
        for row in pred:
            x1, y1, x2, y2, conf, raw_cid = row
            raw_cid  = int(raw_cid)
            conf     = float(conf)
            class_id = map_class(raw_cid, n_cls=6)
            if class_id < 0:
                continue
            conf_min = CONF_THRESH_PED if class_id == 4 else CONF_THRESH_VEH
            if conf < conf_min:
                continue
            x1 = float(np.clip((x1 - pw) / scale, 0, orig_w))
            y1 = float(np.clip((y1 - ph) / scale, 0, orig_h))
            x2 = float(np.clip((x2 - pw) / scale, 0, orig_w))
            y2 = float(np.clip((y2 - ph) / scale, 0, orig_h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            dets.append([x1, y1, x2, y2, conf, class_id])

    # Format B/C : cxcywh + scores
    else:
        if pred.ndim == 3:
            pred = pred[0].T if pred.shape[1] < pred.shape[2] else pred[0]

        n_cls = pred.shape[1] - 4

        for det in pred:
            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            scores  = det[4:]
            raw_cid = int(np.argmax(scores))
            conf    = float(scores[raw_cid])

            class_id = map_class(raw_cid, n_cls=n_cls)
            if class_id < 0:
                continue

            conf_min = CONF_THRESH_PED if class_id == 4 else CONF_THRESH_VEH
            if conf < conf_min:
                continue

            x1 = float(np.clip((cx - bw/2 - pw) / scale, 0, orig_w))
            y1 = float(np.clip((cy - bh/2 - ph) / scale, 0, orig_h))
            x2 = float(np.clip((cx + bw/2 - pw) / scale, 0, orig_w))
            y2 = float(np.clip((cy + bh/2 - ph) / scale, 0, orig_h))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            dets.append([x1, y1, x2, y2, conf, class_id])

    return nms(dets)


def nms(dets, thresh=IOU_THRESH):
    """NMS par classe"""
    if not dets:
        return []
    final = []
    for cid in set(int(d[5]) for d in dets):
        cd      = [d for d in dets if int(d[5]) == cid]
        boxes   = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in cd]
        scores  = [d[4] for d in cd]
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, thresh)
        if len(indices) > 0:
            final.extend([cd[i] for i in indices.flatten()])
    return final


def detect_with_zoom(session, frame, input_name):
    """Detection frame complet + zoom zones N et E"""
    H, W = frame.shape[:2]
    all_dets = []

    blob, s, p = preprocess(frame)
    out  = session.run(None, {input_name: blob})
    all_dets.extend(postprocess(out, s, p, frame.shape))

    rois = {}
    for _, (roi, ox, oy) in rois.items():
        if roi.size == 0:
            continue
        bz, sz, pz = preprocess(roi)
        oz  = session.run(None, {input_name: bz})
        dz  = postprocess(oz, sz, pz, roi.shape)
        for d in dz:
            d2 = d.copy()
            d2[0] += ox; d2[2] += ox
            d2[1] += oy; d2[3] += oy
            all_dets.append(d2)

    return nms(all_dets)


# ════════════════════════════════════════════════════════
# M4 — ANONYMISATION Privacy-by-Design
# ════════════════════════════════════════════════════════

def anonymize_persons(frame, detections):
    """Flou Gaussien pietons avant stockage — Loi 09-08 Maroc"""
    anon_count = 0
    for det in detections:
        if CLASSES[int(det[5])] == "person":
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
                anon_count += 1
    return frame, anon_count


# ════════════════════════════════════════════════════════
# M3 — TRACKER IoU
# ════════════════════════════════════════════════════════

class SimpleTracker:
    def __init__(self, iou_thresh=0.25, max_age=15):
        self.tracks     = {}
        self.next_id    = 0
        self.iou_thresh = iou_thresh
        self.max_age    = max_age

    def iou(self, a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter  = max(0,ix2-ix1)*max(0,iy2-iy1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        union  = area_a + area_b - inter
        return inter/union if union > 0 else 0

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid][4] += 1
                if self.tracks[tid][4] > self.max_age:
                    del self.tracks[tid]
            return []

        matched   = set()
        track_ids = list(self.tracks.keys())
        results   = []

        for det in detections:
            best_iou = self.iou_thresh
            best_tid = None
            for tid in track_ids:
                if tid in matched:
                    continue
                v = self.iou(det, self.tracks[tid])
                if v > best_iou:
                    best_iou = v
                    best_tid = tid

            if best_tid is not None:
                self.tracks[best_tid] = list(det[:4]) + [0]
                matched.add(best_tid)
                results.append(list(det) + [best_tid])
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = list(det[:4]) + [0]
                results.append(list(det) + [tid])

        for tid in track_ids:
            if tid not in matched:
                self.tracks[tid][4] += 1
                if self.tracks[tid][4] > self.max_age:
                    del self.tracks[tid]

        return results


# ════════════════════════════════════════════════════════
# M3 — ZONES & COMPTAGE
# ════════════════════════════════════════════════════════

def get_zones(W, H):
    sx, sy = W/1280, H/720
    zones  = {}
    for name, pts in ZONE_POLYGONS_1280x720.items():
        sc = pts.astype(np.float32).copy()
        sc[:,0] *= sx; sc[:,1] *= sy
        zones[name] = sc.astype(np.int32)
    return zones


def point_in_zone(cx, cy, zones):
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
            return name
    return "C"


def count_per_zone(detections, zones):
    counts = {"N":0,"S":0,"E":0,"W":0,"ped":0,"emergency":False}
    for det in detections:
        class_id = int(det[5])
        cx = (det[0]+det[2])/2
        cy = (det[1]+det[3])/2
        zone  = point_in_zone(cx, cy, zones)
        label = CLASSES[class_id]

        if label == "emergency_vehicle":
            counts["emergency"] = True
            if zone in ["N","S","E","W"]:
                counts[zone] += 1
        elif label == "person":
            counts["ped"] += 1
        elif label in ["car","bus","truck","motorcycle"]:
            if zone in ["N","S","E","W"]:
                counts[zone] += 1
    return counts


# ════════════════════════════════════════════════════════
# M5 — DECISION MDP
# ════════════════════════════════════════════════════════

def mdp_decision(counts):
    """pi*(s) — S=(q_N,q_S,q_E,q_W,p_t) A={15,30,45} R=-AWT"""
    if counts.get("emergency", False):
        return "EMERGENCY", DUR_LONG, "Vehicule urgence detecte"

    score_NS = W_VEH*(counts["N"]+counts["S"]) + W_PED*counts["ped"]
    score_EW = W_VEH*(counts["E"]+counts["W"])

    if score_NS >= score_EW:
        phase = "NORD-SUD"
        q_dom = counts["N"]+counts["S"]
        reason = f"Score NS={score_NS:.1f} > EW={score_EW:.1f}"
    else:
        phase = "EST-OUEST"
        q_dom = counts["E"]+counts["W"]
        reason = f"Score EW={score_EW:.1f} > NS={score_NS:.1f}"

    dur     = DUR_LONG if q_dom >= THETA_H else (DUR_MED if q_dom >= THETA_M else DUR_SHORT)
    reason += f" | q={q_dom} -> {dur}s"
    return phase, dur, reason


# ════════════════════════════════════════════════════════
# M6 — OVERLAY VISUEL
# ════════════════════════════════════════════════════════

def draw_zones_overlay(frame, zones):
    overlay = frame.copy()
    for name, poly in zones.items():
        cv2.fillPoly(overlay, [poly], ZONE_COLORS[name])
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], True, ZONE_COLORS[name], 2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.putText(frame, name, (cx-15,cy+8), font, 1.0, ZONE_COLORS[name], 2)
    return frame


def draw_detections(frame, detections, zones):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in detections:
        x1,y1,x2,y2 = int(det[0]),int(det[1]),int(det[2]),int(det[3])
        conf     = det[4]
        class_id = int(det[5])
        track_id = int(det[6]) if len(det) > 6 else 0
        label    = CLASSES[class_id]
        color    = CLASS_COLORS.get(label, (200,200,200))

        if label == "person":
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_BLUR, 1)
            cv2.putText(frame, "ANON", (x1,max(y1-3,10)),
                       font, 0.35, COLOR_BLUR, 1)
            continue

        if label == "emergency_vehicle":
            cv2.rectangle(frame, (x1-3,y1-3), (x2+3,y2+3), (0,0,255), 3)

        zone = point_in_zone((x1+x2)/2, (y1+y2)/2, zones)
        if zone == "C":
            color = (128,128,128)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"#{track_id} {conf:.2f} [{zone}]",
                   (x1,max(y1-4,10)), font, 0.38, color, 1)
    return frame


def draw_traffic_lights(frame, phase):
    for direction, (px,py) in LIGHT_POS.items():
        if phase == "EMERGENCY":
            c = COLOR_GREEN if direction in ["E","W"] else COLOR_RED
        elif phase == "NORD-SUD":
            c = COLOR_GREEN if direction in ["N","S"] else COLOR_RED
        elif phase == "EST-OUEST":
            c = COLOR_GREEN if direction in ["E","W"] else COLOR_RED
        else:
            c = COLOR_RED
        cv2.rectangle(frame, (px-20,py-20), (px+20,py+20), COLOR_BLACK, -1)
        cv2.rectangle(frame, (px-20,py-20), (px+20,py+20), COLOR_WHITE, 1)
        cv2.circle(frame, (px,py), 14, c, -1)
        cv2.circle(frame, (px,py), 14, COLOR_WHITE, 1)
        cv2.putText(frame, direction, (px-6,py+34),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    return frame


def draw_dashboard(frame, counts, phase, duration, reason,
                   fps, anon_count, frame_idx):
    H, W = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,H-110), (W,H), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    y_base = H - 110 + 18

    emg = " | URGENCE!" if counts["emergency"] else ""
    cv2.putText(frame,
        f"ZONES: N={counts['N']} S={counts['S']} "
        f"E={counts['E']} W={counts['W']} "
        f"Pietons={counts['ped']}{emg}",
        (10,y_base), font, 0.52, COLOR_WHITE, 1)

    pc = (0,0,220) if phase == "EMERGENCY" else COLOR_GREEN
    cv2.putText(frame,
        f"MDP: Phase {phase} | Duree {duration}s | {reason}",
        (10,y_base+26), font, 0.48, pc, 1)

    cv2.putText(frame,
        f"Privacy-by-Design: {anon_count} pieton(s) anonymise(s) [Loi 09-08 Maroc]",
        (10,y_base+52), font, 0.45, COLOR_BLUR, 1)

    cv2.putText(frame,
        f"FPS:{fps:.1f} | Frame:{frame_idx} | YOLO26n INT8 | Raspberry Pi 5",
        (10,y_base+78), font, 0.42, (180,180,180), 1)

    cv2.putText(frame,
        "SMART TRAFFIC AGADIR -- PFE M2 IA Embarquee -- FSA Ait Melloul",
        (10,22), font, 0.48, COLOR_WHITE, 1)
    return frame


# ════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════

def find_video(path):
    if os.path.isfile(path):
        return path
    for ext in ["*.mp4","*.avi","*.mov","*.MP4"]:
        m = glob.glob(os.path.join(path, ext))
        if m:
            return m[0]
    raise FileNotFoundError(f"Aucune video dans : {path}")


def run_pipeline(video_path, model_path=MODEL_PATH, save=False,
                 max_frames=None, start_frame=0,
                 use_mqtt=False, mqtt_host="localhost",
                 profile_arg="auto", show_lights=False):

    video_file = find_video(video_path)

    if profile_arg == "auto":
        profile = "116th"
        for part in Path(video_file).parts:
            if "ne8th"  in part.lower(): profile = "ne8th";  break
            if "116th"  in part.lower(): profile = "116th";  break
    else:
        profile = profile_arg

    ZONE_POLYGONS_1280x720.update(ZONE_PROFILES[profile])

    print(f"\n{'='*62}")
    print(f"  PFE Smart Traffic Agadir — Pipeline Demonstration")
    print(f"{'='*62}")
    print(f"  Video  : {Path(video_file).name}")
    print(f"  Profil : {profile}")
    print(f"  Modele : {Path(model_path).name}")
    print(f"  MQTT   : {'actif -> '+mqtt_host if use_mqtt else 'desactive'}")

    session    = load_model(model_path)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"Impossible d ouvrir : {video_file}")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps0  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {W}x{H} @ {fps0:.1f} FPS")
    print(f"  Total frames : {total} (debut={start_frame})")

    zones   = get_zones(W, H)
    tracker = SimpleTracker(iou_thresh=0.25, max_age=15)

    writer   = None
    out_path = None
    if save:
        stem     = Path(video_file).stem
        out_path = str(Path(video_file).parent / f"{stem}_demo.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps0, (W, H))
        print(f"  Sortie : {out_path}")

    sender = None
    if use_mqtt and MQTT_MODULE_AVAILABLE:
        sender = MQTTSender(broker_host=mqtt_host, verbose=False)
        sender.connect()

    print(f"\n  Appuyer sur [Q] pour quitter\n")

    current_phase    = "NORD-SUD"
    current_duration = DUR_MED
    current_reason   = "Initialisation"
    phase_timer      = 0
    DECISION_INTERVAL = 30
    emergency_handled = False

    total_anon = 0
    total_dets = 0
    frame_idx  = 0
    t_start    = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        t_frame = time.time()

        # M2 : Detection
        detections = detect_with_zoom(session, frame, input_name)
        total_dets += len(detections)

        # M3 : Tracking
        tracked = tracker.update(detections)

        # M4 : Anonymisation AVANT comptage (Privacy-by-Design)
        frame, anon_count = anonymize_persons(frame, tracked)
        total_anon += anon_count

        # M3 : Comptage par zone
        counts = count_per_zone(tracked, zones)

        # M5 : Decision MDP
        phase_timer += 1
        if counts["emergency"] and not emergency_handled:
            current_phase, current_duration, current_reason = mdp_decision(counts)
            emergency_handled = True
            phase_timer = 0
            print(f"  [f={frame_idx:5d}] URGENCE -> {current_phase}/{current_duration}s")
        elif phase_timer >= DECISION_INTERVAL:
            current_phase, current_duration, current_reason = mdp_decision(counts)
            phase_timer = 0
            if counts["emergency"]:
                emergency_handled = True
            print(f"  [f={frame_idx:5d}] "
                  f"N={counts['N']} S={counts['S']} "
                  f"E={counts['E']} W={counts['W']} "
                  f"P={counts['ped']} "
                  f"-> {current_phase}/{current_duration}s")

        fps_display = 1.0 / (time.time() - t_frame + 1e-6)

        if sender and frame_idx % 5 == 0:
            sender.send_counts(counts, frame_id=frame_idx)
            sender.send_decision(current_phase, current_duration,
                                current_reason, frame_id=frame_idx)
            sender.send_privacy(anon_count, frame_id=frame_idx)
            sender.send_metrics(fps=fps_display, frame_id=frame_idx,
                               total_anon=total_anon,
                               detections=len(detections))

        # M6 : Overlay
        draw_zones_overlay(frame, zones)
        draw_detections(frame, tracked, zones)
        if show_lights:
            draw_traffic_lights(frame, current_phase)
        draw_dashboard(frame, counts, current_phase, current_duration,
                      current_reason, fps_display, anon_count, frame_idx)

        cv2.imshow("Smart Traffic Agadir -- PFE Demo", frame)
        if writer:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  Arret utilisateur.")
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"\n  Video sauvegardee -> {out_path}")
    if sender:
        sender.disconnect()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n{'='*62}")
    print(f"  Frames traites     : {frame_idx}")
    print(f"  Detections totales : {total_dets}")
    print(f"  Pietons anonymises : {total_anon}")
    print(f"  FPS moyen          : {frame_idx/elapsed:.1f}")
    print(f"{'='*62}\n")

    os.makedirs("results", exist_ok=True)
    with open("results/pipeline_report.json","w") as f:
        json.dump({
            "video": str(video_file), "model": str(model_path),
            "profile": profile, "frames": frame_idx,
            "detections": total_dets, "anonymized": total_anon,
            "fps_avg": round(frame_idx/elapsed,1),
            "duration_s": round(elapsed,1),
        }, f, indent=2)
    print("  Rapport -> results/pipeline_report.json")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pipeline Smart Traffic Agadir — PFE M2 IA Embarquee")
    ap.add_argument("--video",     type=str, default="data/videos/ne8th")
    ap.add_argument("--model",     type=str, default=MODEL_PATH)
    ap.add_argument("--save",      action="store_true")
    ap.add_argument("--frames",    type=int, default=None)
    ap.add_argument("--start",     type=int, default=0)
    ap.add_argument("--mqtt",      action="store_true")
    ap.add_argument("--mqtt-host", type=str, default="localhost")
    ap.add_argument("--profile",   type=str, default="auto",
                    choices=["auto","116th","ne8th"])
    ap.add_argument("--lights",    action="store_true",
                    help="Afficher les feux MDP sur le frame")
    args = ap.parse_args()

    run_pipeline(
        video_path  = args.video,
        model_path  = args.model,
        save        = args.save,
        max_frames  = args.frames,
        start_frame = args.start,
        use_mqtt    = args.mqtt,
        mqtt_host   = args.mqtt_host,
        profile_arg = args.profile,
        show_lights = args.lights,
    )