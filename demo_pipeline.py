"""
demo_pipeline.py — Version corrigée
PFE Smart Traffic Agadir — Mouad ASSARGUAL
Zones polygonales + sigmoid scores + ASCII only
"""

import cv2, numpy as np, onnxruntime as ort
import argparse, os, time
from pathlib import Path

# ── Config ──
INPUT_SIZE  = 640
CONF_THRESH = 0.15
IOU_THRESH  = 0.45
BLUR_KERNEL = (51, 51)
CLASSES     = ["car","bus","truck","motorcycle","person","emergency_vehicle"]
CLASS_COLORS = {
    "car":               ( 0, 255,   0),
    "bus":               ( 0, 165, 255),
    "truck":             (255, 165,   0),
    "motorcycle":        ( 0, 255, 255),
    "person":            (255,   0, 255),
    "emergency_vehicle": ( 0,   0, 255),
}
W_VEH=1.0; W_PED=1.5; W_EMG=100.0
THETA_H=10; THETA_M=5

# ── Zones polygonales dynamiques ──
# (Les coordonnées sont calculées dynamiquement dans get_zones)

ZONE_COLORS = {
    "N": (255,  50,  50),   # rouge/rose → comme DiTAT
    "S": (255,   0, 255),   # magenta
    "E": (100, 100, 255),   # bleu
    "W": (  0, 255, 255),   # cyan
}
ZONE_LABELS = {
    "N": "NORD",
    "S": "SUD",
    "E": "EST",
    "W": "OUEST"
}

# ── Feux : positions par zone ──
LIGHT_POS = {"N":(640,40), "S":(640,680), "E":(1230,360), "W":(50,360)}
COLOR_GREEN=(0,220,0); COLOR_RED=(0,0,220)
COLOR_WHITE=(255,255,255); COLOR_BLACK=(0,0,0)
COLOR_BLUR=(255,0,255)

def load_model(path):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, sess_options=opts,
                             providers=["CPUExecutionProvider"])
    print(f"  Model : {Path(path).name}  OK")
    return s

def preprocess(frame):
    h, w = frame.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(frame, (nw, nh))
    blob = np.full((INPUT_SIZE,INPUT_SIZE,3), 114, np.uint8)
    ph, pw = (INPUT_SIZE-nh)//2, (INPUT_SIZE-nw)//2
    blob[ph:ph+nh, pw:pw+nw] = resized
    blob = blob.astype(np.float32)/255.0
    blob = blob.transpose(2,0,1)[np.newaxis,...]
    return blob, scale, (pw, ph)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def postprocess(outputs, scale, pad, orig_shape):
    """
    Format NMS intégré Ultralytics : (1, 300, 6)
    Chaque ligne = [x1, y1, x2, y2, conf, class_id]
    Coordonnées dans l'espace 640×640 letterboxed
    """
    orig_h, orig_w = orig_shape[:2]
    pw, ph = pad
    dets = []

    pred = outputs[0][0]  # (300, 6)

    for row in pred:
        x1, y1, x2, y2, conf, class_id = row
        class_id = int(class_id)
        conf     = float(conf)

        if conf < CONF_THRESH:
            continue
        if class_id >= len(CLASSES):
            continue

        # Letterbox → coordonnées originales
        x1 = float(np.clip((x1 - pw) / scale, 0, orig_w))
        y1 = float(np.clip((y1 - ph) / scale, 0, orig_h))
        x2 = float(np.clip((x2 - pw) / scale, 0, orig_w))
        y2 = float(np.clip((y2 - ph) / scale, 0, orig_h))

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        dets.append([x1, y1, x2, y2, conf, class_id])

    # NMS déjà fait par le modèle → retourner directement
    return dets

def nms(dets, thr=IOU_THRESH):
    if not dets: return []
    # ── NMS par classe (évite double bbox sur même objet) ──
    final = []
    classes = set(int(d[5]) for d in dets)
    for cid in classes:
        cls_dets = [d for d in dets if int(d[5])==cid]
        if not cls_dets: continue
        boxes  = np.array([[d[0],d[1],d[2],d[3]] for d in cls_dets])
        scores = np.array([d[4] for d in cls_dets])
        idx    = np.argsort(scores)[::-1]
        keep   = []
        while len(idx):
            i = idx[0]; keep.append(i)
            if len(idx)==1: break
            xx1=np.maximum(boxes[i,0],boxes[idx[1:],0])
            yy1=np.maximum(boxes[i,1],boxes[idx[1:],1])
            xx2=np.minimum(boxes[i,2],boxes[idx[1:],2])
            yy2=np.minimum(boxes[i,3],boxes[idx[1:],3])
            inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
            ai=(boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
            ar=((boxes[idx[1:],2]-boxes[idx[1:],0])*
                (boxes[idx[1:],3]-boxes[idx[1:],1]))
            iou=inter/(ai+ar-inter+1e-6)
            idx=idx[1:][iou<thr]
        final.extend([cls_dets[i] for i in keep])
    return final

def anonymize(frame, dets):
    n = 0
    for x1,y1,x2,y2,_,cid in dets:
        if CLASSES[cid]=="person":
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            roi=frame[y1:y2,x1:x2]
            if roi.size>0:
                frame[y1:y2,x1:x2]=cv2.GaussianBlur(roi,BLUR_KERNEL,0)
                n+=1
    return frame, n

def get_zones(W, H):
    """Calcule les polygones dynamiquement selon la résolution réelle, basé sur vos limites."""
    return {
        "N": np.array([
            [W // 3, H // 10], 
            [W // 2 + 100, H // 10], 
            [W // 2 + 100, H // 5], 
            [W // 2 - 200, H // 3]
        ], np.int32),
        "E": np.array([
            [W // 2 + 150,  H // 5], 
            [W // 2 + 550, H // 5 - 100], 
            [W // 2 + 450, H // 3 + 150], 
            [W // 2 + 100, H // 3 - 100]
        ], np.int32),
        "S": np.array([
            [W // 2,  H - 100], 
            [W, H // 2], 
            [W, H], 
            [W // 2 - 100, H]
        ], np.int32),
        "W": np.array([
            [0,  H // 2], 
            [W // 4 + 100,  H // 2 - 100], 
            [W // 2, H - 50], 
            [0, H]
        ], np.int32)
    }

def point_in_zone(cx, cy, zones):
    for name, poly in zones.items():
        if cv2.pointPolygonTest(poly, (float(cx),float(cy)), False) >= 0:
            return name
    return "C"

def count_zones(dets, zones):
    counts = {"N":0,"S":0,"E":0,"W":0,"ped":0,"emergency":False}
    for x1,y1,x2,y2,_,cid in dets:
        cx,cy = (x1+x2)/2, (y1+y2)/2
        label = CLASSES[cid]
        zone  = point_in_zone(cx, cy, zones)
        if label=="emergency_vehicle":
            counts["emergency"]=True
            if zone in counts: counts[zone]+=1
        elif label=="person":
            counts["ped"]+=1
        elif label in ["car","bus","truck","motorcycle"]:
            if zone in ["N","S","E","W"]: counts[zone]+=1
    return counts

def mdp(counts):
    if counts["emergency"]: return "EMERGENCY",45,"URGENCE ABSOLUE"
    ns = W_VEH*(counts["N"]+counts["S"])+W_PED*counts["ped"]
    ew = W_VEH*(counts["E"]+counts["W"])
    phase = "NORD-SUD" if ns>=ew else "EST-OUEST"
    q = counts["N"]+counts["S"] if phase=="NORD-SUD" else counts["E"]+counts["W"]
    dur = 45 if q>=THETA_H else (30 if q>=THETA_M else 15)
    reason = f"NS={ns:.0f} vs EW={ew:.0f}"
    return phase, dur, reason

def draw_zones_overlay(frame, zones):
    """Remplissage semi-transparent style DiTAT"""
    overlay = frame.copy()
    for name, poly in zones.items():
        color = ZONE_COLORS[name]
        cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    # Contours + labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, poly in zones.items():
        cv2.polylines(frame, [poly], True, ZONE_COLORS[name], 2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.putText(frame, ZONE_LABELS[name], (cx-30, cy+8),
                       font, 0.7, ZONE_COLORS[name], 2)
    return frame

def draw_dets(frame, dets, zones):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x1,y1,x2,y2,conf,cid in dets:
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        label = CLASSES[cid]
        color = CLASS_COLORS.get(label,(200,200,200))
        if label=="person":
            cv2.rectangle(frame,(x1,y1),(x2,y2),COLOR_BLUR,1)
            cv2.putText(frame,"ANON",(x1,max(y1-3,10)),
                       font,0.35,COLOR_BLUR,1)
            continue
        z = point_in_zone((x1+x2)/2,(y1+y2)/2,zones)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{label[:3].upper()} {conf:.2f} [{z}]",
                   (x1,max(y1-4,10)),font,0.38,color,1)
    return frame

def draw_lights(frame, phase):
    """Feux aux 4 branches"""
    for d,(px,py) in LIGHT_POS.items():
        if phase=="EMERGENCY":
            c = COLOR_GREEN if d in ["E","W"] else COLOR_RED
        elif phase=="NORD-SUD":
            c = COLOR_GREEN if d in ["N","S"] else COLOR_RED
        elif phase=="EST-OUEST":
            c = COLOR_GREEN if d in ["E","W"] else COLOR_RED
        else:
            c = COLOR_RED
        cv2.rectangle(frame,(px-20,py-20),(px+20,py+20),COLOR_BLACK,-1)
        cv2.rectangle(frame,(px-20,py-20),(px+20,py+20),COLOR_WHITE,1)
        cv2.circle(frame,(px,py),14,c,-1)
        cv2.circle(frame,(px,py),14,COLOR_WHITE,1)
        cv2.putText(frame,d,(px-6,py+34),
                   cv2.FONT_HERSHEY_SIMPLEX,0.5,COLOR_WHITE,1)
    return frame

def draw_hud(frame, counts, phase, dur, reason, fps, anon, fidx):
    H,W = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov,(0,H-105),(W,H),COLOR_BLACK,-1)
    cv2.addWeighted(ov,0.75,frame,0.25,0,frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    yb   = H-105+18
    emg  = " | URGENCE!" if counts["emergency"] else ""
    cv2.putText(frame,
        f"ZONES: N={counts['N']} S={counts['S']} "
        f"E={counts['E']} W={counts['W']} "
        f"Pietons={counts['ped']}{emg}",
        (10,yb),font,0.52,COLOR_WHITE,1)
    pc = COLOR_RED if phase=="EMERGENCY" else (0,220,0)
    cv2.putText(frame,
        f"MDP: Phase {phase} | Duree {dur}s | {reason}",
        (10,yb+26),font,0.48,pc,1)
    cv2.putText(frame,
        f"Privacy-by-Design: {anon} pieton(s) anonymise(s) [Loi 09-08 Maroc]",
        (10,yb+52),font,0.45,COLOR_BLUR,1)
    cv2.putText(frame,
        f"FPS:{fps:.1f} | Frame:{fidx} | YOLO26n INT8 | Raspberry Pi 5",
        (10,yb+78),font,0.42,(180,180,180),1)
    cv2.putText(frame,
        "SMART TRAFFIC AGADIR -- PFE M2 IA Embarquee -- FSA Ait Melloul",
        (10,22),font,0.48,COLOR_WHITE,1)
    return frame

def run(video_path, model_path, save=False, max_frames=None):
    session = load_model(model_path)
    inp     = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    if not cap.isOpened(): raise IOError(f"Cannot open: {video_path}")
    fps0 = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {W}x{H} @ {fps0:.1f} FPS")

    zones = get_zones(W, H)

    writer = None
    if save:
        out = video_path.replace(Path(video_path).suffix,"_demo.mp4")
        writer = cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*"mp4v"),fps0,(W,H))
        print(f"  Saving → {out}")

    phase,dur,reason = "NORD-SUD",30,"Init"
    timer=0; INTERVAL=30
    total_anon=0; fidx=0; fps_disp=0.0
    t0=time.time()

    while True:
        ret,frame = cap.read()
        if not ret or (max_frames and fidx>=max_frames): break
        tf = time.time()

        blob,scale,pad = preprocess(frame)
        out_ = session.run(None,{inp:blob})
        dets = postprocess(out_,scale,pad,frame.shape)

        # M4 AVANT M3
        frame, anon = anonymize(frame, dets)
        total_anon += anon

        counts = count_zones(dets, zones)
        timer += 1
        if timer>=INTERVAL or counts["emergency"]:
            phase,dur,reason = mdp(counts)
            timer=0
            print(f"  [f={fidx:5d}] N={counts['N']} S={counts['S']} "
                  f"E={counts['E']} W={counts['W']} "
                  f"P={counts['ped']} -> {phase}/{dur}s")

        fps_disp = 1.0/(time.time()-tf+1e-6)
        draw_zones_overlay(frame, zones)
        draw_dets(frame, dets, zones)

        cv2.imshow("Smart Traffic Agadir -- PFE Demo", frame)
        if writer: writer.write(frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        fidx+=1

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    elapsed=time.time()-t0
    print(f"\n  Frames: {fidx} | Anon: {total_anon} | "
          f"FPS moy: {fidx/elapsed:.1f}")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",  default="data/videos/116th/Bellevue_116th_NE12th__2017-09-10_19-08-25.mp4")
    ap.add_argument("--model",  default=os.path.expanduser(
        "~/Desktop/smart-traffic-agadir/models_rpi/custom/yolo26n_best.onnx"))
    ap.add_argument("--save",   action="store_true")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    run(args.video, args.model, args.save, args.frames)
