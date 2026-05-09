"""
Filtre VisDrone — Garde seulement les images avec >= MIN_VEHICLES véhicules
Classes véhicules : 0=car, 1=bus, 2=truck, 3=motorcycle
"""
import os, shutil
from pathlib import Path

BASE       = '/Users/mouadassargual/Desktop/smart-traffic-agadir'
SRC_IMAGES = Path(f'{BASE}/visdrone_yolo/images')
SRC_LABELS = Path(f'{BASE}/visdrone_yolo/labels')
DST_IMAGES = Path(f'{BASE}/visdrone_yolo_filtered/images')
DST_LABELS = Path(f'{BASE}/visdrone_yolo_filtered/labels')
DST_IMAGES.mkdir(parents=True, exist_ok=True)
DST_LABELS.mkdir(parents=True, exist_ok=True)

VEHICLE_CLASSES = {0, 1, 2, 3}  # car, bus, truck, motorcycle
MIN_VEHICLES    = 5              # seuil minimum

kept    = 0
removed = 0

for lbl_path in SRC_LABELS.glob('*.txt'):
    lines    = open(lbl_path).readlines()
    n_veh    = sum(1 for l in lines if l.strip()
                   and int(l.split()[0]) in VEHICLE_CLASSES)

    img_path = SRC_IMAGES / (lbl_path.stem + '.jpg')
    if not img_path.exists():
        continue

    if n_veh >= MIN_VEHICLES:
        shutil.copy2(str(img_path), str(DST_IMAGES / img_path.name))
        shutil.copy2(str(lbl_path), str(DST_LABELS / lbl_path.name))
        kept += 1
    else:
        removed += 1

total = kept + removed
print(f'✅ Gardées  : {kept:,}  ({kept/total*100:.1f}%)')
print(f'❌ Filtrées : {removed:,} ({removed/total*100:.1f}%)')
print(f'   Seuil   : >= {MIN_VEHICLES} véhicules')
print(f'   Dossier : {BASE}/visdrone_yolo_filtered/')
