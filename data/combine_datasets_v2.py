"""
Dataset V2 — Sans VisDrone
Sources : AI-ITS + Pedestrian + Surveillance + Emergency + Lqliaa
Classes : 0=car, 1=bus, 2=truck, 3=motorcycle, 4=person, 5=emergency_vehicle
"""
import os, shutil, random
from pathlib import Path

BASE   = Path('/Users/mouadassargual/Desktop/smart-traffic-agadir')
OUTPUT = BASE / 'dataset_v2'
random.seed(42)

# ─── SOURCES ──────────────────────────────────────────────
SOURCES = {
    'aiits':        (BASE / 'AI-ITS-7',                    'roboflow'),
    'pedestrian':   (BASE / 'Pedestrian-Detection-v1i-yolov8', 'roboflow'),
    'emergency':    (BASE / 'Emergency-Vehicles-12',        'roboflow'),   # CORRECTION 1 : nom exact
    'surveillance': (BASE / 'Surveillance-Cameras-yolov8', 'roboflow'),
    'lqliaa':       (BASE / 'Lqliaa-Traffic',              'roboflow'),   # CORRECTION 2 : Lqliaa pas VisDrone
}

# ─── REMAPPINGS ───────────────────────────────────────────
# Nos 6 classes : 0=car, 1=bus, 2=truck, 3=motorcycle, 4=person, 5=emergency_vehicle

MAPS = {
    # AI ITS (16 classes) → nos 6 classes
    # names: ['2 Axle Truck-M','2 Axle Truck-S','3 Axle Truck','Articulated Truck',
    #         'Bajaj','Bus-L','Bus-S','Car','Medium Public Transportation',
    #         'Micro Truck','Motorcycle','Non-motorized','Person','Pick up',
    #         'Semitrailer Truck','Three-Wheeled Motorcycle']
    'aiits': {
        0:2, 1:2, 2:2, 3:2,   # trucks → truck
        4:3,                    # Bajaj → motorcycle
        5:1, 6:1, 8:1,         # bus variants → bus
        7:0,                    # Car → car
        9:2, 13:2, 14:2,       # micro/pick up/semi → truck
        10:3, 15:3,             # motorcycle variants → motorcycle
        11:-1,                  # Non-motorized → ignorer
        12:4,                   # Person → person
    },

    # Pedestrian Detection (1 classe) → person
    # names: ['pedestrian']
    'pedestrian': {0: 4},

    # Emergency Vehicles (4 classes) → emergency_vehicle (ignorer Transport_Car)
    # names: ['Ambulance','Fire Truck','Police_Car','Transport_Car']
    'emergency': {0:5, 1:5, 2:5},   # 3=Transport_Car → ignoré

    # Surveillance Cameras (3 classes) → nos classes
    # names: ['bus','car','truck']    ← index 0=bus, 1=car, 2=truck
    # CORRECTION 3 : mapping était inversé dans le script original
    'surveillance': {0:1, 1:0, 2:2},  # bus→1, car→0, truck→2

    # Lqliaa (5 classes)
    # names: ['bus','car','motorcycle','person','truck']
    'lqliaa': {0:1, 1:0, 2:3, 3:4, 4:2},  # bus→1, car→0, moto→3, person→4, truck→2
}

# ─── FONCTIONS ────────────────────────────────────────────
def collect(src_dir, mode):
    pairs = []
    if mode == 'roboflow':
        for split in ['train', 'valid', 'val', 'test']:
            img_d = src_dir / split / 'images'
            lbl_d = src_dir / split / 'labels'
            if not img_d.exists():
                continue
            for img in img_d.glob('*.[jp][pn]g'):
                lbl = lbl_d / (img.stem + '.txt')
                if lbl.exists():
                    pairs.append((img, lbl))
    else:  # flat
        img_d = src_dir / 'images'
        lbl_d = src_dir / 'labels'
        if img_d.exists():
            for img in img_d.glob('*.[jp][pn]g'):
                lbl = lbl_d / (img.stem + '.txt')
                if lbl.exists():
                    pairs.append((img, lbl))
    return pairs

def remap(src_lbl, dst_lbl, cmap):
    lines = []
    with open(src_lbl) as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            nc = cmap.get(int(p[0]), -1)
            if nc != -1:
                lines.append(f"{nc} {' '.join(p[1:])}")
    if lines:
        with open(dst_lbl, 'w') as f:
            f.write('\n'.join(lines))
        return True
    return False

# ─── COLLECTE TOTALE ──────────────────────────────────────
print("🔍 Collecte des sources :")
all_pairs = []
for name, (src, mode) in SOURCES.items():
    if not src.exists():
        print(f"⚠️  INTROUVABLE : {src.name}")
        continue
    p = collect(src, mode)
    all_pairs.extend([(img, lbl, MAPS[name], name) for img, lbl in p])
    print(f"   ✅ {name:15s} : {len(p):5d} paires")

print(f"\n📦 Total collecté : {len(all_pairs)} images")

# ─── SPLIT 70/20/10 ───────────────────────────────────────
random.shuffle(all_pairs)
n     = len(all_pairs)
n_tr  = int(n * 0.70)
n_val = int(n * 0.20)
splits = {
    'train': all_pairs[:n_tr],
    'val':   all_pairs[n_tr:n_tr+n_val],
    'test':  all_pairs[n_tr+n_val:],
}

# ─── CRÉER STRUCTURE ──────────────────────────────────────
for sp in ['train', 'val', 'test']:
    (OUTPUT / 'images' / sp).mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'labels' / sp).mkdir(parents=True, exist_ok=True)

# ─── COPIER ET REMAPPER ───────────────────────────────────
print("\n📋 Copie et remapping :")
stats   = {sp: 0 for sp in splits}
skipped = 0

for sp, pairs in splits.items():
    for img, lbl, cmap, src in pairs:
        uname   = f"{src}_{img.name}"
        dst_img = OUTPUT / 'images' / sp / uname
        dst_lbl = OUTPUT / 'labels' / sp / (Path(uname).stem + '.txt')
        if remap(lbl, dst_lbl, cmap):
            shutil.copy2(img, dst_img)
            stats[sp] += 1
        else:
            skipped += 1

# ─── RÉSULTAT ─────────────────────────────────────────────
print(f"\n✅ Dataset V2 créé :")
total = 0
for sp in splits:
    print(f"   {sp:5s} : {stats[sp]:5d} images")
    total += stats[sp]
print(f"   TOTAL : {total:5d} images")
print(f"   ⚠️  Ignorées (aucun objet valide) : {skipped}")

# ─── DATA.YAML ────────────────────────────────────────────
yaml_content = f"""path: {OUTPUT}
train: images/train
val:   images/val
test:  images/test

nc: 6
names: ['car','bus','truck','motorcycle','person','emergency_vehicle']

# Dataset V2 — Sources :
# - AI ITS (surveillance intersection)
# - Pedestrian Detection (pietons)
# - Emergency Vehicles (urgence)
# - Surveillance Cameras (bus/car/truck)
# - Lqliaa Traffic (contexte marocain)
"""

with open(OUTPUT / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"\n📄 data.yaml écrit : {OUTPUT / 'data.yaml'}")
print("🚀 Dataset V2 prêt !")
