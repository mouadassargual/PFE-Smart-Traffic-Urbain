import os, shutil, random
from pathlib import Path

random.seed(42)

BASE = '/Users/mouadassargual/Desktop/smart-traffic-agadir'
OUT  = f'{BASE}/data/dataset'

# Créer structure
for split in ['train', 'val', 'test']:
    os.makedirs(f'{OUT}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUT}/labels/{split}', exist_ok=True)

# ── Mapping classes ───────────────────────────
EMERGENCY_MAP = {0: 5, 1: 5, 2: 5}  # 3=ignorer

LQLIAA_MAP = {0: 1, 1: 0, 2: 3, 3: 4, 4: 2}

def remap_label(label_path, class_map, out_path):
    lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            if cls not in class_map:
                continue
            new_cls = class_map[cls]
            lines.append(
                f"{new_cls} {' '.join(parts[1:])}"
            )
    if lines:
        with open(out_path, 'w') as f:
            f.write('\n'.join(lines))
        return True
    return False

def copy_source(images_dir, labels_dir,
                prefix, class_map=None):
    images = list(Path(images_dir).glob('*.jpg'))
    images += list(Path(images_dir).glob('*.png'))
    random.shuffle(images)

    n     = len(images)
    n_tr  = int(n * 0.70)
    n_val = int(n * 0.20)

    splits = {
        'train': images[:n_tr],
        'val'  : images[n_tr:n_tr+n_val],
        'test' : images[n_tr+n_val:],
    }

    total = 0
    for split, imgs in splits.items():
        for img in imgs:
            lbl = Path(labels_dir) / (img.stem + '.txt')
            if not lbl.exists():
                continue

            new_img = f'{prefix}_{img.name}'
            new_lbl = f'{prefix}_{img.stem}.txt'

            dst_img = f'{OUT}/images/{split}/{new_img}'
            dst_lbl = f'{OUT}/labels/{split}/{new_lbl}'

            if class_map:
                ok = remap_label(lbl, class_map, dst_lbl)
                if not ok:
                    continue
            else:
                shutil.copy2(str(lbl), dst_lbl)

            shutil.copy2(str(img), dst_img)
            total += 1

    print(f'✅ {prefix}: {total} images copiées')
    return total

# ── SOURCE 1 : VisDrone ───────────────────────
copy_source(
    f'{BASE}/visdrone_yolo/images',
    f'{BASE}/visdrone_yolo/labels',
    'visdrone'
    # Pas de remapping (déjà correct)
)

# ── SOURCE 2 : Emergency Vehicles ────────────
for split, folder in [('train','train'),
                      ('val','valid'),
                      ('test','test')]:
    img_dir = f'{BASE}/Emergency-Vehicles-12/{folder}/images'
    lbl_dir = f'{BASE}/Emergency-Vehicles-12/{folder}/labels'
    if not os.path.exists(img_dir):
        continue
    copy_source(img_dir, lbl_dir,
                f'emerg_{split}', EMERGENCY_MAP)

# ── SOURCE 3 : Lqliaa ────────────────────────
for split, folder in [('train','train'),
                      ('val','valid'),
                      ('test','test')]:
    img_dir = f'{BASE}/Lqliaa-Traffic/{folder}/images'
    lbl_dir = f'{BASE}/Lqliaa-Traffic/{folder}/labels'
    if not os.path.exists(img_dir):
        continue
    copy_source(img_dir, lbl_dir,
                f'lqliaa_{split}', LQLIAA_MAP)

# ── data.yaml ────────────────────────────────
yaml = f"""train: {OUT}/images/train
val:   {OUT}/images/val
test:  {OUT}/images/test

nc: 6
names: ['car', 'bus', 'truck',
        'motorcycle', 'person',
        'emergency_vehicle']
"""

with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(yaml)

print('\n✅ Dataset combiné créé')
print(f'   Chemin : {OUT}')

# Compter
for split in ['train', 'val', 'test']:
    n = len(list(Path(
        f'{OUT}/images/{split}'
    ).glob('*')))
    print(f'   {split}: {n} images')


# Ajoute à la fin de combine_datasets.py
# Redistribuer toutes les images emergency

import glob

# Collecter toutes les images emergency
all_emerg_imgs = []
all_emerg_imgs += glob.glob(f'{OUT}/images/train/emerg_*')
all_emerg_imgs += glob.glob(f'{OUT}/images/val/emerg_*')
all_emerg_imgs += glob.glob(f'{OUT}/images/test/emerg_*')

random.shuffle(all_emerg_imgs)
n = len(all_emerg_imgs)
n_tr  = int(n * 0.70)
n_val = int(n * 0.20)

print(f"\n🔄 Redistribution Emergency : {n} images")

for i, img_path in enumerate(all_emerg_imgs):
    img_path = Path(img_path)
    lbl_path = Path(str(img_path).replace(
        '/images/', '/labels/'
    )).with_suffix('.txt')

    if i < n_tr:
        split = 'train'
    elif i < n_tr + n_val:
        split = 'val'
    else:
        split = 'test'

    # Déplacer vers le bon split
    for folder in ['train', 'val', 'test']:
        src_img = f'{OUT}/images/{folder}/{img_path.name}'
        src_lbl = f'{OUT}/labels/{folder}/{lbl_path.name}'
        if os.path.exists(src_img):
            shutil.move(src_img,
                f'{OUT}/images/{split}/{img_path.name}')
        if os.path.exists(src_lbl):
            shutil.move(src_lbl,
                f'{OUT}/labels/{split}/{lbl_path.name}')

# Recompter
print('\n✅ Dataset final :')
for split in ['train', 'val', 'test']:
    n = len(list(Path(
        f'{OUT}/images/{split}'
    ).glob('*')))
    print(f'   {split}: {n} images')