import os
from pathlib import Path

# Mapping VisDrone → nos classes
CLASS_MAP = {
    1: 4,   # pedestrian → person (index 4)
    2: 4,   # people → person
    4: 0,   # car → car (index 0)
    5: 0,   # van → car
    6: 2,   # truck → truck (index 2)
    9: 1,   # bus → bus (index 1)
    10: 3,  # motor → motorcycle (index 3)
}

# Nos 6 classes dans l'ordre
# 0:car, 1:bus, 2:truck, 3:motorcycle,
# 4:person, 5:emergency_vehicle

def convert_visdrone_to_yolo(ann_path, img_w, img_h):
    yolo_lines = []

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            x, y, w, h = map(int, parts[:4])
            score    = int(parts[4])
            category = int(parts[5])

            # Ignorer score=0 et classes non mappées
            if score == 0:
                continue
            if category not in CLASS_MAP:
                continue

            yolo_class = CLASS_MAP[category]

            # Convertir en format YOLO normalisé
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm   = w / img_w
            h_norm   = h / img_h

            yolo_lines.append(
                f"{yolo_class} {x_center:.6f} "
                f"{y_center:.6f} {w_norm:.6f} "
                f"{h_norm:.6f}"
            )

    return yolo_lines


def convert_dataset(images_dir, ann_dir, output_dir):
    import cv2

    os.makedirs(output_dir + '/images', exist_ok=True)
    os.makedirs(output_dir + '/labels', exist_ok=True)

    ann_dir    = Path(ann_dir)
    images_dir = Path(images_dir)
    converted  = 0
    skipped    = 0

    for ann_file in ann_dir.glob('*.txt'):
        img_file = images_dir / (ann_file.stem + '.jpg')

        if not img_file.exists():
            skipped += 1
            continue

        # Lire dimensions image
        img = cv2.imread(str(img_file))
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        # Convertir annotations
        yolo_lines = convert_visdrone_to_yolo(
            ann_file, img_w, img_h
        )

        if not yolo_lines:
            skipped += 1
            continue

        # Sauvegarder
        import shutil
        shutil.copy2(
            str(img_file),
            output_dir + '/images/' + img_file.name
        )

        out_label = output_dir + '/labels/' + ann_file.name
        with open(out_label, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    print(f'✅ Convertis : {converted}')
    print(f'⚠️ Ignorés  : {skipped}')


# Lancer la conversion
convert_dataset(
    images_dir = 'VisDrone2019-DET-train/images',
    ann_dir    = 'VisDrone2019-DET-train/annotations',
    output_dir = 'visdrone_yolo'
)