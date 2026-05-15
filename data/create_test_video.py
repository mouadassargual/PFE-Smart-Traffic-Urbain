import cv2
import os
from pathlib import Path

# Chemin vers les images du dataset d'entraînement
IMG_DIR = "/Users/mouadassargual/Desktop/smart-traffic-agadir/dataset_v2/images/train"
OUT_VIDEO = "/Users/mouadassargual/Desktop/smart-traffic-agadir/data/videos/test.mp4"

# Récupérer 200 images
imgs = sorted(Path(IMG_DIR).glob("*.jpg"))[:200]

if len(imgs) == 0:
    print("❌ Aucune image trouvée dans", IMG_DIR)
    exit(1)

print(f"🎬 Création de la vidéo depuis {len(imgs)} images...")

out = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10,  # 10 FPS
    (640, 640)
)

for img_path in imgs:
    img = cv2.imread(str(img_path))
    if img is not None:
        img = cv2.resize(img, (640, 640))
        out.write(img)

out.release()
print(f"✅ Vidéo de test créée avec succès : {OUT_VIDEO}")
