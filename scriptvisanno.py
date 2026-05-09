import cv2
import os
import random
from pathlib import Path

# Afficher 5 images VisDrone
imgs = list(Path(
    '/Users/mouadassargual/Desktop/smart-traffic-agadir/visdrone_yolo/images'
).glob('*.jpg'))

sample = random.sample(imgs, 5)
for img_path in sample:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    print(f'{img_path.name} : {w}×{h}')
    cv2.imshow('VisDrone', img)
    cv2.waitKey(2000)

cv2.destroyAllWindows()