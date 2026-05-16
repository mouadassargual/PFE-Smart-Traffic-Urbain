import cv2
import numpy as np

ZONES = {
    "N": np.array([[480,50],[750,50],[760,240],[490,240]], np.int32),
    "E": np.array([[840,220],[1280,180],[1280,520],[1100,520],[840,350]], np.int32),
    "S": np.array([[300,560],[900,540],[1000,720],[150,720]], np.int32),
    "W": np.array([[0,280],[430,220],[430,600],[350,620],[0,580]], np.int32),
}
COLORS = {"N":(255,100,100),"S":(100,255,100),"E":(100,100,255),"W":(255,255,100)}

frame = cv2.imread("frame_ne8th_54k.jpg")
if frame is None:
    print("Erreur : frame_ne8th_54k.jpg introuvable")
    exit()

overlay = frame.copy()
for name, poly in ZONES.items():
    cv2.fillPoly(overlay, [poly], COLORS[name])
cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
for name, poly in ZONES.items():
    cv2.polylines(frame, [poly], True, COLORS[name], 2)
    M = cv2.moments(poly)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        cv2.putText(frame, name, (cx-15, cy+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS[name], 3)

cv2.imwrite("zones_ne8th_check.jpg", frame)
print("OK → zones_ne8th_check.jpg")
