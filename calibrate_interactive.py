# calibrate_interactive.py
import cv2
import numpy as np

frame = cv2.imread("frame_ne8th_54k.jpg")
clone = frame.copy()
points = []
current_zone = "N"
zones = {}

COLORS = {"N":(255,100,100),"S":(100,255,100),
          "E":(100,100,255),"W":(255,255,100)}
ZONE_ORDER = ["N","E","S","W"]
zone_idx = 0

def draw(img, pts, color):
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], 
                     False, color, 2)
    for p in pts:
        cv2.circle(img, p, 5, color, -1)

def mouse_click(event, x, y, flags, param):
    global points, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        frame = clone.copy()
        draw(frame, points, COLORS[current_zone])
        cv2.putText(frame, 
            f"Zone {current_zone} — clic pour points, ENTER pour valider, R pour reset",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imshow("Calibration", frame)

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_click)

while zone_idx < len(ZONE_ORDER):
    current_zone = ZONE_ORDER[zone_idx]
    points = []
    frame = clone.copy()
    cv2.putText(frame,
        f"Clique les coins de la Zone {current_zone} — ENTER pour valider — R pour reset",
        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        COLORS[current_zone], 2)
    cv2.imshow("Calibration", frame)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 and len(points) >= 3:  # ENTER
            zones[current_zone] = points.copy()
            zone_idx += 1
            break
        elif key == ord('r'):  # Reset
            points = []
            frame = clone.copy()
            cv2.putText(frame,
                f"Zone {current_zone} — reset",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, COLORS[current_zone], 2)
            cv2.imshow("Calibration", frame)

cv2.destroyAllWindows()

# Afficher résultat final
print("\nPolygones calibrés :")
print("ZONE_POLYGONS_1280x720 = {")
for name, pts in zones.items():
    arr = np.array(pts, np.int32).tolist()
    print(f'    "{name}": np.array({arr}, np.int32),')
print("}")

# Sauvegarder visualisation
result = clone.copy()
overlay = result.copy()
for name, pts in zones.items():
    poly = np.array(pts, np.int32)
    cv2.fillPoly(overlay, [poly], COLORS[name])
cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
for name, pts in zones.items():
    poly = np.array(pts, np.int32)
    cv2.polylines(result, [poly], True, COLORS[name], 2)
    M = cv2.moments(poly)
    if M["m00"] != 0:
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        cv2.putText(result, name, (cx-15,cy+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS[name], 3)
cv2.imwrite("zones_final.jpg", result)
print("\nSauvegarde → zones_final.jpg")