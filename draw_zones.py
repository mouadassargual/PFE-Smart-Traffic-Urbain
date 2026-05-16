"""
draw_zones.py — Outil interactif de dessin de polygones
Clic gauche : ajouter un point
ENTER       : valider la zone et passer à la suivante
R           : recommencer la zone courante
Q           : quitter
"""
import cv2
import numpy as np

IMAGE = "frame_ne8th_54k.jpg"
ZONES_ORDER = ["N", "E", "S", "W"]
COLORS = {
    "N": (255, 100, 100),
    "E": (100, 100, 255),
    "S": (100, 255, 100),
    "W": (0,   220, 220),
}

frame  = cv2.imread(IMAGE)
if frame is None:
    print(f"Erreur : {IMAGE} introuvable")
    exit()

clone  = frame.copy()
points = []
zones  = {}
zone_idx = 0

def draw_state(img, pts, zone_name):
    disp = img.copy()
    color = COLORS[zone_name]
    # Dessiner les points et lignes
    for i, p in enumerate(pts):
        cv2.circle(disp, p, 6, color, -1)
        if i > 0:
            cv2.line(disp, pts[i-1], p, color, 2)
    if len(pts) > 2:
        cv2.line(disp, pts[-1], pts[0], color, 1)  # fermeture
        overlay = disp.copy()
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], color)
        cv2.addWeighted(overlay, 0.2, disp, 0.8, 0, disp)
    # Instructions
    cv2.rectangle(disp, (0,0), (700,40), (0,0,0), -1)
    cv2.putText(disp,
        f"Zone {zone_name} ({len(pts)} pts) | CLIC=ajouter | ENTER=valider | R=reset | Q=quitter",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    # Zones déjà faites
    for done_name, done_pts in zones.items():
        done_color = COLORS[done_name]
        poly = np.array(done_pts, np.int32)
        cv2.polylines(disp, [poly], True, done_color, 2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(disp, done_name, (cx-15,cy+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, done_color, 3)
    return disp

def mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

cv2.namedWindow("Draw Zones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Draw Zones", 1280, 720)
cv2.setMouseCallback("Draw Zones", mouse_click)

while zone_idx < len(ZONES_ORDER):
    current_zone = ZONES_ORDER[zone_idx]
    points = []

    while True:
        disp = draw_state(clone, points, current_zone)
        cv2.imshow("Draw Zones", disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(points) >= 3:    # ENTER
            zones[current_zone] = points.copy()
            zone_idx += 1
            break
        elif key == ord('r'):                  # Reset
            points = []
        elif key == ord('q'):                  # Quitter
            zone_idx = len(ZONES_ORDER)
            break

cv2.destroyAllWindows()

# Afficher résultat
print("\nZONE_PROFILES['ne8th'] = {")
for name, pts in zones.items():
    arr = np.array(pts, np.int32).tolist()
    print(f'    "{name}": np.array({arr}, np.int32),')
print("}")

# Sauvegarder visualisation finale
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
cv2.imwrite("zones_ne8th_final.jpg", result)
print("\nSauvegarde → zones_ne8th_final.jpg")
