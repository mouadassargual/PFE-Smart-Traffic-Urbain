# simulate_intersection.py
# Génère 4 vidéos simulées avec tous les cas

import cv2
import numpy as np
import random
import math

def create_simulation_video(
        output_path, scenario, duration=300):
    """
    Crée une vidéo simulée d'intersection.

    scenario : 'dense_north', 'emergency',
               'pedestrians', 'normal'
    """
    w, h   = 640, 480
    fps    = 15
    frames = duration

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    # Couleurs
    ROAD   = (80,  80,  80)
    LINE   = (200, 200, 200)
    GRASS  = (34,  100, 34)
    colors = {
        'car'              : (0,   255,  0  ),
        'bus'              : (255, 0,    0  ),
        'truck'            : (0,   0,    255),
        'motorcycle'       : (255, 165,  0  ),
        'person'           : (255, 0,    255),
        'emergency_vehicle': (0,   255,  255),
    }

    # Objets mobiles
    objects = []

    # Définir scénario
    if scenario == 'dense_north':
        for i in range(8):
            objects.append({
                'type' : 'car',
                'x'    : 280 + random.randint(-20,20),
                'y'    : -50 - i * 60,
                'dx'   : 0, 'dy': 2,
                'w'    : 40, 'h': 60,
            })
        objects.append({
            'type': 'bus', 'x': 320, 'y': -200,
            'dx': 0, 'dy': 1.5,
            'w': 50, 'h': 90,
        })

    elif scenario == 'emergency':
        for i in range(3):
            objects.append({
                'type': 'car',
                'x': 350 + i*50, 'y': 200,
                'dx': 1.5, 'dy': 0,
                'w': 40, 'h': 60,
            })
        objects.append({
            'type': 'emergency_vehicle',
            'x': -80, 'y': 210,
            'dx': 3, 'dy': 0,
            'w': 50, 'h': 70,
            'emergency': True,
        })

    elif scenario == 'pedestrians':
        for i in range(5):
            objects.append({
                'type': 'person',
                'x': 290 + i*10, 'y': -30 - i*20,
                'dx': 0, 'dy': 1,
                'w': 20, 'h': 35,
            })
        for i in range(3):
            objects.append({
                'type': 'car',
                'x': 200 - i*60, 'y': 220,
                'dx': 1, 'dy': 0,
                'w': 40, 'h': 55,
            })

    else:  # normal
        for i in range(4):
            objects.append({
                'type': random.choice(
                    ['car','car','car','motorcycle']),
                'x': 270 + random.randint(-30,30),
                'y': -40 - i * 80,
                'dx': 0, 'dy': 1.5,
                'w': 35, 'h': 55,
            })

    # Générer frames
    for f in range(frames):
        # Fond
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = GRASS

        # Routes
        cv2.rectangle(img, (220, 0), (420, h),
                      ROAD, -1)
        cv2.rectangle(img, (0, 180), (w, 300),
                      ROAD, -1)

        # Lignes de route
        for y in range(0, h, 40):
            cv2.line(img, (315, y), (325, y+20),
                     LINE, 2)
        for x in range(0, w, 40):
            cv2.line(img, (x, 235), (x+20, 245),
                     LINE, 2)

        # Passage piéton
        for i in range(5):
            cv2.rectangle(img,
                (220 + i*20, 300),
                (236 + i*20, 330),
                LINE, -1)

        # Mettre à jour et dessiner objets
        for obj in objects:
            obj['x'] += obj['dx']
            obj['y'] += obj['dy']

            # Reboucler si hors écran
            if obj['dy'] > 0 and obj['y'] > h + 100:
                obj['y'] = -100
            if obj['dx'] > 0 and obj['x'] > w + 100:
                obj['x'] = -100

            x  = int(obj['x'])
            y  = int(obj['y'])
            ow = obj['w']
            oh = obj['h']

            color = colors.get(obj['type'],
                                (200, 200, 200))

            # Clignoter si urgence
            if obj.get('emergency') and f % 10 < 5:
                color = (0, 0, 255)

            cv2.rectangle(img,
                (x, y), (x+ow, y+oh),
                color, -1)
            cv2.rectangle(img,
                (x, y), (x+ow, y+oh),
                (0, 0, 0), 1)

            # Label
            cv2.putText(img, obj['type'][:3],
                (x, y-3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, color, 1)

        # Overlay info
        cv2.putText(img, f"Scenario: {scenario}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)
        cv2.putText(img, f"Frame: {f}/{frames}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

        out.write(img)

    out.release()
    print(f'✅ {output_path} créé ({frames} frames)')


if __name__ == '__main__':
    import os
    os.makedirs('data/videos', exist_ok=True)

    scenarios = [
        ('data/videos/north.mp4', 'dense_north'),
        ('data/videos/south.mp4', 'normal'),
        ('data/videos/east.mp4',  'emergency'),
        ('data/videos/west.mp4',  'pedestrians'),
    ]

    print("🎬 Génération vidéos simulation...\n")
    for path, scenario in scenarios:
        create_simulation_video(path, scenario)

    print("\n✅ 4 vidéos créées dans data/videos/")
    print("   Lance : python dashboard_main.py")