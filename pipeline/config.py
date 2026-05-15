# =============================================================
# config.py — Configuration centrale du pipeline
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# FSA Aït Melloul — Université Ibn Zohr — 2024/2025
# =============================================================

import os
import platform

# ── Détection automatique de la plateforme ────────────────────
IS_RPI = platform.machine() in ['aarch64', 'armv7l']
IS_MAC = platform.system() == 'Darwin'

# ── Chemins des modèles ───────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models_rpi')

if IS_RPI:
    # Pi 5 → INT8 optimisé
    MODEL_PATH = os.path.join(MODELS_DIR, 'yolo26n_int8.onnx')
else:
    # Mac/PC → FP32 standard
    MODEL_PATH = os.path.join(MODELS_DIR, 'yolo26n_fp32.onnx')

# ── Source vidéo ─────────────────────────────────────────────
# 0 = webcam, ou chemin vers fichier vidéo
VIDEO_SOURCE = os.path.join(
    BASE_DIR, '..', 'data', 'videos', 'test.mp4'
)

# ── Classes du modèle ─────────────────────────────────────────
CLASSES = [
    'car',               # 0
    'bus',               # 1
    'truck',             # 2
    'motorcycle',        # 3
    'person',            # 4
    'emergency_vehicle', # 5
]
NUM_CLASSES = len(CLASSES)

# ── Paramètres de détection ───────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25   # Seuil de confiance minimum
IOU_THRESHOLD        = 0.45   # Seuil IoU pour NMS
INPUT_SIZE           = 640    # Taille d'entrée YOLO

# ── Paramètres d'anonymisation ────────────────────────────────
# Classes à anonymiser (piétons uniquement)
CLASSES_TO_ANONYMIZE = [4]    # person
BLUR_KERNEL_SIZE     = (51, 51)  # Kernel gaussien
BLUR_SIGMA           = 0         # Auto-calculé par OpenCV

# ── Paramètres de tracking ────────────────────────────────────
IOU_TRACKING_THRESHOLD = 0.3  # Seuil IoU pour associer tracks
MAX_FRAMES_LOST        = 10   # Frames avant suppression track

# ── Zones de l'intersection ───────────────────────────────────
# Format : (x1, y1, x2, y2) en pixels normalisés (0-1)
# À calibrer selon la caméra réelle
ZONES = {
    'North' : (0.2, 0.0, 0.8, 0.4),
    'South' : (0.2, 0.6, 0.8, 1.0),
    'East'  : (0.6, 0.2, 1.0, 0.8),
    'West'  : (0.0, 0.2, 0.4, 0.8),
    'Pedestrian': (0.3, 0.3, 0.7, 0.7),
}

# ── Paramètres MDP ───────────────────────────────────────────
# Coefficients de pondération
W_VEHICLE    = 1.0     # Poids véhicule standard
W_PEDESTRIAN = 1.5     # Priorité piétons
W_EMERGENCY  = 100.0   # Priorité absolue urgence

# Durées des phases (secondes)
PHASE_DURATION_HIGH   = 45  # Densité élevée (>10 véhicules)
PHASE_DURATION_MEDIUM = 30  # Densité moyenne (5-10 véhicules)
PHASE_DURATION_LOW    = 15  # Densité faible (<5 véhicules)

# Seuils de densité
DENSITY_HIGH   = 10
DENSITY_MEDIUM = 5

# ── Paramètres Dashboard ─────────────────────────────────────
DASHBOARD_HOST = '0.0.0.0'
DASHBOARD_PORT = 5001
STREAM_FPS     = 10    # FPS du stream MJPEG

# ── Paramètres MLOps ─────────────────────────────────────────
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# ── Affichage ─────────────────────────────────────────────────
# Couleurs BGR pour OpenCV
COLORS = {
    'car'               : (0,   255,  0  ),  # Vert
    'bus'               : (255, 0,    0  ),  # Bleu
    'truck'             : (0,   0,    255),  # Rouge
    'motorcycle'        : (255, 165,  0  ),  # Orange
    'person'            : (255, 0,    255),  # Magenta
    'emergency_vehicle' : (0,   255,  255),  # Cyan
}

# ── Validation au démarrage ───────────────────────────────────
def validate_config():
    """Vérifie que tous les fichiers nécessaires existent."""
    errors = []

    if not os.path.exists(MODEL_PATH):
        errors.append(f"Modèle non trouvé : {MODEL_PATH}")

    if isinstance(VIDEO_SOURCE, str):
        if not os.path.exists(VIDEO_SOURCE):
            errors.append(f"Vidéo non trouvée : {VIDEO_SOURCE}")

    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False

    print(f"✅ Config validée")
    print(f"   Plateforme : {'Raspberry Pi 5' if IS_RPI else 'Mac/PC'}")
    print(f"   Modèle     : {os.path.basename(MODEL_PATH)}")
    print(f"   Source     : {VIDEO_SOURCE}")
    return True

if __name__ == '__main__':
    validate_config()
