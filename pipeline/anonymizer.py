# =============================================================
# anonymizer.py — Module M4 : Anonymisation Privacy-by-Design
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import numpy as np
import logging
from config import CLASSES_TO_ANONYMIZE, BLUR_KERNEL_SIZE, BLUR_SIGMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Anonymizer:
    """
    Module M4 — Anonymisation temps réel par floutage gaussien.
    Appliqué AVANT le tracking (Privacy-by-Design).
    """

    def __init__(self,
                 classes_to_anonymize=CLASSES_TO_ANONYMIZE,
                 kernel_size=BLUR_KERNEL_SIZE,
                 sigma=BLUR_SIGMA):
        self.classes_to_anonymize = classes_to_anonymize
        self.kernel_size          = kernel_size
        self.sigma                = sigma
        self.anonymized_count     = 0
        logger.info(f"✅ Anonymizer initialisé")
        logger.info(f"   Classes anonymisées : "
                    f"{[CLASSES[i] if i < len(CLASSES) else i for i in classes_to_anonymize]}")

    def anonymize(self, frame, detections):
        """
        Applique le floutage gaussien sur les détections
        des classes sensibles.

        Args:
            frame      : image BGR originale
            detections : liste de Detection

        Returns:
            frame_anonymized : image avec zones floutées
            count            : nombre d'objets anonymisés
        """
        frame_out = frame.copy()
        count     = 0

        for det in detections:
            if det.class_id not in self.classes_to_anonymize:
                continue

            x1, y1, x2, y2 = det.bbox

            # Clamp aux dimensions de l'image
            h, w = frame_out.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Extraire la région
            roi = frame_out[y1:y2, x1:x2]

            # Appliquer floutage gaussien
            roi_blurred = cv2.GaussianBlur(
                roi,
                self.kernel_size,
                self.sigma
            )

            # Réinsérer la région floutée
            frame_out[y1:y2, x1:x2] = roi_blurred
            count += 1

        self.anonymized_count += count
        return frame_out, count

    def get_stats(self):
        return {'total_anonymized': self.anonymized_count}


# ── Config locale ─────────────────────────────────────────────
try:
    from config import CLASSES
except ImportError:
    CLASSES = ['car', 'bus', 'truck',
               'motorcycle', 'person', 'emergency_vehicle']


# ── Test standalone ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from acquisition import VideoCapture
    from detection   import Detector
    from config      import VIDEO_SOURCE, MODEL_PATH

    source = sys.argv[1] if len(sys.argv) > 1 else VIDEO_SOURCE

    print(f"\n🔒 Test M4 — Anonymisation Privacy-by-Design")
    print(f"   Source : {source}\n")

    cap      = VideoCapture(source)
    detector = Detector(MODEL_PATH)
    anon     = Anonymizer()

    for frame in cap:
        detections          = detector.detect(frame)
        frame_anon, count   = anon.anonymize(frame, detections)
        frame_out           = detector.draw(frame_anon, detections)

        cv2.putText(frame_out,
                    f"Anonymises: {count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow('M4 — Anonymisation', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Total anonymisés : {anon.anonymized_count}")