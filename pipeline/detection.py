# =============================================================
# detection.py — Module M2 : Détection YOLO26n
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO

# Import local car on est dans le dossier pipeline
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Detection:
    """Représente un objet détecté."""
    def __init__(self, bbox, class_id, confidence, class_name):
        self.bbox        = bbox         # [x1, y1, x2, y2]
        self.class_id    = class_id
        self.confidence  = confidence
        self.class_name  = class_name
        self.center      = (
            int((bbox[0] + bbox[2]) / 2),
            int((bbox[1] + bbox[3]) / 2)
        )

    def __repr__(self):
        return (f"Detection({self.class_name}, "
                f"conf={self.confidence:.2f})")


class Detector:
    """
    Module M2 — Détection d'objets via YOLO26n ONNX.
    """

    def __init__(self, model_path=config.MODEL_PATH):
        logger.info(f"⏳ Chargement modèle : {model_path}")
        self.model      = YOLO(model_path)
        self.classes    = config.CLASSES
        self.conf       = config.CONFIDENCE_THRESHOLD
        self.iou        = config.IOU_THRESHOLD
        self.input_size = config.INPUT_SIZE
        self.inf_times  = []
        logger.info("✅ Modèle chargé")

    def detect(self, frame):
        """
        Détecte les objets dans une frame.

        Args:
            frame : image BGR (numpy array)

        Returns:
            list[Detection] : liste des détections
        """
        t0      = time.time()
        results = self.model(
            frame,
            imgsz   = self.input_size,
            conf    = self.conf,
            iou     = self.iou,
            verbose = False
        )
        inf_time = (time.time() - t0) * 1000
        self.inf_times.append(inf_time)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id    = int(box.cls[0])
                confidence  = float(box.conf[0])
                class_name  = self.classes[class_id]

                detections.append(Detection(
                    bbox       = [x1, y1, x2, y2],
                    class_id   = class_id,
                    confidence = confidence,
                    class_name = class_name
                ))

        return detections

    def draw(self, frame, detections):
        """
        Dessine les bounding boxes sur la frame.

        Args:
            frame      : image BGR originale
            detections : liste de Detection

        Returns:
            frame avec annotations
        """
        frame_out = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color  = config.COLORS.get(det.class_name, (255, 255, 255))
            label  = f"{det.class_name} {det.confidence:.2f}"

            # Bounding box
            cv2.rectangle(frame_out, (x1, y1), (x2, y2),
                          color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame_out,
                          (x1, y1 - th - 6),
                          (x1 + tw, y1),
                          color, -1)

            # Label texte
            cv2.putText(frame_out, label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)

        # Stats en haut
        avg_inf = np.mean(self.inf_times[-30:]) \
                  if self.inf_times else 0
        fps_inf = 1000 / avg_inf if avg_inf > 0 else 0

        cv2.putText(frame_out,
                    f"Detections: {len(detections)} | "
                    f"Latence: {avg_inf:.0f}ms | "
                    f"FPS: {fps_inf:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        return frame_out

    def get_stats(self):
        """Retourne les statistiques d'inférence."""
        if not self.inf_times:
            return {}
        return {
            'avg_latency_ms' : round(np.mean(self.inf_times), 2),
            'min_latency_ms' : round(np.min(self.inf_times), 2),
            'max_latency_ms' : round(np.max(self.inf_times), 2),
            'avg_fps'        : round(
                1000 / np.mean(self.inf_times), 2
            ),
            'total_frames'   : len(self.inf_times),
        }

    def count_by_class(self, detections):
        """Compte les détections par classe."""
        counts = {cls: 0 for cls in self.classes}
        for det in detections:
            counts[det.class_name] += 1
        return counts


# ── Test standalone ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from acquisition import VideoCapture

    source = sys.argv[1] if len(sys.argv) > 1 else config.VIDEO_SOURCE

    print(f"\n🔍 Test M2 — Détection YOLO26n")
    print(f"   Modèle : {config.MODEL_PATH}")
    print(f"   Source : {source}\n")

    cap      = VideoCapture(source)
    detector = Detector()

    for frame in cap:
        detections = detector.detect(frame)
        counts     = detector.count_by_class(detections)
        frame_out  = detector.draw(frame, detections)

        cv2.imshow('M2 — Detection', frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    stats = detector.get_stats()
    print(f"\n✅ Statistiques détection :")
    for k, v in stats.items():
        print(f"   {k:20s} : {v}")
