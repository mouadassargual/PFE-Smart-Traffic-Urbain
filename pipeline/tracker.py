# =============================================================
# tracker.py — Module M3 : Tracking IoU + Comptage par zone
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import numpy as np
import logging
from config import (
    IOU_TRACKING_THRESHOLD, MAX_FRAMES_LOST,
    ZONES, CLASSES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Track:
    """Représente un objet suivi."""
    _id_counter = 0

    def __init__(self, detection):
        Track._id_counter += 1
        self.id         = Track._id_counter
        self.bbox       = detection.bbox
        self.class_id   = detection.class_id
        self.class_name = detection.class_name
        self.confidence = detection.confidence
        self.frames_lost = 0
        self.history    = [detection.bbox]

    def update(self, detection):
        self.bbox       = detection.bbox
        self.confidence = detection.confidence
        self.frames_lost = 0
        self.history.append(detection.bbox)

    def mark_lost(self):
        self.frames_lost += 1

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class IoUTracker:
    """
    Module M3 — Tracker IoU léger.
    Pas de features visuelles → compatible Privacy-by-Design.
    """

    def __init__(self,
                 iou_threshold=IOU_TRACKING_THRESHOLD,
                 max_frames_lost=MAX_FRAMES_LOST):
        self.iou_threshold  = iou_threshold
        self.max_frames_lost = max_frames_lost
        self.tracks         = []
        self.frame_id       = 0
        logger.info("✅ IoU Tracker initialisé")

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
        area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """
        Met à jour les tracks avec les nouvelles détections.

        Args:
            detections : liste de Detection

        Returns:
            active_tracks : liste des tracks actifs
        """
        self.frame_id += 1

        if not self.tracks:
            for det in detections:
                self.tracks.append(Track(det))
            return self.tracks.copy()

        # Matrice IoU
        matched_tracks = set()
        matched_dets   = set()

        for i, track in enumerate(self.tracks):
            best_iou  = self.iou_threshold
            best_det  = None
            best_j    = None

            for j, det in enumerate(detections):
                if j in matched_dets:
                    continue
                if det.class_id != track.class_id:
                    continue

                iou = self._compute_iou(track.bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det
                    best_j   = j

            if best_det is not None:
                track.update(best_det)
                matched_tracks.add(i)
                matched_dets.add(best_j)

        # Marquer les tracks non associés
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_lost()

        # Créer nouveaux tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(det))

        # Supprimer tracks perdus
        self.tracks = [
            t for t in self.tracks
            if t.frames_lost <= self.max_frames_lost
        ]

        return self.tracks.copy()

    def count_by_zone(self, tracks, frame_shape):
        """
        Compte les objets par zone et par classe.

        Args:
            tracks      : liste de Track actifs
            frame_shape : (height, width) de l'image

        Returns:
            dict : {zone_name: {class_name: count}}
        """
        h, w = frame_shape[:2]
        counts = {
            zone: {cls: 0 for cls in CLASSES}
            for zone in ZONES
        }

        for track in tracks:
            cx, cy = track.center
            cx_norm = cx / w
            cy_norm = cy / h

            for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
                if (zx1 <= cx_norm <= zx2 and
                        zy1 <= cy_norm <= zy2):
                    counts[zone_name][track.class_name] += 1

        return counts

    def get_density_per_direction(self, zone_counts):
        """
        Calcule la densité de trafic par direction.

        Returns:
            dict : {direction: density_score}
        """
        vehicle_classes = [
            'car', 'bus', 'truck', 'motorcycle'
        ]
        density = {}

        for zone, class_counts in zone_counts.items():
            if zone == 'Pedestrian':
                continue
            density[zone] = sum(
                class_counts.get(cls, 0)
                for cls in vehicle_classes
            )

        return density

    def reset(self):
        self.tracks = []
        Track._id_counter = 0


# ── Test standalone ──────────────────────────────────────────
if __name__ == '__main__':
    import sys, cv2
    from acquisition import VideoCapture
    from detection   import Detector
    from anonymizer  import Anonymizer
    from config      import VIDEO_SOURCE, MODEL_PATH, COLORS

    source = sys.argv[1] if len(sys.argv) > 1 else VIDEO_SOURCE

    print(f"\n📍 Test M3 — Tracking IoU")
    print(f"   Source : {source}\n")

    cap      = VideoCapture(source)
    detector = Detector(MODEL_PATH)
    anon     = Anonymizer()
    tracker  = IoUTracker()

    for frame in cap:
        detections         = detector.detect(frame)
        frame_anon, _      = anon.anonymize(frame, detections)
        tracks             = tracker.update(detections)
        zone_counts        = tracker.count_by_zone(
                               tracks, frame.shape)
        density            = tracker.get_density_per_direction(
                               zone_counts)

        frame_out = frame_anon.copy()

        # Dessiner tracks avec ID
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = COLORS.get(track.class_name,
                                (255, 255, 255))
            cv2.rectangle(frame_out,
                          (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_out,
                        f"ID:{track.id} {track.class_name}",
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1)

        # Stats
        y = 30
        for zone, d in density.items():
            cv2.putText(frame_out,
                        f"{zone}: {d} veh",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            y += 20

        cv2.imshow('M3 — Tracking', frame_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()