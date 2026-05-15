# =============================================================
# main.py — Pipeline principal M1→M2→M4→M3→M5→M6
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import time
import threading
import argparse
import logging
from config import validate_config, VIDEO_SOURCE, MODEL_PATH

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [%(name)s] %(message)s',
    datefmt = '%H:%M:%S'
)
logger = logging.getLogger('main')


def run_pipeline(source=VIDEO_SOURCE,
                 model=MODEL_PATH,
                 zone='North',
                 show=True):
    """
    Lance le pipeline complet pour une zone.

    M1 → M2 → M4 → M3 → M5 → M6
    """
    from acquisition import VideoCapture
    from detection   import Detector
    from anonymizer  import Anonymizer
    from tracker     import IoUTracker
    from decision    import TrafficDecision
    from dashboard   import Dashboard

    logger.info(f"🚀 Démarrage pipeline — Zone : {zone}")

    # ── Initialisation modules ────────────────────────────────
    cap      = VideoCapture(source)        # M1
    detector = Detector(model)             # M2
    anon     = Anonymizer()                # M4
    tracker  = IoUTracker()                # M3
    decision = TrafficDecision()           # M5
    dash     = Dashboard()                 # M6

    # Lancer dashboard en arrière-plan
    dash.run_background()
    time.sleep(1)
    logger.info(f"🌐 Dashboard : http://localhost:5000")

    # ── Boucle principale ─────────────────────────────────────
    frame_count    = 0
    total_det      = 0
    t_start        = time.time()
    decision_cycle = 30  # Prendre décision toutes les 30 frames

    # État pour décision courante
    current_decision = {
        'phase'   : 'All-Red',
        'duration': 15,
        'reason'  : 'Initialisation',
        'lights'  : {
            'North': 'red', 'South': 'red',
            'East' : 'red', 'West' : 'red',
            'Pedestrian': 'red'
        },
        'time_remaining': 15,
    }
    zone_counts_cumul = {
        'North'     : {c: 0 for c in
                       ['car','bus','truck','motorcycle',
                        'person','emergency_vehicle']},
        'South'     : {c: 0 for c in
                       ['car','bus','truck','motorcycle',
                        'person','emergency_vehicle']},
        'East'      : {c: 0 for c in
                       ['car','bus','truck','motorcycle',
                        'person','emergency_vehicle']},
        'West'      : {c: 0 for c in
                       ['car','bus','truck','motorcycle',
                        'person','emergency_vehicle']},
        'Pedestrian': {c: 0 for c in
                       ['car','bus','truck','motorcycle',
                        'person','emergency_vehicle']},
    }

    logger.info("🎬 Pipeline démarré. Appuyez sur Q pour quitter.")

    for frame in cap:
        frame_count += 1
        t_frame = time.time()

        # ── M2 : Détection ────────────────────────────────────
        detections = detector.detect(frame)
        total_det += len(detections)

        # ── M4 : Anonymisation (Privacy-by-Design) ────────────
        frame_anon, anon_count = anon.anonymize(
            frame, detections
        )

        # ── M3 : Tracking + Comptage par zone ─────────────────
        tracks      = tracker.update(detections)
        zone_counts = tracker.count_by_zone(
            tracks, frame.shape
        )

        # Accumuler pour décision
        for z in zone_counts:
            for c in zone_counts[z]:
                zone_counts_cumul[z][c] = max(
                    zone_counts_cumul[z][c],
                    zone_counts[z][c]
                )

        # ── M5 : Décision MDP (toutes les N frames) ───────────
        if frame_count % decision_cycle == 0:
            dec = decision.decide(zone_counts_cumul)
            current_decision = dec
            current_decision['time_remaining'] = \
                decision.time_remaining()

            # Reset compteurs
            for z in zone_counts_cumul:
                for c in zone_counts_cumul[z]:
                    zone_counts_cumul[z][c] = 0

            logger.info(
                f"📋 Décision : {dec['phase']} "
                f"({dec['duration']}s) — {dec['reason']}"
            )

        # ── Annoter la frame ──────────────────────────────────
        frame_out = detector.draw(frame_anon, detections)

        # Overlay feux
        _draw_lights_overlay(
            frame_out,
            current_decision.get('lights', {}),
            zone
        )

        # Overlay stats
        elapsed = time.time() - t_start
        fps     = frame_count / elapsed if elapsed > 0 else 0
        _draw_stats_overlay(
            frame_out, frame_count,
            len(detections), fps,
            anon_count, zone
        )

        # ── M6 : Dashboard ────────────────────────────────────
        zone_cls_counts = zone_counts.get(zone, {})
        dash.update(zone, frame_out,
                    zone_cls_counts, current_decision)
        dash.update_stats(total_det, fps)

        # ── Affichage local ───────────────────────────────────
        if show:
            cv2.imshow(f'Smart Traffic — {zone}', frame_out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Fin ───────────────────────────────────────────────────
    cap.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*50}")
    logger.info(f"✅ Pipeline terminé")
    logger.info(f"   Frames      : {frame_count}")
    logger.info(f"   Détections  : {total_det}")
    logger.info(f"   Durée       : {elapsed:.1f}s")
    logger.info(f"   FPS moyen   : {frame_count/elapsed:.1f}")
    det_stats = detector.get_stats()
    logger.info(f"   Latence moy : {det_stats.get('avg_latency_ms', 0):.1f}ms")
    logger.info(f"{'='*50}")


def _draw_lights_overlay(frame, lights, zone):
    """Dessine l'état des feux en overlay."""
    h, w = frame.shape[:2]
    colors_bgr = {
        'green' : (0,   200,  0  ),
        'red'   : (0,   0,    200),
        'orange': (0,   165,  255),
    }
    zone_color = lights.get(zone, 'red')
    color_bgr  = colors_bgr.get(zone_color, (0, 0, 200))

    # Cercle feu en haut à droite
    cx, cy = w - 50, 50
    cv2.circle(frame, (cx, cy), 20, color_bgr, -1)
    cv2.circle(frame, (cx, cy), 20, (255, 255, 255), 2)
    cv2.putText(frame, zone_color.upper(),
                (cx - 18, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1)


def _draw_stats_overlay(frame, frame_id, n_det,
                         fps, anon_count, zone):
    """Dessine les statistiques en overlay."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 90),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        f"Zone: {zone}",
        f"Frame: {frame_id} | FPS: {fps:.1f}",
        f"Detections: {n_det} | Anon: {anon_count}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)


# ── CLI ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smart Traffic Agadir — Pipeline'
    )
    parser.add_argument('--source', default=VIDEO_SOURCE,
                        help='Source vidéo')
    parser.add_argument('--model',  default=MODEL_PATH,
                        help='Modèle ONNX')
    parser.add_argument('--zone',   default='North',
                        choices=['North','South','East','West'],
                        help='Zone de l intersection')
    parser.add_argument('--no-show', action='store_true',
                        help='Sans affichage local')

    args = parser.parse_args()

    # Valider configuration
    if not validate_config():
        print("❌ Configuration invalide. Arrêt.")
        exit(1)

    run_pipeline(
        source = args.source,
        model  = args.model,
        zone   = args.zone,
        show   = not args.no_show,
    )