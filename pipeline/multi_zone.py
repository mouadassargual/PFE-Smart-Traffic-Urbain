# =============================================================
# multi_zone.py — Test pipeline 4 zones en parallèle
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import time
import threading
import argparse
import logging
from config import MODEL_PATH

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('multi_zone')

# ── État partagé entre tous les threads ──────────────────────
shared_state = {
    'zone_counts': {
        'North'     : {},
        'South'     : {},
        'East'      : {},
        'West'      : {},
        'Pedestrian': {},
    },
    'zone_frames': {
        'North': None, 'South': None,
        'East' : None, 'West' : None,
    },
    'stats': {
        'total_detections': 0,
        'total_frames'    : 0,
        'fps'             : 0.0,
        'uptime'          : 0,
    },
    'decision': {
        'phase'   : 'All-Red',
        'duration': 15,
        'reason'  : 'Initialisation...',
        'lights'  : {
            'North': 'red', 'South': 'red',
            'East' : 'red', 'West' : 'red',
            'Pedestrian': 'red'
        },
        'time_remaining': 15,
    },
    'start_time': time.time(),
}
lock = threading.Lock()


def zone_worker(zone_name, video_path,
                detector, anon, tracker):
    """
    Thread pour chaque zone.
    Lit la vidéo en boucle et met à jour l'état partagé.
    """
    from acquisition import VideoCapture

    logger.info(f"▶️  Zone {zone_name} démarrée : {video_path}")

    frame_count = 0

    while True:  # Boucle infinie = vidéo en boucle
        cap = VideoCapture(video_path)

        for frame in cap:
            frame_count += 1
            t0 = time.time()

            # M2 — Détection
            detections = detector.detect(frame)

            # M4 — Anonymisation
            frame_anon, _ = anon.anonymize(frame, detections)

            # M3 — Tracking
            tracks      = tracker.update(detections)
            zone_counts = tracker.count_by_zone(
                tracks, frame.shape
            )

            # Annoter la frame
            frame_out = detector.draw(frame_anon, detections)
            _draw_zone_overlay(
                frame_out, zone_name,
                zone_counts.get(zone_name, {}),
                shared_state['decision'].get('lights', {})
            )

            # Mettre à jour l'état partagé
            with lock:
                shared_state['zone_frames'][zone_name] = \
                    frame_out.copy()
                shared_state['zone_counts'][zone_name] = \
                    zone_counts.get(zone_name, {})
                shared_state['stats']['total_detections'] += \
                    len(detections)
                shared_state['stats']['total_frames'] += 1
                shared_state['stats']['uptime'] = int(
                    time.time() - shared_state['start_time']
                )

        cap.release()
        # Redémarre la vidéo depuis le début


def decision_worker(decision_module, dashboard,
                    interval=30):
    """
    Thread de décision — se déclenche toutes les N frames.
    Lit les comptages de toutes les zones et prend une décision.
    """
    frame_counter = 0

    while True:
        time.sleep(0.1)
        frame_counter += 1

        if frame_counter % interval == 0:
            with lock:
                zone_counts = dict(shared_state['zone_counts'])

            # M5 — Décision MDP
            dec = decision_module.decide(zone_counts)
            dec['time_remaining'] = decision_module.time_remaining()

            with lock:
                shared_state['decision'] = dec

            logger.info(
                f"🚦 Décision : {dec['phase']} "
                f"({dec['duration']}s) — {dec['reason']}"
            )

        # M6 — Mettre à jour dashboard
        with lock:
            frames   = dict(shared_state['zone_frames'])
            counts   = dict(shared_state['zone_counts'])
            decision = dict(shared_state['decision'])
            stats    = dict(shared_state['stats'])

        # Calculer FPS global
        uptime = stats.get('uptime', 1) or 1
        fps    = stats['total_frames'] / uptime
        stats['fps'] = fps
        dashboard.update_stats(
            stats['total_detections'], fps
        )

        for zone_name, frame in frames.items():
            if frame is not None:
                dashboard.update(
                    zone_name, frame,
                    counts.get(zone_name, {}),
                    decision
                )


def _draw_zone_overlay(frame, zone_name,
                        counts, lights):
    """Overlay sur chaque frame de zone."""
    # Fond semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (200, 70),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Nom zone
    cv2.putText(frame, f"Zone: {zone_name}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)

    # Compteurs
    veh = (counts.get('car', 0) +
           counts.get('bus', 0) +
           counts.get('truck', 0) +
           counts.get('motorcycle', 0))
    ped = counts.get('person', 0)
    emg = counts.get('emergency_vehicle', 0)

    cv2.putText(frame,
                f"Veh:{veh} Ped:{ped} Urg:{emg}",
                (8, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    # Indicateur feu
    color_map = {
        'green' : (0, 200, 0),
        'red'   : (0, 0, 200),
        'orange': (0, 165, 255),
    }
    light_color = lights.get(zone_name, 'red')
    bgr         = color_map.get(light_color,
                                (0, 0, 200))

    cv2.circle(frame, (185, 15), 12, bgr, -1)
    cv2.circle(frame, (185, 15), 12,
               (255, 255, 255), 1)


def run_multi_zone(videos_dict, model=MODEL_PATH):
    """
    Lance le pipeline complet pour 4 zones.

    Args:
        videos_dict : {zone: chemin_video}
        model       : chemin modèle ONNX
    """
    from detection  import Detector
    from anonymizer import Anonymizer
    from tracker    import IoUTracker
    from decision   import TrafficDecision
    from dashboard  import Dashboard

    logger.info("🚀 Démarrage pipeline multi-zones")
    logger.info(f"   Zones : {list(videos_dict.keys())}")

    # Modules partagés entre zones
    # Note: Detector est thread-safe en lecture
    detector = Detector(model)
    decision_module = TrafficDecision()
    dashboard       = Dashboard()

    # Dashboard en arrière-plan
    dashboard.run_background()
    time.sleep(1)
    logger.info("🌐 Dashboard : http://localhost:5000")

    # Lancer un thread par zone
    threads = []
    for zone_name, video_path in videos_dict.items():
        if video_path is None:
            logger.warning(
                f"⚠️  Zone {zone_name} : pas de vidéo"
            )
            continue

        anon    = Anonymizer()
        tracker = IoUTracker()

        t = threading.Thread(
            target=zone_worker,
            args=(zone_name, video_path,
                  detector, anon, tracker),
            daemon=True
        )
        t.start()
        threads.append(t)
        logger.info(f"✅ Thread {zone_name} démarré")

    # Thread de décision
    t_dec = threading.Thread(
        target=decision_worker,
        args=(decision_module, dashboard, 30),
        daemon=True
    )
    t_dec.start()

    logger.info("\n✅ Tous les threads démarrés")
    logger.info("   Ouvre http://localhost:5000")
    logger.info("   Ctrl+C pour arrêter\n")

    # Garder le programme actif
    try:
        while True:
            time.sleep(1)

            # Log stats toutes les 10s
            with lock:
                s = shared_state['stats']
                d = shared_state['decision']

            if s['uptime'] % 10 == 0 and s['uptime'] > 0:
                fps = s['total_frames'] / max(s['uptime'], 1)
                logger.info(
                    f"📊 Stats : {s['total_frames']} frames | "
                    f"{fps:.1f} FPS | "
                    f"Phase: {d['phase']}"
                )

    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt du pipeline")


# ── CLI ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smart Traffic — Pipeline 4 zones'
    )
    parser.add_argument('--north', default=None,
                        help='Vidéo zone Nord')
    parser.add_argument('--south', default=None,
                        help='Vidéo zone Sud')
    parser.add_argument('--east',  default=None,
                        help='Vidéo zone Est')
    parser.add_argument('--west',  default=None,
                        help='Vidéo zone Ouest')
    parser.add_argument('--model', default=MODEL_PATH,
                        help='Modèle ONNX')
    parser.add_argument('--demo',  action='store_true',
                        help='Mode démo : même vidéo partout')
    parser.add_argument('--video', default=None,
                        help='Vidéo unique (mode démo)')

    args = parser.parse_args()

    # Mode démo : même vidéo pour les 4 zones
    if args.demo and args.video:
        videos = {
            'North': args.video,
            'South': args.video,
            'East' : args.video,
            'West' : args.video,
        }
        logger.info(f"🎭 Mode DÉMO — vidéo : {args.video}")

    else:
        videos = {
            'North': args.north,
            'South': args.south,
            'East' : args.east,
            'West' : args.west,
        }
        # Garder seulement les zones avec vidéo
        videos = {z: v for z, v in videos.items()
                  if v is not None}

        if not videos:
            print("❌ Aucune vidéo fournie.")
            print("\nUsage :")
            print("  Mode démo (1 vidéo) :")
            print("  python multi_zone.py --demo "
                  "--video ma_video.mp4")
            print("\n  Mode complet (4 vidéos) :")
            print("  python multi_zone.py "
                  "--north n.mp4 --south s.mp4 "
                  "--east e.mp4 --west w.mp4")
            exit(1)

    run_multi_zone(videos, model=args.model)