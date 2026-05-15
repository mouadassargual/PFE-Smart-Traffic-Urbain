# =============================================================
# acquisition.py — Module M1 : Acquisition vidéo
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import time
import logging
from config import VIDEO_SOURCE, INPUT_SIZE

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Module M1 — Acquisition vidéo via OpenCV.
    Supporte fichiers .mp4 et flux caméra USB.
    """

    def __init__(self, source=VIDEO_SOURCE):
        """
        Initialise la capture vidéo.

        Args:
            source : chemin fichier vidéo ou index caméra (0)
        """
        self.source   = source
        self.cap      = None
        self.width    = 0
        self.height   = 0
        self.fps      = 0
        self.frame_count = 0
        self._connect()

    def _connect(self):
        """Ouvre la source vidéo."""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"❌ Impossible d'ouvrir la source : {self.source}"
            )

        # Récupérer les propriétés
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS)
        total       = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"✅ Source ouverte : {self.source}")
        logger.info(f"   Résolution : {self.width}×{self.height}")
        logger.info(f"   FPS source : {self.fps:.1f}")
        logger.info(f"   Frames totales : {total}")

    def read(self):
        """
        Lit la prochaine frame.

        Returns:
            frame (np.ndarray) : image BGR ou None si fin
        """
        if self.cap is None:
            return None

        ret, frame = self.cap.read()

        if not ret:
            logger.info("📹 Fin de la vidéo.")
            return None

        self.frame_count += 1
        return frame

    def read_resized(self, size=INPUT_SIZE):
        """
        Lit et redimensionne la frame pour YOLO.

        Args:
            size (int) : taille cible (640 par défaut)

        Returns:
            frame_original : frame originale
            frame_resized  : frame redimensionnée pour YOLO
        """
        frame = self.read()
        if frame is None:
            return None, None

        frame_resized = cv2.resize(frame, (size, size))
        return frame, frame_resized

    def reset(self):
        """Remet la vidéo au début (pour les fichiers)."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            logger.info("🔄 Vidéo remise au début")

    def get_info(self):
        """Retourne les informations de la source."""
        return {
            'source'     : self.source,
            'width'      : self.width,
            'height'     : self.height,
            'fps'        : self.fps,
            'frame_count': self.frame_count,
        }

    def release(self):
        """Libère les ressources."""
        if self.cap:
            self.cap.release()
            logger.info("✅ Capture vidéo libérée")

    def __del__(self):
        self.release()

    def __iter__(self):
        """Permet d'utiliser VideoCapture dans une boucle for."""
        return self

    def __next__(self):
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame


# ── Test standalone ──────────────────────────────────────────
if __name__ == '__main__':
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else VIDEO_SOURCE

    print(f"\n🎥 Test M1 — Acquisition vidéo")
    print(f"   Source : {source}\n")

    cap     = VideoCapture(source)
    t_start = time.time()
    n       = 0

    for frame in cap:
        n += 1
        cv2.imshow('M1 — Acquisition', frame)

        # Afficher FPS toutes les 30 frames
        if n % 30 == 0:
            elapsed = time.time() - t_start
            fps     = n / elapsed
            print(f"   Frame {n:4d} | FPS réel : {fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n✅ Test terminé")
    print(f"   Frames lues : {n}")
    print(f"   Durée       : {elapsed:.1f}s")
    print(f"   FPS moyen   : {n/elapsed:.1f}")