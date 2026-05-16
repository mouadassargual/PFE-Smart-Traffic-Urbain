"""
mqtt_sender.py
==============
Module MQTT — Raspberry Pi 5
Envoie les données du pipeline en temps réel vers :
  - Flask Dashboard (affichage)
  - SUMO Bridge Windows (simulation)

Topics publiés :
  traffic/counts    → comptages zones N/S/E/W
  traffic/decision  → décision MDP (phase, durée)
  traffic/privacy   → statistiques anonymisation
  traffic/metrics   → AWT, FPS, frame
  traffic/emergency → alerte urgence

Auteur  : Mouad ASSARGUAL
PFE M2  : Intelligence Artificielle Embarquée
FSA Aït Melloul — Université Ibn Zohr, Agadir
Année   : 2025/2026
"""

import json
import time
import threading
import logging
from datetime import datetime

# ── Tentative import paho-mqtt ──
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("  [MQTT] paho-mqtt non installe → pip install paho-mqtt")
    print("  [MQTT] Mode simulation (pas d'envoi réel)")


# ── Configuration ──
BROKER_HOST = "localhost"   # Sur Pi 5 : IP du broker Mosquitto
                             # Sur Mac  : "localhost" pour test
BROKER_PORT = 1883
KEEPALIVE   = 60
CLIENT_ID   = "smart_traffic_pi5"

# Topics
TOPIC_COUNTS    = "traffic/counts"
TOPIC_DECISION  = "traffic/decision"
TOPIC_PRIVACY   = "traffic/privacy"
TOPIC_METRICS   = "traffic/metrics"
TOPIC_EMERGENCY = "traffic/emergency"
TOPIC_STATUS    = "traffic/status"

# QoS
QOS_REALTIME = 0   # Fire and forget → temps réel
QOS_RELIABLE = 1   # At least once   → décisions importantes


class MQTTSender:
    """
    Gestionnaire MQTT pour le pipeline Smart Traffic Pi 5

    Usage :
        sender = MQTTSender(broker_host="192.168.1.X")
        sender.connect()

        # Dans la boucle pipeline :
        sender.send_counts({"N":5, "S":2, "E":1, "W":0})
        sender.send_decision("NORD-SUD", 45, "Dense NS")
        sender.send_privacy(3)
        sender.send_metrics(fps=5.0, frame=120, awt=8.2)

        sender.disconnect()
    """

    def __init__(self, broker_host: str = BROKER_HOST,
                 broker_port: int = BROKER_PORT,
                 verbose: bool = True):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.verbose     = verbose
        self.connected   = False
        self.client      = None
        self._lock       = threading.Lock()

        # Stats
        self.messages_sent  = 0
        self.messages_failed = 0
        self.last_send_time  = None

        # Buffer pour mode offline
        self._offline_buffer = []

        logging.basicConfig(level=logging.WARNING)

    # ── Connexion ──

    def connect(self) -> bool:
        """Connecte au broker MQTT"""
        if not MQTT_AVAILABLE:
            print("  [MQTT] Mode offline — données bufferisées localement")
            return False

        try:
            self.client = mqtt.Client(
                client_id=CLIENT_ID,
                clean_session=True,
                protocol=mqtt.MQTTv311
            )

            self.client.on_connect    = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish    = self._on_publish

            # Will message (status offline si déconnexion brutale)
            self.client.will_set(
                TOPIC_STATUS,
                json.dumps({"status": "offline", "client": CLIENT_ID}),
                qos=QOS_RELIABLE,
                retain=True
            )

            self.client.connect(self.broker_host, self.broker_port, KEEPALIVE)
            self.client.loop_start()

            # Attendre connexion
            timeout = 5.0
            t0 = time.time()
            while not self.connected and (time.time() - t0) < timeout:
                time.sleep(0.1)

            if self.connected:
                self._publish(TOPIC_STATUS, {
                    "status":    "online",
                    "client":    CLIENT_ID,
                    "timestamp": self._ts()
                }, retain=True)
                if self.verbose:
                    print(f"  [MQTT] Connecte → {self.broker_host}:{self.broker_port}")
            else:
                print(f"  [MQTT] Timeout connexion → {self.broker_host}:{self.broker_port}")
                print("  [MQTT] Mode offline active")

            return self.connected

        except Exception as e:
            print(f"  [MQTT] Erreur connexion : {e}")
            print("  [MQTT] Mode offline active")
            return False

    def disconnect(self):
        """Déconnecte proprement"""
        if self.client and self.connected:
            self._publish(TOPIC_STATUS, {
                "status":    "offline",
                "client":    CLIENT_ID,
                "timestamp": self._ts()
            }, retain=True)
            self.client.loop_stop()
            self.client.disconnect()
            if self.verbose:
                print(f"  [MQTT] Deconnecte ({self.messages_sent} messages envoyes)")

    # ── Callbacks ──

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
        else:
            print(f"  [MQTT] Erreur connexion rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            print(f"  [MQTT] Deconnexion inattendue rc={rc}")

    def _on_publish(self, client, userdata, mid):
        pass  # Confirmation publication

    # ── Publication ──

    def _publish(self, topic: str, data: dict,
                 qos: int = QOS_REALTIME,
                 retain: bool = False) -> bool:
        """Publication générique avec gestion offline"""
        payload = json.dumps(data, ensure_ascii=False)

        if not MQTT_AVAILABLE or not self.connected:
            # Mode offline : buffer local
            self._offline_buffer.append({
                "topic": topic, "data": data,
                "ts": self._ts()
            })
            if len(self._offline_buffer) > 1000:
                self._offline_buffer.pop(0)
            return False

        try:
            with self._lock:
                result = self.client.publish(topic, payload, qos=qos, retain=retain)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self.messages_sent += 1
                    self.last_send_time = time.time()
                    return True
                else:
                    self.messages_failed += 1
                    return False
        except Exception as e:
            self.messages_failed += 1
            if self.verbose:
                print(f"  [MQTT] Erreur publish {topic} : {e}")
            return False

    # ── API Publique ──

    def send_counts(self, counts: dict, frame_id: int = 0):
        """
        Envoie les comptages par zone

        Args:
            counts   : {"N":int, "S":int, "E":int, "W":int,
                        "ped":int, "emergency":bool}
            frame_id : numéro de frame courant
        """
        data = {
            "N":         counts.get("N", 0),
            "S":         counts.get("S", 0),
            "E":         counts.get("E", 0),
            "W":         counts.get("W", 0),
            "ped":       counts.get("ped", 0),
            "emergency": counts.get("emergency", False),
            "total":     counts.get("N",0) + counts.get("S",0) +
                         counts.get("E",0) + counts.get("W",0),
            "frame":     frame_id,
            "timestamp": self._ts()
        }
        self._publish(TOPIC_COUNTS, data, qos=QOS_REALTIME)

        # Alerte urgence séparée
        if counts.get("emergency", False):
            self.send_emergency(frame_id)

    def send_decision(self, phase: str, duration: int,
                      reason: str = "", frame_id: int = 0):
        """
        Envoie la décision MDP

        Args:
            phase    : "NORD-SUD", "EST-OUEST", ou "EMERGENCY"
            duration : durée en secondes (15, 30, ou 45)
            reason   : justification textuelle
            frame_id : numéro de frame
        """
        data = {
            "phase":     phase,
            "duration":  duration,
            "reason":    reason,
            "frame":     frame_id,
            "timestamp": self._ts()
        }
        # Décisions envoyées avec QoS 1 (fiable)
        self._publish(TOPIC_DECISION, data, qos=QOS_RELIABLE)

        if self.verbose:
            print(f"  [MQTT→] Decision: {phase}/{duration}s | {reason}")

    def send_privacy(self, anon_count: int, frame_id: int = 0):
        """
        Envoie les stats d'anonymisation

        Args:
            anon_count : nombre de piétons anonymisés ce frame
            frame_id   : numéro de frame
        """
        data = {
            "anonymized_count": anon_count,
            "method":           "Gaussian Blur 51x51",
            "compliance":       "Loi 09-08 Maroc",
            "frame":            frame_id,
            "timestamp":        self._ts()
        }
        self._publish(TOPIC_PRIVACY, data, qos=QOS_REALTIME)

    def send_metrics(self, fps: float, frame_id: int,
                     awt: float = 0.0, total_anon: int = 0,
                     detections: int = 0):
        """
        Envoie les métriques de performance

        Args:
            fps        : FPS courant du pipeline
            frame_id   : numéro de frame
            awt        : Average Waiting Time estimé (s)
            total_anon : total piétons anonymisés depuis démarrage
            detections : nombre de détections ce frame
        """
        data = {
            "fps":          round(fps, 1),
            "frame":        frame_id,
            "awt_sec":      round(awt, 2),
            "total_anon":   total_anon,
            "detections":   detections,
            "model":        "YOLO26n INT8",
            "hardware":     "Raspberry Pi 5",
            "timestamp":    self._ts()
        }
        self._publish(TOPIC_METRICS, data, qos=QOS_REALTIME)

    def send_emergency(self, frame_id: int = 0):
        """Envoie alerte urgence avec QoS élevé"""
        data = {
            "alert":     True,
            "type":      "emergency_vehicle",
            "action":    "Phase EMERGENCY / 45s",
            "frame":     frame_id,
            "timestamp": self._ts()
        }
        self._publish(TOPIC_EMERGENCY, data, qos=QOS_RELIABLE)
        if self.verbose:
            print(f"  [MQTT→] 🚨 URGENCE DETECTEE frame={frame_id}")

    def send_video_frame(self, frame_b64: str):
        """
        Envoie un frame vidéo encodé base64 pour le dashboard
        Utiliser avec modération (bande passante)
        """
        data = {
            "frame_b64": frame_b64,
            "timestamp": self._ts()
        }
        self._publish("traffic/video", data, qos=QOS_REALTIME)

    # ── Utilitaires ──

    def get_stats(self) -> dict:
        """Retourne les statistiques de publication"""
        return {
            "connected":       self.connected,
            "broker":          f"{self.broker_host}:{self.broker_port}",
            "messages_sent":   self.messages_sent,
            "messages_failed": self.messages_failed,
            "offline_buffer":  len(self._offline_buffer),
            "last_send":       self.last_send_time
        }

    def flush_offline_buffer(self):
        """Renvoie les messages bufferisés en offline"""
        if not self._offline_buffer:
            return 0
        sent = 0
        while self._offline_buffer and self.connected:
            msg = self._offline_buffer.pop(0)
            if self._publish(msg["topic"], msg["data"]):
                sent += 1
        return sent

    @staticmethod
    def _ts() -> str:
        """Timestamp ISO format"""
        return datetime.now().isoformat()


# ── Singleton global pour usage dans demo_pipeline.py ──
_sender_instance = None

def get_sender(broker_host: str = BROKER_HOST) -> MQTTSender:
    """Retourne l'instance singleton du sender"""
    global _sender_instance
    if _sender_instance is None:
        _sender_instance = MQTTSender(broker_host=broker_host)
        _sender_instance.connect()
    return _sender_instance


# ── Test standalone ──
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Test mqtt_sender.py")
    print("  Smart Traffic Agadir — PFE M2 IA Embarquee")
    print("="*50)

    # Tester avec broker local
    sender = MQTTSender(broker_host="localhost", verbose=True)
    connected = sender.connect()

    print(f"\n  Connexion : {'OK' if connected else 'OFFLINE (mode simulation)'}")
    print(f"  Debut envoi de donnees de test...\n")

    # Simuler 10 cycles du pipeline
    for i in range(10):
        # Simuler comptages variables
        import random
        counts = {
            "N": random.randint(0, 15),
            "S": random.randint(0, 8),
            "E": random.randint(0, 5),
            "W": random.randint(0, 5),
            "ped": random.randint(0, 3),
            "emergency": (i == 7)  # Urgence au cycle 7
        }

        # Décision MDP simulée
        ns = counts["N"] + counts["S"]
        ew = counts["E"] + counts["W"]
        if counts["emergency"]:
            phase, dur, reason = "EMERGENCY", 45, "Vehicule urgence"
        elif ns >= ew:
            phase = "NORD-SUD"
            dur   = 45 if ns >= 10 else (30 if ns >= 5 else 15)
            reason = f"NS={ns} > EW={ew}"
        else:
            phase = "EST-OUEST"
            dur   = 45 if ew >= 10 else (30 if ew >= 5 else 15)
            reason = f"EW={ew} > NS={ns}"

        # Envoyer toutes les données
        sender.send_counts(counts, frame_id=i*30)
        sender.send_decision(phase, dur, reason, frame_id=i*30)
        sender.send_privacy(counts["ped"], frame_id=i*30)
        sender.send_metrics(fps=5.0, frame_id=i*30,
                           awt=round(random.uniform(3, 20), 1),
                           total_anon=counts["ped"]*i,
                           detections=counts["N"]+counts["S"]+counts["E"]+counts["W"])

        print(f"  Cycle {i+1:2d} : N={counts['N']} S={counts['S']} "
              f"E={counts['E']} W={counts['W']} "
              f"→ {phase}/{dur}s"
              f"{'  🚨' if counts['emergency'] else ''}")

        time.sleep(1)

    stats = sender.get_stats()
    print(f"\n  Stats : {stats['messages_sent']} messages envoyes, "
          f"{stats['messages_failed']} echecs")

    sender.disconnect()
    print("\n  Test termine.")