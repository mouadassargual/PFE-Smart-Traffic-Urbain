"""
mqtt_receiver.py
================
Bridge MQTT → SUMO — Windows PC
Reçoit les données du Pi 5 et pilote SUMO via TraCI

Flux :
  Pi 5 publie → traffic/counts   {N:14, S:2, E:7, W:6}
  Pi 5 publie → traffic/decision {phase:NORD-SUD, dur:45}
  Windows reçoit → injecte véhicules dans SUMO
  Windows reçoit → applique feux via TraCI
  Windows mesure → AWT, throughput
  Windows publie → traffic/metrics {awt:8.2, tp:6.4}

Usage (Windows) :
  pip install paho-mqtt traci
  python mqtt_receiver.py

Auteur  : Mouad ASSARGUAL
PFE M2  : Intelligence Artificielle Embarquee
FSA Ait Melloul — Universite Ibn Zohr, Agadir
Annee   : 2025/2026
"""

import json
import time
import threading
import traci
import paho.mqtt.client as mqtt
from datetime import datetime

# ════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════

BROKER_HOST = "localhost"     # IP du broker Mosquitto sur Windows
BROKER_PORT = 1883
CLIENT_ID   = "sumo_bridge_windows"

# Topics MQTT
TOPIC_COUNTS   = "traffic/counts"
TOPIC_DECISION = "traffic/decision"
TOPIC_METRICS  = "traffic/sumo_metrics"
TOPIC_STATUS   = "traffic/sumo_status"

# SUMO
SUMO_NET    = "sumo_validation/network/talborjt.net.xml"
TL_ID       = "1054617769"

# Edges entrantes par direction
EDGES_IN = {
    "N": "180596058#0",
    "S": "-180596058#3",
    "E": "180596121#0",
    "W": "-180596121#2",
}

# Routes dans SUMO
ROUTES = {
    "N": "NS",
    "S": "SN",
    "E": "EW",
    "W": "WE",
}

# Phases SUMO
PHASE_NS        = 0
PHASE_NS_YELLOW = 1
PHASE_EW        = 2
PHASE_EW_YELLOW = 3
YELLOW_DUR      = 3


# ════════════════════════════════════════════════════════
# ÉTAT GLOBAL
# ════════════════════════════════════════════════════════

state = {
    "counts":           {"N": 0, "S": 0, "E": 0, "W": 0, "emergency": False},
    "phase":            "NORD-SUD",
    "duration":         30,
    "sumo_running":     False,
    "new_decision":     False,
    "vehicle_counter":  0,
    "awt_buffer":       [],
    "arrived":          0,
    "lock":             threading.Lock(),
}


# ════════════════════════════════════════════════════════
# MQTT CALLBACKS
# ════════════════════════════════════════════════════════

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"  [MQTT] Connecte au broker {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC_COUNTS)
        client.subscribe(TOPIC_DECISION)
        print(f"  [MQTT] Ecoute sur :")
        print(f"         {TOPIC_COUNTS}")
        print(f"         {TOPIC_DECISION}")
    else:
        print(f"  [MQTT] Erreur connexion rc={rc}")


def on_message(client, userdata, msg):
    """Reçoit les données du Pi 5"""
    topic   = msg.topic
    payload = json.loads(msg.payload.decode())

    with state["lock"]:
        if topic == TOPIC_COUNTS:
            state["counts"] = {
                "N":         payload.get("N", 0),
                "S":         payload.get("S", 0),
                "E":         payload.get("E", 0),
                "W":         payload.get("W", 0),
                "emergency": payload.get("emergency", False),
            }
            print(f"  [Pi->] Counts: "
                  f"N={state['counts']['N']} "
                  f"S={state['counts']['S']} "
                  f"E={state['counts']['E']} "
                  f"W={state['counts']['W']}"
                  f"{'  URGENCE!' if state['counts']['emergency'] else ''}")

        elif topic == TOPIC_DECISION:
            state["phase"]        = payload.get("phase", "NORD-SUD")
            state["duration"]     = payload.get("duration", 30)
            state["new_decision"] = True
            print(f"  [Pi->] Decision: {state['phase']} / {state['duration']}s")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"  [MQTT] Deconnexion inattendue rc={rc}")


# ════════════════════════════════════════════════════════
# INJECTION VEHICULES SUMO
# ════════════════════════════════════════════════════════

def inject_vehicles(counts):
    """
    Injecte des véhicules dans SUMO proportionnellement
    aux comptages reçus du Pi 5
    """
    injected = 0
    current_time = traci.simulation.getTime()

    for direction, count in [
        ("N", counts["N"]),
        ("S", counts["S"]),
        ("E", counts["E"]),
        ("W", counts["W"]),
    ]:
        if count <= 0:
            continue

        route_id = ROUTES.get(direction)
        if not route_id:
            continue

        # Injecter 1 véhicule représentatif par direction active
        # (count = densité, pas nombre exact de véhicules à injecter)
        n_inject = max(1, count // 5)  # 1 véhicule pour 5 détectés

        for i in range(n_inject):
            vid = f"veh_{state['vehicle_counter']}"
            state["vehicle_counter"] += 1

            try:
                # Type selon urgence
                vtype = "emergency" if counts.get("emergency") else "car"
                traci.vehicle.add(
                    vehID    = vid,
                    routeID  = route_id,
                    typeID   = vtype,
                    depart   = str(current_time + i * 2),
                )
                injected += 1
            except Exception as e:
                pass  # Véhicule déjà existant ou route indisponible

    return injected


# ════════════════════════════════════════════════════════
# APPLICATION FEUX
# ════════════════════════════════════════════════════════

def apply_traffic_lights(phase, duration):
    """Applique la décision MDP aux feux SUMO via TraCI"""
    try:
        if phase == "EMERGENCY":
            traci.trafficlight.setPhase(TL_ID, PHASE_EW)
            traci.trafficlight.setPhaseDuration(TL_ID, duration)
        elif phase == "NORD-SUD":
            traci.trafficlight.setPhase(TL_ID, PHASE_NS)
            traci.trafficlight.setPhaseDuration(TL_ID, duration)
        elif phase == "EST-OUEST":
            traci.trafficlight.setPhase(TL_ID, PHASE_EW)
            traci.trafficlight.setPhaseDuration(TL_ID, duration)
        return True
    except Exception as e:
        print(f"  [SUMO] Erreur feux : {e}")
        return False


# ════════════════════════════════════════════════════════
# COLLECTE METRIQUES
# ════════════════════════════════════════════════════════

def collect_metrics():
    """Collecte AWT et throughput depuis SUMO"""
    try:
        # Véhicules arrivés ce step
        arrived = traci.simulation.getArrivedNumber()
        state["arrived"] += arrived

        # Temps d'attente
        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(vid)
            if wt > 0:
                state["awt_buffer"].append(wt)

        # Garder buffer raisonnable
        if len(state["awt_buffer"]) > 1000:
            state["awt_buffer"] = state["awt_buffer"][-500:]

    except Exception:
        pass


def get_metrics():
    """Calcule les métriques actuelles"""
    awt = (sum(state["awt_buffer"]) / len(state["awt_buffer"])
           if state["awt_buffer"] else 0.0)
    return {
        "awt_sec":   round(awt, 2),
        "arrived":   state["arrived"],
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════
# BOUCLE SUMO PRINCIPALE
# ════════════════════════════════════════════════════════

def run_sumo(mqtt_client):
    """
    Lance et pilote SUMO en temps réel
    Synchronisé avec les données du Pi 5
    """
    print("\n  Demarrage SUMO...")

    # Creer fichier config temporaire
    import os
    sumocfg = "sumo_validation/agadir_live.sumocfg"

    # Ecrire config SUMO (réseau seulement, véhicules injectés dynamiquement)
    with open(sumocfg, "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network/talborjt.net.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
</configuration>
""")

    # Définir les routes
    routes_xml = "sumo_validation/network/talborjt_routes.rou.xml"
    with open(routes_xml, "w") as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car"       length="4.5" maxSpeed="13.89" accel="2.6" decel="4.5"/>
    <vType id="bus"       length="12.0" maxSpeed="11.11" accel="1.2" decel="3.0"/>
    <vType id="truck"     length="8.0"  maxSpeed="11.11" accel="1.0" decel="3.0"/>
    <vType id="emergency" length="5.5"  maxSpeed="16.67" accel="3.5" decel="5.0" color="1,0,0"/>

    <route id="NS" edges="180596058#0 180596058#1 180596058#2 180596058#3"/>
    <route id="SN" edges="-180596058#3 -180596058#2 -180596058#1 -180596058#0"/>
    <route id="EW" edges="180596121#0 180596121#1 180596121#2"/>
    <route id="WE" edges="-180596121#2 -180596121#1 -180596121#0"/>
</routes>
""")

    # Mettre à jour config avec routes
    with open(sumocfg, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="network/talborjt.net.xml"/>
        <route-files value="network/talborjt_routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1.0"/>
    </time>
</configuration>
""")

    # Démarrer SUMO GUI
    traci.start([
        "sumo-gui",
        "-c", sumocfg,
        "--delay", "100",
        "--start",
        "--quit-on-end",
        "--no-warnings",
    ])

    state["sumo_running"] = True
    print("  [SUMO] Demarre sur carte Talborjt Agadir")
    print("  [SUMO] En attente des donnees Pi 5...\n")

    # Phase initiale
    apply_traffic_lights("NORD-SUD", 30)

    step          = 0
    last_inject   = 0
    INJECT_INTERVAL = 30  # Injecter véhicules toutes les 30 steps

    while state["sumo_running"]:
        try:
            traci.simulationStep()
            step += 1

            # Collecter métriques
            collect_metrics()

            # Appliquer nouvelle décision MDP
            with state["lock"]:
                if state["new_decision"]:
                    apply_traffic_lights(
                        state["phase"],
                        state["duration"]
                    )
                    state["new_decision"] = False
                    print(f"  [SUMO] Feux appliques : "
                          f"{state['phase']} / {state['duration']}s")

                # Injecter véhicules périodiquement
                if step - last_inject >= INJECT_INTERVAL:
                    counts  = state["counts"].copy()
                    n       = inject_vehicles(counts)
                    last_inject = step
                    if n > 0:
                        print(f"  [SUMO] {n} vehicules injectes "
                              f"(N={counts['N']} S={counts['S']} "
                              f"E={counts['E']} W={counts['W']})")

            # Publier métriques toutes les 10 steps
            if step % 10 == 0:
                metrics = get_metrics()
                mqtt_client.publish(
                    TOPIC_METRICS,
                    json.dumps(metrics)
                )
                print(f"  [SUMO] Step={step:4d} | "
                      f"AWT={metrics['awt_sec']:.1f}s | "
                      f"Arrives={metrics['arrived']}")

        except traci.exceptions.FatalTraCIError:
            print("  [SUMO] Simulation terminee.")
            break
        except Exception as e:
            print(f"  [SUMO] Erreur step {step} : {e}")
            break

    state["sumo_running"] = False
    try:
        traci.close()
    except Exception:
        pass


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  SUMO Bridge — Smart Traffic Agadir")
    print("  Mouad ASSARGUAL — PFE M2 IA Embarquee")
    print("  Intersection Talborjt, Agadir")
    print("="*60)

    # Connexion MQTT
    client = mqtt.Client(client_id=CLIENT_ID)
    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"\n  Connexion broker MQTT {BROKER_HOST}:{BROKER_PORT}...")
    try:
        client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    except Exception as e:
        print(f"  Erreur : {e}")
        print("  Verifier que Mosquitto est demarre sur Windows")
        print("  → net start mosquitto")
        return

    # Démarrer MQTT en arrière-plan
    client.loop_start()

    # Publier statut
    client.publish(TOPIC_STATUS, json.dumps({
        "status":    "ready",
        "component": "sumo_bridge",
        "timestamp": datetime.now().isoformat(),
    }))

    # Attendre quelques secondes pour recevoir premières données
    print("\n  Attente donnees Pi 5 (5s)...")
    time.sleep(5)

    # Lancer SUMO dans thread principal (TraCI doit être dans main thread)
    try:
        run_sumo(client)
    except KeyboardInterrupt:
        print("\n  Arret demande.")
    finally:
        state["sumo_running"] = False
        client.publish(TOPIC_STATUS, json.dumps({
            "status":    "offline",
            "component": "sumo_bridge",
        }))
        client.loop_stop()
        client.disconnect()
        print("  Arret propre.")


if __name__ == "__main__":
    main()
