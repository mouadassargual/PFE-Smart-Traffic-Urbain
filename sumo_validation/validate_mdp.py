"""
validate_mdp.py — Version finale
PFE Smart Traffic Agadir — Mouad ASSARGUAL
"""
import os, json, time
import traci

SUMO_BIN = "sumo"
NET_FILE = "network/agadir.net.xml"
TL_ID    = "CENTER"

PHASES = {
    "NS":   "GGGGrrrrGGGGrrrr",
    "NS_Y": "yyyyrrrryyyyrrrr",
    "EW":   "rrrrGGGGrrrrGGGG",
    "EW_Y": "rrrryyyyrrrryyyyrr",
    "EMG":  "rrrrGGGGrrrrGGGG",
}

def mdp_decision(counts):
    if counts["emergency"]:
        return "EMG", 45
    score_NS = counts["N"] + counts["S"]
    score_EW = counts["E"] + counts["W"]
    phase = "NS" if score_NS >= score_EW else "EW"
    q = score_NS if phase == "NS" else score_EW
    dur = 45 if q >= 10 else (30 if q >= 5 else 15)
    return phase, dur

def check_emergency_all():
    try:
        for vid in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(vid) == "emergency":
                return True
    except: pass
    return False

def get_counts():
    counts = {"N":0,"S":0,"E":0,"W":0,"emergency":False}
    for d, edge in [("N","N_IN"),("S","S_IN"),("E","E_IN"),("W","W_IN")]:
        try:
            for vid in traci.edge.getLastStepVehicleIDs(edge):
                vt = traci.vehicle.getTypeID(vid)
                if vt == "emergency": counts["emergency"] = True
                else: counts[d] += 1
        except: pass
    return counts

def run_sim(rou_file, use_mdp, duration=300):
    cmd = [SUMO_BIN, "-n", NET_FILE, "-r", rou_file,
           "--no-warnings", "--step-length", "1.0",
           "--waiting-time-memory", "1000",
           "--time-to-teleport", "-1"]
    traci.start(cmd)

    waits, arrived = [], 0
    phase, dur, phase_t, yellow = "NS", 30, 0, False
    emergency_handled = False
    decisions = []

    if use_mdp:
        traci.trafficlight.setRedYellowGreenState(TL_ID, PHASES["NS"])

    for step in range(duration):
        traci.simulationStep()
        arrived += traci.simulation.getArrivedNumber()

        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getWaitingTime(vid)
            if wt > 0:
                waits.append(wt)

        if use_mdp:
            # Priorité absolue urgence — vérifié à chaque step
            if not emergency_handled and check_emergency_all():
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, PHASES["EMG"])
                traci.trafficlight.setPhaseDuration(TL_ID, 45)
                decisions.append(
                    f"t={step:3d}s → EMG/45s [🚨 PRIORITÉ ABSOLUE]")
                emergency_handled = True
                phase, dur, phase_t, yellow = "EMG", 45, step, False
                continue

            elapsed = step - phase_t
            if not yellow and elapsed >= dur:
                y_key = phase+"_Y" if (phase+"_Y") in PHASES else "NS_Y"
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, PHASES[y_key])
                yellow, phase_t = True, step

            elif yellow and (step - phase_t) >= 3:
                c = get_counts()
                phase, dur = mdp_decision(c)
                traci.trafficlight.setRedYellowGreenState(
                    TL_ID, PHASES[phase])
                decisions.append(
                    f"t={step:3d}s → {phase}/{dur}s "
                    f"[N={c['N']} S={c['S']} E={c['E']} W={c['W']}]")
                yellow, phase_t = False, step

    traci.close()
    awt = sum(waits)/len(waits) if waits else 0.0
    return round(awt,2), round(arrived/(duration/60),2), decisions


scenarios = [
    ("Dense Nord",  "scenarios/scenario_dense.rou.xml"),
    ("Équilibré",   "scenarios/scenario_equilibre.rou.xml"),
    ("Urgence Est", "scenarios/scenario_urgence.rou.xml"),
    ("Vide",        "scenarios/scenario_vide.rou.xml"),
]

print("\n" + "="*68)
print("  VALIDATION MDP — Smart Traffic Agadir")
print("  Mouad ASSARGUAL — PFE M2 IA Embarquée, Ibn Zohr")
print("="*68)

results = []
for name, rou in scenarios:
    if not os.path.exists(rou):
        print(f"  ⚠️  Manquant : {rou}"); continue

    print(f"\n  ▶ {name}")
    awt_f, tp_f, _ = run_sim(rou, use_mdp=False)
    print(f"    Fixe  : AWT={awt_f}s | TP={tp_f} véh/min")
    time.sleep(0.5)

    awt_m, tp_m, decs = run_sim(rou, use_mdp=True)
    print(f"    MDP   : AWT={awt_m}s | TP={tp_m} véh/min")
    for d in decs[:5]:
        print(f"            {d}")

    red_awt  = ((awt_f - awt_m)/awt_f*100) if awt_f > 0 else 0
    gain_tp  = ((tp_m - tp_f)/tp_f*100)    if tp_f  > 0 else 0

    # H5 validée : MDP meilleur OU les deux sont à 0 (trafic nul = correct)
    h5_ok  = (awt_m < awt_f) or (awt_f == 0 and awt_m == 0)
    h5_lbl = "✅ H5 validée" if h5_ok else "⚠️  à vérifier"
    print(f"    Résultat : ΔAWT={red_awt:+.1f}% → {h5_lbl}")

    results.append({
        "scenario": name,
        "AWT_fixed": awt_f, "AWT_mdp": awt_m,
        "reduction_AWT_pct": round(red_awt,1),
        "TP_fixed": tp_f,   "TP_mdp": tp_m,
        "gain_TP_pct": round(gain_tp,1),
        "H5_validated": h5_ok,
        "decisions_count": len(decs)
    })
    time.sleep(0.5)

# ── Tableau final ──
print("\n" + "="*68)
print(f"  {'Scénario':<16} │ {'AWT Fixe':>9} {'AWT MDP':>8} │ "
      f"{'Réd. AWT':>9} │ {'Statut':>12}")
print("-"*68)
for r in results:
    icon = "✅" if r["H5_validated"] else "⚠️"
    note = "Trafic nul" if (r["AWT_fixed"]==0 and r["AWT_mdp"]==0) else ""
    print(f"  {r['scenario']:<16} │ {r['AWT_fixed']:>8.1f}s "
          f"{r['AWT_mdp']:>7.1f}s │ "
          f"{r['reduction_AWT_pct']:>+8.1f}% │ {icon} {note}")

validated = sum(1 for r in results if r["H5_validated"])
print("="*68)
print(f"\n  Scénarios validés : {validated}/{len(results)}")
print(f"  → H5 {'✅ CONFIRMÉE' if validated >= 3 else '⚠️ partielle'}")

os.makedirs("results", exist_ok=True)
with open("results/validation.json","w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("  ✓ Résultats → results/validation.json\n")
