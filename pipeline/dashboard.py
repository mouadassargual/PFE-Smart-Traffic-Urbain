# =============================================================
# dashboard.py — Module M6 : Dashboard gouvernemental Flask
# Smart Traffic Agadir — PFE Master 2 IA Embarquée
# =============================================================

import cv2
import time
import json
import threading
import numpy as np
from flask import (Flask, render_template_string,
                   Response, request, jsonify)
from config import DASHBOARD_HOST, DASHBOARD_PORT, CLASSES

app = Flask(__name__)

# ── État global partagé ───────────────────────────────────────
state = {
    'zones': {
        'North' : {'frame': None, 'counts': {}, 'active': False},
        'South' : {'frame': None, 'counts': {}, 'active': False},
        'East'  : {'frame': None, 'counts': {}, 'active': False},
        'West'  : {'frame': None, 'counts': {}, 'active': False},
    },
    'decision': {
        'phase'   : 'All-Red',
        'duration': 0,
        'reason'  : 'Initialisation...',
        'lights'  : {
            'North': 'red', 'South': 'red',
            'East' : 'red', 'West' : 'red',
            'Pedestrian': 'red'
        },
        'time_remaining': 0,
    },
    'stats': {
        'total_detections': 0,
        'total_frames'    : 0,
        'fps'             : 0.0,
        'uptime'          : 0,
    },
    'start_time': time.time(),
}
state_lock = threading.Lock()

# ── Template HTML Dashboard ───────────────────────────────────
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,
      initial-scale=1.0">
<meta http-equiv="refresh" content="2">
<title>Smart Traffic Agadir — Dashboard</title>
<style>
  :root {
    --primary   : #003366;
    --secondary : #005599;
    --accent    : #CC0000;
    --light     : #F5F7FA;
    --text      : #1A1A2E;
    --border    : #D0D7E3;
    --green     : #00873E;
    --red       : #CC0000;
    --orange    : #FF8C00;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background  : var(--light);
    color       : var(--text);
    min-height  : 100vh;
  }

  /* ── Header ─────────────────────────── */
  header {
    background  : var(--primary);
    color       : white;
    padding     : 12px 24px;
    display     : flex;
    align-items : center;
    justify-content: space-between;
    border-bottom: 3px solid var(--accent);
  }

  header .logo-section {
    display     : flex;
    align-items : center;
    gap         : 16px;
  }

  header .logo-icon {
    width       : 48px;
    height      : 48px;
    background  : white;
    border-radius: 8px;
    display     : flex;
    align-items : center;
    justify-content: center;
    font-size   : 28px;
  }

  header h1 {
    font-size   : 1.2rem;
    font-weight : 700;
    line-height : 1.3;
  }

  header p {
    font-size   : 0.75rem;
    opacity     : 0.8;
  }

  .header-right {
    text-align  : right;
    font-size   : 0.8rem;
    opacity     : 0.9;
  }

  /* ── Layout ─────────────────────────── */
  main {
    padding     : 20px;
    display     : grid;
    gap         : 20px;
  }

  /* ── Stats bar ──────────────────────── */
  .stats-bar {
    display     : grid;
    grid-template-columns: repeat(4, 1fr);
    gap         : 12px;
  }

  .stat-card {
    background  : white;
    border      : 1px solid var(--border);
    border-radius: 8px;
    padding     : 16px;
    text-align  : center;
    border-top  : 3px solid var(--secondary);
  }

  .stat-card .value {
    font-size   : 2rem;
    font-weight : 700;
    color       : var(--primary);
  }

  .stat-card .label {
    font-size   : 0.75rem;
    color       : #666;
    margin-top  : 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Decision panel ─────────────────── */
  .decision-panel {
    background  : white;
    border      : 1px solid var(--border);
    border-radius: 8px;
    padding     : 20px;
    display     : grid;
    grid-template-columns: 1fr 1fr;
    gap         : 20px;
    align-items : center;
  }

  .decision-info h3 {
    font-size   : 0.8rem;
    text-transform: uppercase;
    color       : #666;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
  }

  .phase-badge {
    display     : inline-block;
    padding     : 6px 16px;
    border-radius: 20px;
    font-weight : 700;
    font-size   : 0.95rem;
    margin-bottom: 8px;
  }

  .phase-NS       { background: #E8F5E9; color: #1B5E20; }
  .phase-EW       { background: #E3F2FD; color: #0D47A1; }
  .phase-PED      { background: #FFF3E0; color: #E65100; }
  .phase-EMG      { background: #FFEBEE; color: #B71C1C; }
  .phase-RED      { background: #F5F5F5; color: #424242; }

  .reason {
    font-size   : 0.85rem;
    color       : #555;
    margin-top  : 4px;
  }

  .duration-bar {
    margin-top  : 12px;
  }

  .duration-bar label {
    font-size   : 0.8rem;
    color       : #666;
  }

  .progress {
    height      : 8px;
    background  : #E0E0E0;
    border-radius: 4px;
    margin-top  : 4px;
    overflow    : hidden;
  }

  .progress-fill {
    height      : 100%;
    border-radius: 4px;
    background  : var(--secondary);
    transition  : width 1s linear;
  }

  /* ── Traffic lights ─────────────────── */
  .lights-grid {
    display     : grid;
    grid-template-columns: repeat(3, 1fr);
    gap         : 10px;
  }

  .light-zone {
    text-align  : center;
  }

  .light-zone .zone-name {
    font-size   : 0.7rem;
    font-weight : 600;
    text-transform: uppercase;
    color       : #666;
    margin-bottom: 6px;
  }

  .traffic-light {
    width       : 40px;
    height      : 100px;
    background  : #1a1a1a;
    border-radius: 6px;
    margin      : 0 auto;
    padding     : 8px 6px;
    display     : flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .bulb {
    width       : 28px;
    height      : 28px;
    border-radius: 50%;
    border      : 2px solid #333;
  }

  .bulb.red    { background: #CC0000; box-shadow: 0 0 10px #CC0000; }
  .bulb.orange { background: #FF8C00; box-shadow: 0 0 10px #FF8C00; }
  .bulb.green  { background: #00873E; box-shadow: 0 0 10px #00873E; }
  .bulb.off    { background: #333; box-shadow: none; }

  /* ── Video zones ────────────────────── */
  .zones-grid {
    display     : grid;
    grid-template-columns: repeat(2, 1fr);
    gap         : 16px;
  }

  .zone-card {
    background  : white;
    border      : 1px solid var(--border);
    border-radius: 8px;
    overflow    : hidden;
  }

  .zone-header {
    background  : var(--primary);
    color       : white;
    padding     : 10px 16px;
    display     : flex;
    justify-content: space-between;
    align-items : center;
  }

  .zone-header h3 {
    font-size   : 0.9rem;
    font-weight : 600;
  }

  .zone-light-indicator {
    width       : 12px;
    height      : 12px;
    border-radius: 50%;
    border      : 2px solid rgba(255,255,255,0.5);
  }

  .indicator-green  { background: #00FF7F; }
  .indicator-red    { background: #FF4444; }
  .indicator-orange { background: #FFA500; }

  .zone-video {
    width       : 100%;
    height      : 220px;
    background  : #0a0a0a;
    display     : flex;
    align-items : center;
    justify-content: center;
    color       : #444;
    font-size   : 0.85rem;
    position    : relative;
  }

  .zone-video img {
    width       : 100%;
    height      : 100%;
    object-fit  : cover;
  }

  .no-video {
    text-align  : center;
  }

  .no-video .upload-btn {
    display     : block;
    margin      : 10px auto 0;
    padding     : 6px 16px;
    background  : var(--secondary);
    color       : white;
    border      : none;
    border-radius: 4px;
    cursor      : pointer;
    font-size   : 0.8rem;
  }

  .zone-stats {
    padding     : 12px 16px;
    display     : grid;
    grid-template-columns: repeat(3, 1fr);
    gap         : 8px;
  }

  .zone-stat {
    text-align  : center;
  }

  .zone-stat .n {
    font-size   : 1.3rem;
    font-weight : 700;
    color       : var(--primary);
  }

  .zone-stat .l {
    font-size   : 0.65rem;
    color       : #888;
    text-transform: uppercase;
  }

  /* ── Upload modal ───────────────────── */
  .upload-overlay {
    display     : none;
    position    : fixed;
    inset       : 0;
    background  : rgba(0,0,0,0.5);
    z-index     : 1000;
    align-items : center;
    justify-content: center;
  }

  .upload-overlay.active {
    display     : flex;
  }

  .upload-modal {
    background  : white;
    border-radius: 12px;
    padding     : 32px;
    width       : 400px;
    text-align  : center;
  }

  .upload-modal h3 {
    font-size   : 1.1rem;
    color       : var(--primary);
    margin-bottom: 16px;
  }

  .upload-modal input[type=file] {
    display     : block;
    width       : 100%;
    padding     : 12px;
    border      : 2px dashed var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    cursor      : pointer;
  }

  .btn-primary {
    padding     : 10px 24px;
    background  : var(--primary);
    color       : white;
    border      : none;
    border-radius: 6px;
    cursor      : pointer;
    font-size   : 0.9rem;
    margin-right: 8px;
  }

  .btn-secondary {
    padding     : 10px 24px;
    background  : #E0E0E0;
    color       : #333;
    border      : none;
    border-radius: 6px;
    cursor      : pointer;
    font-size   : 0.9rem;
  }

  /* ── Footer ─────────────────────────── */
  footer {
    text-align  : center;
    padding     : 12px;
    font-size   : 0.75rem;
    color       : #888;
    border-top  : 1px solid var(--border);
    margin-top  : 20px;
  }
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo-section">
    <div class="logo-icon">🚦</div>
    <div>
      <h1>Système de Gestion Intelligente du Trafic</h1>
      <p>Smart City Agadir — Edge AI sur Raspberry Pi 5</p>
    </div>
  </div>
  <div class="header-right">
    <div id="clock">--:--:--</div>
    <div>FSA Aït Melloul — IBN ZOHR</div>
  </div>
</header>

<main>

  <!-- Stats bar -->
  <div class="stats-bar">
    <div class="stat-card">
      <div class="value" id="total-det">{{ stats.total_detections }}</div>
      <div class="label">Détections totales</div>
    </div>
    <div class="stat-card">
      <div class="value" id="total-frames">{{ stats.total_frames }}</div>
      <div class="label">Frames traitées</div>
    </div>
    <div class="stat-card">
      <div class="value" id="fps">{{ "%.1f"|format(stats.fps) }}</div>
      <div class="label">FPS pipeline</div>
    </div>
    <div class="stat-card">
      <div class="value" id="uptime">{{ stats.uptime }}s</div>
      <div class="label">Temps actif</div>
    </div>
  </div>

  <!-- Decision panel -->
  <div class="decision-panel">
    <div class="decision-info">
      <h3>🚦 Décision courante</h3>
      <div class="phase-badge {{ phase_class }}">
        {{ decision.phase }}
      </div>
      <div class="reason">{{ decision.reason }}</div>
      <div class="duration-bar">
        <label>Durée phase : {{ decision.duration }}s
          (restant : {{ "%.0f"|format(decision.time_remaining) }}s)
        </label>
        <div class="progress">
          <div class="progress-fill" id="progress-bar"
               style="width:{{ progress_pct }}%"></div>
        </div>
      </div>
    </div>

    <div>
      <h3 style="font-size:0.8rem;color:#666;
                 text-transform:uppercase;
                 letter-spacing:0.1em;
                 margin-bottom:12px;">
        État des feux
      </h3>
      <div class="lights-grid">
        {% for zone, color in decision.lights.items() %}
        <div class="light-zone">
          <div class="zone-name">{{ zone }}</div>
          <div class="traffic-light">
            <div class="bulb {{ 'red' if color == 'red' else 'off' }}"></div>
            <div class="bulb {{ 'orange' if color == 'orange' else 'off' }}"></div>
            <div class="bulb {{ 'green' if color == 'green' else 'off' }}"></div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <!-- Video zones -->
  <div class="zones-grid">
    {% for zone_name, zone_data in zones.items() %}
    <div class="zone-card">
      <div class="zone-header">
        <h3>📷 Zone {{ zone_name }}</h3>
        <div class="zone-light-indicator
          {{ 'indicator-green' if decision.lights.get(zone_name) == 'green'
             else 'indicator-red' }}">
        </div>
      </div>

      <div class="zone-video">
        {% if zone_data.active %}
        <img src="/stream/{{ zone_name }}"
             alt="Flux {{ zone_name }}"
             onerror="this.style.display='none'">
        {% else %}
        <div class="no-video">
          <div>📹</div>
          <div>Aucune vidéo chargée</div>
          <button class="upload-btn"
            onclick="openUpload('{{ zone_name }}')">
            Charger une vidéo
          </button>
        </div>
        {% endif %}
      </div>

      <div class="zone-stats">
        <div class="zone-stat">
          <div class="n">
            {{ zone_data.counts.get('car', 0) +
               zone_data.counts.get('truck', 0) +
               zone_data.counts.get('bus', 0) }}
          </div>
          <div class="l">Véhicules</div>
        </div>
        <div class="zone-stat">
          <div class="n">{{ zone_data.counts.get('person', 0) }}</div>
          <div class="l">Piétons</div>
        </div>
        <div class="zone-stat">
          <div class="n">
            {{ zone_data.counts.get('emergency_vehicle', 0) }}
          </div>
          <div class="l">Urgences</div>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>

</main>

<!-- Upload modal -->
<div class="upload-overlay" id="upload-overlay">
  <div class="upload-modal">
    <h3>📤 Charger une vidéo</h3>
    <p style="color:#666;font-size:0.85rem;margin-bottom:16px;">
      Zone sélectionnée : <strong id="modal-zone"></strong>
    </p>
    <input type="file" id="file-input"
           accept="video/*">
    <div>
      <button class="btn-primary" onclick="uploadVideo()">
        Charger
      </button>
      <button class="btn-secondary"
              onclick="closeUpload()">
        Annuler
      </button>
    </div>
  </div>
</div>

<footer>
  PFE Master 2 Intelligence Artificielle Embarquée —
  Mouad Assargual — FSA Aït Melloul, Université Ibn Zohr
  — 2025/2026
</footer>

<script>
  let selectedZone = '';

  // Horloge
  function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent =
      now.toLocaleTimeString('fr-FR');
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Upload modal
  function openUpload(zone) {
    selectedZone = zone;
    document.getElementById('modal-zone').textContent = zone;
    document.getElementById('upload-overlay')
      .classList.add('active');
  }

  function closeUpload() {
    document.getElementById('upload-overlay')
      .classList.remove('active');
  }

  function uploadVideo() {
    const file = document.getElementById('file-input').files[0];
    if (!file) { alert('Veuillez sélectionner un fichier'); return; }

    const formData = new FormData();
    formData.append('video', file);
    formData.append('zone', selectedZone);

    fetch('/upload', { method: 'POST', body: formData })
      .then(r => r.json())
      .then(data => {
        if (data.success) {
          closeUpload();
          location.reload();
        } else {
          alert('Erreur : ' + data.error);
        }
      });
  }

  // Mise à jour stats sans rechargement
  setInterval(() => {
    fetch('/api/stats')
      .then(r => r.json())
      .then(data => {
        document.getElementById('total-det').textContent =
          data.total_detections;
        document.getElementById('total-frames').textContent =
          data.total_frames;
        document.getElementById('fps').textContent =
          data.fps.toFixed(1);
        document.getElementById('uptime').textContent =
          data.uptime + 's';
      }).catch(() => {});
  }, 2000);
</script>
</body>
</html>
"""


class Dashboard:
    """Module M6 — Dashboard Flask gouvernemental."""

    def __init__(self, host=DASHBOARD_HOST,
                 port=DASHBOARD_PORT):
        self.host = host
        self.port = port
        self._setup_routes()

    def _setup_routes(self):

        @app.route('/')
        def index():
            with state_lock:
                d = state['decision']
                phase = d.get('phase', 'All-Red')
                remaining = d.get('time_remaining', 0)
                duration  = d.get('duration', 1)
                pct = (remaining / duration * 100
                       if duration > 0 else 0)

                phase_classes = {
                    'North-South': 'phase-NS',
                    'East-West'  : 'phase-EW',
                    'Pedestrian' : 'phase-PED',
                    'Emergency'  : 'phase-EMG',
                    'All-Red'    : 'phase-RED',
                }

                return render_template_string(
                    HTML_TEMPLATE,
                    zones         = state['zones'],
                    decision      = d,
                    stats         = state['stats'],
                    phase_class   = phase_classes.get(
                                      phase, 'phase-RED'),
                    progress_pct  = round(pct, 1),
                )

        @app.route('/stream/<zone_name>')
        def stream(zone_name):
            def generate():
                while True:
                    with state_lock:
                        frame = state['zones'].get(
                            zone_name, {}
                        ).get('frame', None)

                    if frame is not None:
                        _, buf = cv2.imencode(
                            '.jpg', frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 75]
                        )
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + buf.tobytes() + b'\r\n')
                    time.sleep(0.1)

            return Response(
                generate(),
                mimetype='multipart/x-mixed-replace; '
                         'boundary=frame'
            )

        @app.route('/upload', methods=['POST'])
        def upload():
            zone  = request.form.get('zone')
            file  = request.files.get('video')
            if not zone or not file:
                return jsonify({'success': False,
                                'error': 'Données manquantes'})

            import os
            path = f'/tmp/zone_{zone}.mp4'
            file.save(path)

            with state_lock:
                if zone in state['zones']:
                    state['zones'][zone]['active'] = True

            return jsonify({'success': True, 'path': path})

        @app.route('/api/stats')
        def api_stats():
            with state_lock:
                return jsonify(state['stats'])

        @app.route('/api/decision')
        def api_decision():
            with state_lock:
                return jsonify(state['decision'])

    def update(self, zone_name, frame,
               counts, decision):
        """
        Met à jour l'état du dashboard.

        Args:
            zone_name : nom de la zone
            frame     : image annotée
            counts    : {class: count}
            decision  : décision MDP
        """
        with state_lock:
            if zone_name in state['zones']:
                state['zones'][zone_name]['frame']  = frame
                state['zones'][zone_name]['counts'] = counts
                state['zones'][zone_name]['active'] = True

            state['decision'] = decision
            state['stats']['total_frames'] += 1
            state['stats']['uptime'] = int(
                time.time() - state['start_time']
            )

    def update_stats(self, total_det, fps):
        with state_lock:
            state['stats']['total_detections'] = total_det
            state['stats']['fps']              = fps

    def run(self):
        """Lance le serveur Flask."""
        import logging as lg
        log = lg.getLogger('werkzeug')
        log.setLevel(lg.ERROR)
        print(f"\n🌐 Dashboard disponible sur :")
        print(f"   http://localhost:{self.port}")
        app.run(host=self.host, port=self.port,
                threaded=True, debug=False)

    def run_background(self):
        """Lance le serveur en arrière-plan."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t