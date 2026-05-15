# =============================================================
# dashboard_main.py — Point d'entrée unique
# Upload vidéos → Pipeline automatique → Dashboard temps réel
# =============================================================

import cv2, time, threading, os, logging
from flask import (Flask, render_template_string,
                   Response, request, jsonify)
from config import MODEL_PATH, DASHBOARD_HOST, DASHBOARD_PORT

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger('dashboard')

app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/smart_traffic_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── État global ───────────────────────────────────────────────
state = {
    'zones': {
        'North': {'frame': None, 'counts': {},
                  'video_path': None, 'active': False,
                  'running': False},
        'South': {'frame': None, 'counts': {},
                  'video_path': None, 'active': False,
                  'running': False},
        'East':  {'frame': None, 'counts': {},
                  'video_path': None, 'active': False,
                  'running': False},
        'West':  {'frame': None, 'counts': {},
                  'video_path': None, 'active': False,
                  'running': False},
    },
    'decision': {
        'phase'         : 'All-Red',
        'duration'      : 15,
        'reason'        : 'En attente de vidéos...',
        'lights'        : {
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
lock = threading.Lock()

# ── Modules IA (chargés une seule fois) ──────────────────────
detector        = None
decision_module = None

def load_ai_modules():
    global detector, decision_module
    from detection import Detector
    from decision  import TrafficDecision
    logger.info("⏳ Chargement modules IA...")
    detector        = Detector(MODEL_PATH)
    decision_module = TrafficDecision()
    logger.info("✅ Modules IA prêts")

# ── Worker par zone ───────────────────────────────────────────
def zone_pipeline(zone_name, video_path):
    """Pipeline complet pour une zone."""
    from acquisition import VideoCapture
    from anonymizer  import Anonymizer
    from tracker     import IoUTracker

    anon    = Anonymizer()
    tracker = IoUTracker()

    logger.info(f"▶️  Pipeline {zone_name} démarré")

    with lock:
        state['zones'][zone_name]['running'] = True

    while True:
        # Lire la vidéo en boucle
        cap = VideoCapture(video_path)

        for frame in cap:
            # Vérifier si nouvelle vidéo uploadée
            with lock:
                current_path = state['zones'][zone_name]['video_path']
            if current_path != video_path:
                cap.release()
                video_path = current_path
                tracker.reset()
                break

            # M2 — Détection
            detections = detector.detect(frame)

            # M4 — Anonymisation
            frame_anon, anon_count = anon.anonymize(
                frame, detections
            )

            # M3 — Tracking + Comptage
            tracks      = tracker.update(detections)
            zone_counts = tracker.count_by_zone(
                tracks, frame.shape
            )

            # Annoter frame
            frame_out = detector.draw(frame_anon, detections)
            _draw_overlay(
                frame_out, zone_name,
                zone_counts.get(zone_name, {}),
                state['decision'].get('lights', {})
            )

            # Mettre à jour état
            with lock:
                state['zones'][zone_name]['frame']  = \
                    frame_out.copy()
                state['zones'][zone_name]['counts'] = \
                    zone_counts.get(zone_name, {})
                state['stats']['total_detections'] += \
                    len(detections)
                state['stats']['total_frames']     += 1
                state['stats']['uptime'] = int(
                    time.time() - state['start_time']
                )

        cap.release()
        # Reboucle automatiquement

# ── Worker de décision ────────────────────────────────────────
def decision_worker():
    """Prend une décision toutes les 3 secondes."""
    while True:
        time.sleep(3)

        # Collecter comptages de toutes les zones
        with lock:
            active_zones = {
                z: d for z, d in state['zones'].items()
                if d['active']
            }

        if not active_zones:
            continue

        # Construire zone_counts complet
        all_classes = ['car','bus','truck','motorcycle',
                       'person','emergency_vehicle']
        zone_counts = {
            z: {c: 0 for c in all_classes}
            for z in ['North','South','East',
                      'West','Pedestrian']
        }

        with lock:
            for zone_name in state['zones']:
                counts = state['zones'][zone_name].get(
                    'counts', {}
                )
                for cls, n in counts.items():
                    if cls in zone_counts.get(zone_name, {}):
                        zone_counts[zone_name][cls] = n

        # M5 — Décision MDP
        dec = decision_module.decide(zone_counts)
        dec['time_remaining'] = decision_module.time_remaining()

        with lock:
            state['decision'] = dec

        # FPS global
        with lock:
            s   = state['stats']
            upt = max(s['uptime'], 1)
            s['fps'] = round(s['total_frames'] / upt, 1)

        logger.info(
            f"🚦 {dec['phase']} ({dec['duration']}s)"
            f" — {dec['reason']}"
        )

# ── Helpers ───────────────────────────────────────────────────
def _draw_overlay(frame, zone_name, counts, lights):
    h, w = frame.shape[:2]

    # Fond semi-transparent
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (220,65), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    veh = sum(counts.get(c,0) for c in
              ['car','bus','truck','motorcycle'])
    ped = counts.get('person', 0)
    emg = counts.get('emergency_vehicle', 0)

    cv2.putText(frame, f"Zone: {zone_name}",
                (8,18), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255,255,255), 1)
    cv2.putText(frame,
                f"Veh:{veh}  Ped:{ped}  Urg:{emg}",
                (8,38), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,255), 1)

    # Indicateur feu
    colors = {'green':(0,200,0),'red':(0,0,200),
              'orange':(0,165,255)}
    lc  = lights.get(zone_name, 'red')
    bgr = colors.get(lc, (0,0,200))
    cv2.circle(frame, (w-20, 20), 14, bgr, -1)
    cv2.circle(frame, (w-20, 20), 14, (255,255,255), 2)

# ── HTML Template ─────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Smart Traffic Agadir</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:'Segoe UI',Arial,sans-serif;
  background:#F0F4F8; color:#1A1A2E;
}

/* Header */
header {
  background:linear-gradient(135deg,#003366,#005599);
  color:white; padding:14px 28px;
  display:flex; align-items:center;
  justify-content:space-between;
  border-bottom:4px solid #CC0000;
  box-shadow:0 2px 8px rgba(0,0,0,0.3);
}
.header-left { display:flex; align-items:center; gap:14px; }
.logo { font-size:2.2rem; }
.title-block h1 { font-size:1.15rem; font-weight:700; }
.title-block p  { font-size:0.72rem; opacity:0.85; }
.header-right   { text-align:right; font-size:0.8rem; }
#clock { font-size:1.1rem; font-weight:700;
         font-family:monospace; }

/* Layout */
main { padding:18px; display:grid; gap:16px; }

/* Stats */
.stats-row {
  display:grid;
  grid-template-columns:repeat(4,1fr);
  gap:12px;
}
.stat {
  background:white; border-radius:8px;
  padding:14px; text-align:center;
  border-top:3px solid #005599;
  box-shadow:0 1px 4px rgba(0,0,0,0.08);
}
.stat .val {
  font-size:1.8rem; font-weight:700;
  color:#003366;
}
.stat .lbl {
  font-size:0.68rem; color:#777;
  text-transform:uppercase; letter-spacing:.06em;
  margin-top:3px;
}

/* Decision + Lights */
.decision-row {
  display:grid;
  grid-template-columns:1fr auto;
  gap:16px;
  background:white; border-radius:8px;
  padding:18px;
  box-shadow:0 1px 4px rgba(0,0,0,0.08);
  align-items:center;
}
.phase-info h4 {
  font-size:0.7rem; text-transform:uppercase;
  color:#888; letter-spacing:.1em; margin-bottom:6px;
}
.phase-badge {
  display:inline-block; padding:5px 18px;
  border-radius:20px; font-weight:700;
  font-size:1rem; margin-bottom:6px;
}
.NS  { background:#E8F5E9; color:#1B5E20; }
.EW  { background:#E3F2FD; color:#0D47A1; }
.PED { background:#FFF3E0; color:#E65100; }
.EMG { background:#FFEBEE; color:#B71C1C;
       animation:pulse 1s infinite; }
.RED { background:#F5F5F5; color:#424242; }
@keyframes pulse {
  0%,100%{ opacity:1; } 50%{ opacity:0.6; }
}
.reason { font-size:0.82rem; color:#555;
          margin-bottom:10px; }
.prog-wrap label {
  font-size:0.72rem; color:#888;
}
.prog {
  height:7px; background:#EEE;
  border-radius:4px; overflow:hidden; margin-top:4px;
}
.prog-fill {
  height:100%; border-radius:4px;
  background:linear-gradient(90deg,#005599,#00AAFF);
  transition:width 1s linear;
}

/* Traffic lights panel */
.lights-panel {
  display:grid;
  grid-template-columns:repeat(3,52px);
  gap:8px;
}
.lt { text-align:center; }
.lt .lname {
  font-size:0.55rem; font-weight:700;
  text-transform:uppercase; color:#666;
  margin-bottom:4px;
}
.tlight {
  width:36px; height:88px;
  background:#111; border-radius:5px;
  margin:0 auto; padding:6px 4px;
  display:flex; flex-direction:column;
  justify-content:space-between;
}
.bulb {
  width:28px; height:28px; border-radius:50%;
  border:1px solid #333;
}
.b-red    { background:#CC0000;
            box-shadow:0 0 8px #CC0000; }
.b-orange { background:#FF8C00;
            box-shadow:0 0 8px #FF8C00; }
.b-green  { background:#00873E;
            box-shadow:0 0 8px #00873E; }
.b-off    { background:#2a2a2a; }

/* Zones grid */
.zones {
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:14px;
}
.zone-card {
  background:white; border-radius:8px;
  overflow:hidden;
  box-shadow:0 1px 4px rgba(0,0,0,0.08);
}
.zone-header {
  background:#003366; color:white;
  padding:9px 14px;
  display:flex; justify-content:space-between;
  align-items:center;
}
.zone-header h3 { font-size:0.88rem; }
.zlight {
  width:11px; height:11px; border-radius:50%;
  border:2px solid rgba(255,255,255,0.4);
}
.zg { background:#00FF7F; }
.zr { background:#FF4444; }
.zo { background:#FFA500; }

.zone-video {
  width:100%; height:210px;
  background:#0D0D0D;
  display:flex; align-items:center;
  justify-content:center;
  position:relative; overflow:hidden;
}
.zone-video img {
  width:100%; height:100%; object-fit:cover;
}
.no-video-msg {
  text-align:center; color:#555;
}
.no-video-msg .icon { font-size:2rem; }
.no-video-msg p {
  font-size:0.78rem; margin:6px 0 10px;
}
.upload-btn {
  padding:7px 18px;
  background:#005599; color:white;
  border:none; border-radius:5px;
  cursor:pointer; font-size:0.8rem;
}
.upload-btn:hover { background:#003366; }

.zone-stats {
  padding:10px 14px;
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:6px;
}
.zstat { text-align:center; }
.zstat .n {
  font-size:1.4rem; font-weight:700;
  color:#003366;
}
.zstat .l {
  font-size:0.62rem; color:#999;
  text-transform:uppercase;
}

/* Upload modal */
.modal-bg {
  display:none; position:fixed; inset:0;
  background:rgba(0,0,0,0.55);
  z-index:999; align-items:center;
  justify-content:center;
}
.modal-bg.open { display:flex; }
.modal {
  background:white; border-radius:12px;
  padding:30px; width:380px;
  box-shadow:0 8px 32px rgba(0,0,0,0.3);
}
.modal h3 {
  font-size:1.1rem; color:#003366;
  margin-bottom:6px;
}
.modal p {
  font-size:0.82rem; color:#666;
  margin-bottom:16px;
}
.file-input-wrapper {
  border:2px dashed #CCC; border-radius:8px;
  padding:20px; text-align:center;
  margin-bottom:16px; cursor:pointer;
  transition:border-color .2s;
}
.file-input-wrapper:hover {
  border-color:#005599;
}
.file-input-wrapper input { display:none; }
.file-input-wrapper .icon { font-size:2rem; }
.file-input-wrapper p {
  font-size:0.8rem; color:#888; margin:6px 0 0;
}
#file-name {
  font-size:0.78rem; color:#005599;
  margin-bottom:12px; min-height:18px;
}
.modal-btns { display:flex; gap:8px; }
.btn-ok {
  flex:1; padding:10px;
  background:#003366; color:white;
  border:none; border-radius:6px;
  cursor:pointer; font-size:0.88rem;
}
.btn-ok:hover { background:#005599; }
.btn-cancel {
  padding:10px 20px;
  background:#EEE; color:#333;
  border:none; border-radius:6px;
  cursor:pointer; font-size:0.88rem;
}

.upload-progress {
  display:none; margin-top:10px;
}
.upload-progress .bar {
  height:6px; background:#EEE;
  border-radius:3px; overflow:hidden;
}
.upload-progress .fill {
  height:100%; background:#005599;
  width:0%; transition:width .3s;
}

/* Footer */
footer {
  text-align:center; padding:12px;
  font-size:0.72rem; color:#999;
  border-top:1px solid #E0E0E0;
}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo">🚦</div>
    <div class="title-block">
      <h1>Système de Gestion Intelligente du Trafic Urbain</h1>
      <p>Smart City Agadir — Edge AI Raspberry Pi 5 —
         Privacy-by-Design — Loi 09-08</p>
    </div>
  </div>
  <div class="header-right">
    <div id="clock">--:--:--</div>
    <div>FSA Aït Melloul — Université Ibn Zohr</div>
    <div>PFE 2025/2026 — Mouad ASSARGUAL</div>
  </div>
</header>

<main>

  <!-- Stats -->
  <div class="stats-row">
    <div class="stat">
      <div class="val" id="s-det">0</div>
      <div class="lbl">Détections totales</div>
    </div>
    <div class="stat">
      <div class="val" id="s-frames">0</div>
      <div class="lbl">Frames traitées</div>
    </div>
    <div class="stat">
      <div class="val" id="s-fps">0.0</div>
      <div class="lbl">FPS pipeline</div>
    </div>
    <div class="stat">
      <div class="val" id="s-uptime">0s</div>
      <div class="lbl">Temps actif</div>
    </div>
  </div>

  <!-- Décision + Feux -->
  <div class="decision-row">
    <div class="phase-info">
      <h4>🧠 Décision MDP courante</h4>
      <div class="phase-badge" id="phase-badge">
        All-Red
      </div>
      <div class="reason" id="phase-reason">
        En attente de vidéos...
      </div>
      <div class="prog-wrap">
        <label id="prog-label">Durée : 0s</label>
        <div class="prog">
          <div class="prog-fill" id="prog-fill"
               style="width:0%"></div>
        </div>
      </div>
    </div>

    <div>
      <div style="font-size:0.68rem;color:#888;
                  text-transform:uppercase;
                  letter-spacing:.1em;
                  margin-bottom:10px;">
        État des feux
      </div>
      <div class="lights-panel" id="lights-panel">
        <!-- Généré par JS -->
      </div>
    </div>
  </div>

  <!-- Zones vidéo -->
  <div class="zones" id="zones-grid">
    {% for zone in ['North','South','East','West'] %}
    <div class="zone-card" id="card-{{ zone }}">
      <div class="zone-header">
        <h3>📷 Zone {{ zone }}</h3>
        <div class="zlight zr"
             id="zlight-{{ zone }}"></div>
      </div>
      <div class="zone-video" id="zvid-{{ zone }}">
        <div class="no-video-msg">
          <div class="icon">📹</div>
          <p>Aucune vidéo chargée</p>
          <button class="upload-btn"
            onclick="openModal('{{ zone }}')">
            ↑ Charger vidéo
          </button>
        </div>
      </div>
      <div class="zone-stats">
        <div class="zstat">
          <div class="n" id="z-veh-{{ zone }}">0</div>
          <div class="l">Véhicules</div>
        </div>
        <div class="zstat">
          <div class="n" id="z-ped-{{ zone }}">0</div>
          <div class="l">Piétons</div>
        </div>
        <div class="zstat">
          <div class="n" id="z-emg-{{ zone }}">0</div>
          <div class="l">Urgences</div>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>

</main>

<!-- Upload Modal -->
<div class="modal-bg" id="modal-bg">
  <div class="modal">
    <h3>📤 Charger une vidéo</h3>
    <p>Zone sélectionnée :
      <strong id="modal-zone-name"></strong>
    </p>
    <div class="file-input-wrapper"
         onclick="document.getElementById('file-in').click()">
      <input type="file" id="file-in"
             accept="video/*"
             onchange="onFileSelect(this)">
      <div class="icon">🎬</div>
      <p>Cliquer pour sélectionner une vidéo<br>
         (.mp4, .avi, .mov)</p>
    </div>
    <div id="file-name"></div>
    <div class="upload-progress" id="upload-prog">
      <div class="bar">
        <div class="fill" id="prog-bar"></div>
      </div>
    </div>
    <div class="modal-btns">
      <button class="btn-ok" onclick="doUpload()">
        ▶ Lancer
      </button>
      <button class="btn-cancel" onclick="closeModal()">
        Annuler
      </button>
    </div>
  </div>
</div>

<footer>
  Gestion Intelligente des Feux de Circulation — Smart City
  via Edge AI — PFE Master 2 IA Embarquée —
  FSA Aït Melloul, Université Ibn Zohr — 2025/2026
</footer>

<script>
const ZONES = ['North','South','East','West'];
let selectedZone = '';
let selectedFile = null;

// ── Horloge ───────────────────────────────────────────────────
function tick() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('fr-FR');
}
setInterval(tick, 1000); tick();

// ── Génerer feux ─────────────────────────────────────────────
function renderLights(lights) {
  const panel = document.getElementById('lights-panel');
  const zones = ['North','South','East',
                 'West','Pedestrian'];
  panel.innerHTML = zones.map(z => {
    const c = lights[z] || 'red';
    return `
    <div class="lt">
      <div class="lname">${z.substring(0,3)}</div>
      <div class="tlight">
        <div class="bulb ${c==='red'?'b-red':'b-off'}"></div>
        <div class="bulb ${c==='orange'?'b-orange':'b-off'}">
        </div>
        <div class="bulb ${c==='green'?'b-green':'b-off'}">
        </div>
      </div>
    </div>`;
  }).join('');
}

// ── Mise à jour décision ─────────────────────────────────────
function updateDecision(d) {
  const badge = document.getElementById('phase-badge');
  const classes = {
    'North-South':'NS','East-West':'EW',
    'Pedestrian':'PED','Emergency':'EMG',
    'All-Red':'RED'
  };
  badge.className = 'phase-badge ' +
    (classes[d.phase] || 'RED');
  badge.textContent = d.phase;

  document.getElementById('phase-reason').textContent =
    d.reason;

  const rem = d.time_remaining || 0;
  const dur = d.duration || 1;
  const pct = Math.round(rem / dur * 100);
  document.getElementById('prog-fill').style.width =
    pct + '%';
  document.getElementById('prog-label').textContent =
    `Durée: ${dur}s — Restant: ${Math.round(rem)}s`;

  renderLights(d.lights || {});

  // Indicateurs de zone
  ZONES.forEach(z => {
    const el = document.getElementById('zlight-' + z);
    if (!el) return;
    const c = (d.lights || {})[z] || 'red';
    el.className = 'zlight ' +
      (c==='green'?'zg':c==='orange'?'zo':'zr');
  });
}

// ── Mise à jour zones ────────────────────────────────────────
function updateZones(zones) {
  ZONES.forEach(z => {
    const zd = zones[z] || {};

    // Stats
    const counts = zd.counts || {};
    const veh = (counts.car||0)+(counts.bus||0)+
                (counts.truck||0)+(counts.motorcycle||0);
    const ped = counts.person || 0;
    const emg = counts.emergency_vehicle || 0;

    const ev = document.getElementById('z-veh-'+z);
    const ep = document.getElementById('z-ped-'+z);
    const ee = document.getElementById('z-emg-'+z);
    if (ev) ev.textContent = veh;
    if (ep) ep.textContent = ped;
    if (ee) ee.textContent = emg;

    // Stream vidéo
    if (zd.active) {
      const vid = document.getElementById('zvid-'+z);
      if (vid && !vid.querySelector('img')) {
        vid.innerHTML =
          `<img src="/stream/${z}"
                alt="Zone ${z}"
                onerror="this.src='/static/nostream.png'">`;
      }
    }
  });
}

// ── Polling API ───────────────────────────────────────────────
function poll() {
  fetch('/api/state')
    .then(r => r.json())
    .then(data => {
      // Stats
      document.getElementById('s-det').textContent =
        data.stats.total_detections;
      document.getElementById('s-frames').textContent =
        data.stats.total_frames;
      document.getElementById('s-fps').textContent =
        data.stats.fps.toFixed(1);
      document.getElementById('s-uptime').textContent =
        data.stats.uptime + 's';

      updateDecision(data.decision);
      updateZones(data.zones);
    })
    .catch(() => {});
}
setInterval(poll, 1500);
poll();

// ── Upload modal ──────────────────────────────────────────────
function openModal(zone) {
  selectedZone = zone;
  selectedFile = null;
  document.getElementById('modal-zone-name').textContent =
    zone;
  document.getElementById('file-name').textContent = '';
  document.getElementById('file-in').value = '';
  document.getElementById('modal-bg').classList.add('open');
}

function closeModal() {
  document.getElementById('modal-bg')
    .classList.remove('open');
}

function onFileSelect(input) {
  const file = input.files[0];
  if (file) {
    selectedFile = file;
    document.getElementById('file-name').textContent =
      '✓ ' + file.name +
      ' (' + (file.size/1e6).toFixed(1) + ' MB)';
  }
}

function doUpload() {
  if (!selectedFile) {
    alert('Veuillez sélectionner un fichier vidéo.');
    return;
  }

  const prog = document.getElementById('upload-prog');
  const bar  = document.getElementById('prog-bar');
  prog.style.display = 'block';

  const formData = new FormData();
  formData.append('video', selectedFile);
  formData.append('zone',  selectedZone);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload');

  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      bar.style.width =
        Math.round(e.loaded/e.total*100) + '%';
    }
  };

  xhr.onload = () => {
    const data = JSON.parse(xhr.responseText);
    if (data.success) {
      closeModal();
      prog.style.display = 'none';
      bar.style.width = '0%';
    } else {
      alert('Erreur : ' + data.error);
    }
  };

  xhr.send(formData);
}
</script>
</body>
</html>
"""

# ── Routes Flask ──────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/stream/<zone_name>')
def stream(zone_name):
    def gen():
        while True:
            with lock:
                frame = state['zones'].get(
                    zone_name, {}
                ).get('frame')
            if frame is not None:
                _, buf = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75]
                )
                yield (b'--frame\r\nContent-Type:'
                       b'image/jpeg\r\n\r\n'
                       + buf.tobytes() + b'\r\n')
            time.sleep(0.08)
    return Response(gen(),
        mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    zone = request.form.get('zone')
    file = request.files.get('video')
    if not zone or not file:
        return jsonify({'success':False,
                        'error':'Données manquantes'})

    path = os.path.join(UPLOAD_FOLDER,
                        f'zone_{zone}.mp4')
    file.save(path)
    logger.info(f"✅ Vidéo zone {zone} sauvegardée")

    with lock:
        old_path = state['zones'][zone]['video_path']
        state['zones'][zone]['video_path'] = path
        state['zones'][zone]['active']     = True

    # Lancer pipeline si pas encore démarré
    if not state['zones'][zone]['running']:
        t = threading.Thread(
            target=zone_pipeline,
            args=(zone, path),
            daemon=True
        )
        t.start()
    else:
        logger.info(f"🔄 Zone {zone} : nouvelle vidéo")

    return jsonify({'success':True})

@app.route('/api/state')
def api_state():
    with lock:
        return jsonify({
            'zones'   : {
                z: {
                    'active' : d['active'],
                    'counts' : d['counts'],
                }
                for z, d in state['zones'].items()
            },
            'decision': state['decision'],
            'stats'   : state['stats'],
        })


# ── Lancement ─────────────────────────────────────────────────
if __name__ == '__main__':
    # Charger modules IA
    load_ai_modules()

    # Thread de décision
    t_dec = threading.Thread(
        target=decision_worker, daemon=True
    )
    t_dec.start()

    print("\n" + "="*55)
    print("🚦 Smart Traffic Agadir — Dashboard")
    print("="*55)
    print(f"🌐 Ouvre : http://localhost:{DASHBOARD_PORT}")
    print("📤 Upload tes vidéos depuis le dashboard")
    print("⚡ Le pipeline démarre automatiquement")
    print("="*55 + "\n")

    import logging as lg
    lg.getLogger('werkzeug').setLevel(lg.ERROR)

    app.run(host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            threaded=True, debug=False)