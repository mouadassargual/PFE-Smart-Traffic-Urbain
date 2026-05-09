<div align="center">
  <img src="docs/logo fsaaam.png" alt="FSA Aït Melloul Logo" width="150" />
  
  # 🚦 Smart Traffic Agadir
  **Gestion Intelligente des Feux de Circulation via Edge AI**
  
  *PFE Master 2 Intelligence Artificielle Embarquée*  
  *FSA Aït Melloul — Université Ibn Zohr — 2024/2026*
</div>

---

## 📖 Présentation du Projet

Ce projet vise à concevoir et déployer un système de gestion du trafic urbain intelligent (Smart Traffic) adapté aux intersections complexes de la ville d'Agadir, en s'appuyant sur l'Edge AI et la vision par ordinateur. 

L'objectif est d'optimiser les temps d'attente aux feux tricolores de manière dynamique en utilisant un modèle de détection d'objets en temps réel, déployé localement sur un **Raspberry Pi 5**. Le système assure également la confidentialité des citoyens grâce à un module d'anonymisation à la source.

---

## 🛠️ Architecture du Système (Pipeline 5 Modules)

Le système est structuré en 5 modules indépendants et interconnectés :

1. **Acquisition** : Capture et prétraitement des flux vidéo (depuis caméras de surveillance ou drones DJI).
2. **Détection (YOLO)** : Détection des véhicules, piétons et véhicules d'urgence.
3. **Tracking (BoT-SORT)** : Suivi temporel pour éviter les doubles comptages et estimer les trajectoires.
4. **Anonymisation** : Floutage temps réel des visages et plaques d'immatriculation (Conformité RGPD/CNDP).
5. **Décision (MDP)** : Logique de contrôle dynamique des feux tricolores basée sur Processus de Décision Markoviens.

---

## 📊 Modèles & Benchmark

Une étude comparative rigoureuse a été menée sur plusieurs architectures Nano pour identifier le meilleur compromis entre précision (mAP) et latence sur architecture ARM (Raspberry Pi 5). Voici les résultats finaux obtenus sur le Dataset V2 (11 364 images) après 150 epochs d'entraînement sur Google Colab Pro (GPU A100) :

| Modèle | mAP@50 | mAP@50-95 | Remarques |
| :--- | :--- | :--- | :--- |
| **YOLOv8n** | 92.64 % | ~ 71.0 % | Modèle de référence très performant et stable. |
| **YOLOv11n** | **92.70 %** | ~ 72.0 % | Meilleur mAP@50 global, architecture très équilibrée. |
| **YOLO26n** | 92.41 % | **72.54 %** | Excellente précision sur les détections strictes (mAP@50-95). |

*Conclusion : Les trois modèles présentent des performances exceptionnelles et très proches (tous au-dessus de 92% de mAP@50). YOLOv11n et YOLO26n s'imposent comme d'excellents candidats pour le déploiement final sur le Raspberry Pi 5 en raison de leurs améliorations architecturales récentes.*

### 🗂️ Dataset V2 (Smart Traffic Agadir)
Le dataset a été minutieusement constitué pour ce projet, totalisant **11 364 images** réparties sur **6 classes** unifiées :
`car`, `bus`, `truck`, `motorcycle`, `person`, `emergency_vehicle`.

Sources consolidées :
- Données locales (Lqliaa Traffic V1)
- Emergency Vehicles Dataset
- Surveillance Cameras Dataset
- AI-ITS Pedestrian Dataset

---

## 📂 Structure du Répertoire

```text
├── data/                    # Scripts de gestion et fusion du dataset
├── dataset_v2/              # Dataset final (téléchargeable via Kaggle/DVC)
├── docs/                    # Documents et ressources graphiques
├── Figures yolov8n/         # Résultats et métriques des benchmarks
├── models/                  # Poids des modèles entraînés (.pt, .onnx)
├── notebooks/               # Notebooks d'exploration locale
├── pipeline/                # Code source des 5 modules du système embarqué
│   ├── acquisition.py
│   ├── detection.py
│   ├── tracker.py
│   ├── anonymizer.py
│   └── dashboard.py         # Interface de supervision de la municipalité
├── YOLO Smart Traffic Benchmark.ipynb # Script de training Colab Pro
└── README.md
```

---

## 🚀 Reproductibilité (Benchmark)

Pour relancer l'entraînement et l'évaluation des 3 modèles :
1. Importer `YOLO Smart Traffic Benchmark.ipynb` sur **Google Colab Pro**.
2. Renseigner la clé API Kaggle dans la cellule de téléchargement du Dataset.
3. Lancer l'exécution (*Run All*). Les modèles entraînés et les métriques (JSON) seront automatiquement sauvegardés sur Google Drive.

---

*Développé par **Mouad Assargual** dans le cadre du Projet de Fin d'Études.*
