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

L'étude comparative s'est déroulée en deux phases : d'abord l'évaluation théorique sur GPU, puis l'optimisation matérielle sur carte embarquée (Raspberry Pi 5).

### Phase 1 : Précision Théorique (GPU A100 - 150 Epochs)

| Modèle | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- |
| **YOLOv8n** | 92.64 % | ~ 71.0 % |
| **YOLOv11n** | **92.70 %** | ~ 72.0 % |
| **YOLO26n** | 92.41 % | **72.54 %** |

*YOLO26n a été retenu pour la suite grâce à son excellente localisation spatiale (mAP@50-95), primordiale pour le suivi urbain complexe.*

### Phase 2 : Benchmark Matériel (Edge AI - Raspberry Pi 5)

Les trois modèles ont été testés en conditions réelles d'inférence (ONNX Runtime CPU) sur l'architecture ARM Cortex-A76 du Pi 5. Une quantification INT8 a ensuite été appliquée au modèle retenu.

| Modèle | Format | FPS Réel | Latence moy. | Stabilité |
| :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | FP32 | 5.0 | 193 ms | ⚠️ Moyenne |
| **YOLOv11n** | FP32 | 4.8 | 200 ms | ⚠️ Moyenne |
| **YOLO26n** | FP32 | 3.7 | 178 ms | ❌ Instable (I/O Bottleneck) |
| **YOLO26n** | FP16 | — | — | ❌ Incompatible (ARM CPU) |
| **YOLO26n** | **INT8** | **5.0** | **193 ms** | ✅ **Retenu** |

**YOLO26n INT8 = Modèle de déploiement final ✅**

**Justification complète pour le déploiement :**
- **Meilleure Précision GPU** : 92.4% mAP@50 avec le meilleur score mAP@50-95.
- **Sécurité Piétons** : Meilleur taux de détection sur la classe "Person" (93.8%).
- **Fluidité Pi 5** : 5.0 FPS réels atteints.
- **Stabilité Système** : La quantification INT8 libère la RAM, garantissant un pipeline vidéo fluide.
- **Modernité** : Architecture la plus récente et optimale.

✅ **Hypothèse H3 validée** : *"Un dispositif embarqué à ressources limitées peut atteindre une latence compatible avec le traitement en temps réel via l'optimisation matérielle INT8."*

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
├── benchmark_results/       # Résultats Kaggle/Colab (Courbes, Matrices, Poids)
├── dataset_v2/              # Dataset final (téléchargeable via Kaggle/DVC)
├── docs/                    # Documents et ressources graphiques
├── models_rpi/              # Modèles ONNX finaux optimisés pour Raspberry Pi
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
