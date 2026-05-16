# Registre des modeles — Smart Traffic Agadir

## yolo26n_int8.onnx — MODELE PRINCIPAL
Entraine par : Mouad ASSARGUAL
Dataset      : v2 (10 228 images marocaines)
mAP@50       : 92.4%
Format       : INT8 (ONNX Runtime)
FPS Pi 5     : 5.0
Usage        : Production Pi 5

## yolo26n_finetuned.onnx — FINE-TUNING
Base         : yolo26n_best.pt
Dataset      : v3 (v2 + 38 frames Bellevue)
Date         : 16 Mai 2026

## bellevue_best.onnx — BASELINE REFERENCE
Source       : DiTAT [Pradhananga et al., SUMO 2025]
GitHub       : github.com/shebywiliamsjr/TrafficDT
Dataset      : VisDrone (10 classes)
Usage        : Baseline validation visuelle
Citation     : doi.org/10.52825/scp.v6i.2619
