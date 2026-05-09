import time
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

def benchmark_model(model_path, video_path, num_frames=300):
    """
    Benchmark YOLO model inference speed and FPS on Raspberry Pi.
    """
    print(f"\n{'='*50}")
    print(f"🚀 Benchmarking {model_path} sur Raspberry Pi 5")
    print(f"{'='*50}")
    
    # 1. Charger le modèle
    print("⏳ Chargement du modèle...")
    try:
        # On précise task='detect' pour les modèles ONNX
        model = YOLO(model_path, task='detect')
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return

    # 2. Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la vidéo: {video_path}")
        return
    
    # Récupérer les infos de la vidéo
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 Vidéo source : {orig_width}x{orig_height} @ {orig_fps:.1f} FPS")
    print(f"🔄 Test sur les {num_frames} premières frames (Warm-up inclus)...")

    latencies = []
    frames_processed = 0
    start_total_time = time.time()

    # 3. Boucle d'inférence
    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Démarrer le chronomètre
        start_time = time.perf_counter()
        
        # Inférence avec paramètres optimisés pour la vitesse
        # imgsz=640 (taille standard), half=True (si supporté), verbose=False
        results = model(frame, imgsz=640, verbose=False)
        
        # Arrêter le chronomètre
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        frames_processed += 1
        
        # Afficher la progression
        if frames_processed % 50 == 0:
            print(f"   [{frames_processed}/{num_frames}] Latence: {latency_ms:.1f} ms | FPS instantané: {1000/latency_ms:.1f}")

    end_total_time = time.time()
    cap.release()

    if frames_processed == 0:
        print("❌ Aucune frame traitée.")
        return

    # 4. Calcul des statistiques (en ignorant les 10 premières frames de warm-up)
    warmup_frames = min(10, frames_processed - 1)
    valid_latencies = latencies[warmup_frames:]
    
    avg_latency = np.mean(valid_latencies)
    min_latency = np.min(valid_latencies)
    max_latency = np.max(valid_latencies)
    
    avg_fps = 1000 / avg_latency if avg_latency > 0 else 0
    total_time = end_total_time - start_total_time
    throughput = frames_processed / total_time

    # 5. Rapport Final
    print(f"\n📊 RÉSULTATS DU BENCHMARK ({frames_processed} frames)")
    print(f"--------------------------------------------------")
    print(f"⏱️  Latence moyenne   : {avg_latency:.2f} ms")
    print(f"🔻 Latence Min/Max   : {min_latency:.2f} ms / {max_latency:.2f} ms")
    print(f"🚀 FPS Théorique     : {avg_fps:.1f} images/sec (Basé sur latence inférence)")
    print(f"🌍 FPS Réel global   : {throughput:.1f} images/sec (Inclus lecture vidéo)")
    print(f"--------------------------------------------------")
    
    if avg_fps > 15:
        print("✅ PERFORMANCE OPTIMALE : Déploiement Raspberry Pi recommandé.")
    elif avg_fps > 5:
        print("⚠️ PERFORMANCE MOYENNE : Utilisable mais avec des sauts de frames.")
    else:
        print("❌ PERFORMANCE CRITIQUE : Modèle trop lourd. Préférer format ONNX ou NCNN.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO sur Raspberry Pi 5")
    parser.add_argument("--model", type=str, default="models/yolo26n.onnx", help="Chemin vers le modèle (.pt, .onnx, .engine)")
    parser.add_argument("--video", type=str, required=True, help="Chemin vers la vidéo de test")
    parser.add_argument("--frames", type=int, default=300, help="Nombre de frames à traiter")
    
    args = parser.parse_args()
    benchmark_model(args.model, args.video, args.frames)
