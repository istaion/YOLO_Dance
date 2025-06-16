# app/config.py
import os
import sys
from pathlib import Path

# Configuration des chemins selon votre structure
BASE_DIR = Path(__file__).parent.parent  # Remonte au dossier racine du projet
APP_DIR = BASE_DIR / "app"
SCRIPTS_DIR = BASE_DIR / "scripts" / "yolo_pipeline"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"

# Ajouter le chemin des scripts YOLO au Python path
sys.path.append(str(SCRIPTS_DIR))

# Configuration des chemins spécifiques
YOLO_MODEL_PATH = MODELS_DIR / "best_model.pth"
TEACHABLE_MACHINE_MODEL_DIR = MODELS_DIR / "teachable_machine"
YOLO_POSE_MODEL = MODELS_DIR / "yolov8n-pose.pt"

# Répertoires d'images
TM_IMAGES_DIR = IMAGES_DIR / "teachable_machine"
YOLO_IMAGES_DIR = IMAGES_DIR / "yolo_dance"

# Créer les répertoires s'ils n'existent pas
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
TM_IMAGES_DIR.mkdir(exist_ok=True)
YOLO_IMAGES_DIR.mkdir(exist_ok=True)
TEACHABLE_MACHINE_MODEL_DIR.mkdir(exist_ok=True)

# Classes YOLO Dance (selon votre modèle entraîné)
YOLO_CLASSES = [
    'hands_up', 'dab', 'twerk', 
    'jul', 'neutral', 
    'crossarm'
]

# Configuration par défaut
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_PHOTO_COOLDOWN = 2.0
DEFAULT_GESTURE_DURATION = 1.0

# Configuration de la caméra
CAMERA_FPS_LIMIT = 30
CAMERA_RESOLUTION = (640, 480)

# Gestes prioritaires par défaut
DEFAULT_YOLO_PRIORITY_GESTURES = ['dab', 'twerk', 'hands_up', 'jul', 'crossarm']
DEFAULT_TM_PRIORITY_GESTURES = ["Twerk", "Dab", "Macaréna", "Floss", "Funk", "CrossArm"]

print(f"📁 Configuration chargée:")
print(f"   - Base: {BASE_DIR}")
print(f"   - Scripts YOLO: {SCRIPTS_DIR}")
print(f"   - Modèles: {MODELS_DIR}")
print(f"   - Images: {IMAGES_DIR}")
print(f"   - Classes YOLO: {len(YOLO_CLASSES)}")

# Vérifications
if not SCRIPTS_DIR.exists():
    print(f"⚠️  Attention: Le répertoire scripts n'existe pas: {SCRIPTS_DIR}")

if not YOLO_MODEL_PATH.exists():
    print(f"⚠️  Attention: Modèle YOLO non trouvé: {YOLO_MODEL_PATH}")

if not YOLO_POSE_MODEL.exists():
    print(f"⚠️  Attention: YOLOv8 pose non trouvé: {YOLO_POSE_MODEL}")
    print("   Téléchargement automatique lors du premier lancement...")