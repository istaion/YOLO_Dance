"""
Configuration de l'application YOLO Dance
"""
import os

# 📁 Chemins
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # racine du projet
SAVE_DIR = os.path.join(PROJECT_ROOT, "images")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model", "teachable_machine")

# 🎯 Paramètres par défaut
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_PHOTO_COOLDOWN = 2.0
DEFAULT_GESTURE_DURATION_THRESHOLD = 1.0

# 🕺 Gestes disponibles
ALL_GESTURES = [
    "Twerk", "Dab", "Macaréna", "Floss", "Funk", 
    "CrossArm", "V_Signs", "RussianMoove", "PasDuBourré",
    "CrossFeat", "GlassMoove", "Fuck", "JulSign", 
    "Neutral", "Blood"
]

# 🎯 Gestes prioritaires par défaut
DEFAULT_PRIORITY_GESTURES = [
    "Twerk", "Dab", "Macaréna", "Floss", "Funk", "CrossArm"
]

# 📱 Configuration Streamlit
PAGE_CONFIG = {
    "page_title": " YOLO Dance",
    "page_icon": "🕺",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 🎨 Émojis pour les gestes
GESTURE_EMOJIS = {
    "Twerk": "🍑",
    "Dab": "💪",
    "Macaréna": "💃",
    "Floss": "🦷",
    "Funk": "🕺",
    "CrossArm": "❌",
    "V_Signs": "✌️",
    "RussianMoove": "🇷🇺",
    "PasDuBourré": "🩰",
    "CrossFeat": "⚡",
    "GlassMoove": "🥃",
    "Fuck": "🖕",
    "JulSign": "👌",
    "Neutral": "😐",
    "Blood": "🩸"
}

# 📸 Configuration photo
PHOTO_CONFIG = {
    "prefix": "tm_",
    "format": "jpg",
    "timestamp_format": "%Y%m%d_%H%M%S"
}

# 🎥 Configuration caméra
CAMERA_CONFIG = {
    "fps_limit": 30,
    "mirror": True,
    "default_camera_id": 0
}