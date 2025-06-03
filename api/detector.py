# Détection (mock ou vrai modèle)

import random

def detect_movement(frame):
    # Détection simulée : 1 frame sur 50 déclenche
    return random.randint(0, 50) == 0
