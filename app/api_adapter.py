# app/api_adapter.py

import requests
import streamlit as st
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import io
from PIL import Image
import json

class APIAdapter:
    """
    Adaptateur pour communiquer avec l'API de détection de poses
    Interface commune avec le modèle Teachable Machine
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.current_token = None
        self.current_user_id = None
        
    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authentifie l'utilisateur auprès de l'API
        Returns: (success, message)
        """
        try:
            response = self.session.post(
                f"{self.base_url}/log/token",
                json={"username": username, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_token = data.get("access_token")
                
                # Configurer l'en-tête d'autorisation pour les futures requêtes
                self.session.headers.update({
                    "Authorization": f"Bearer {self.current_token}"
                })
                
                return True, "Connexion réussie !"
            else:
                error_detail = response.json().get("detail", "Erreur de connexion")
                return False, f"Erreur d'authentification: {error_detail}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Erreur de connexion à l'API: {str(e)}"
    
    def register(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Enregistre un nouvel utilisateur
        Returns: (success, message)
        """
        try:
            response = self.session.post(
                f"{self.base_url}/log/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "Inscription réussie ! Vous pouvez maintenant vous connecter."
            else:
                error_detail = response.json().get("detail", "Erreur d'inscription")
                return False, f"Erreur: {error_detail}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Erreur de connexion à l'API: {str(e)}"
    
    def logout(self):
        """Déconnecte l'utilisateur"""
        self.current_token = None
        self.current_user_id = None
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]
    
    def is_authenticated(self) -> bool:
        """Vérifie si l'utilisateur est authentifié"""
        return self.current_token is not None
    
    def detect_gesture_api(self, frame: np.ndarray) -> Tuple[bool, Dict]:
        """
        Détecte les gestes via l'API
        Compatible avec l'interface du modèle Teachable Machine
        """
        if not self.is_authenticated():
            return False, {"error": "Non authentifié"}
        
        try:
            # Convertir la frame en image
            _, buffer = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            
            # Appel à l'API
            response = self.session.post(
                f"{self.base_url}/api/detect-gesture",
                files=files,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Adapter le format de retour pour être compatible avec TM
                gesture = result.get("gesture", "Neutral")
                detected = result.get("detected", False)
                user_id = result.get("user_id")
                
                return detected, {
                    "gesture": gesture,
                    "detected": detected,
                    "confidence": 0.8 if detected else 0.1,  # L'API ne retourne pas de confiance
                    "user_id": user_id,
                    "source": "api"
                }
            else:
                error_detail = response.json().get("detail", "Erreur de détection")
                return False, {"error": error_detail}
                
        except requests.exceptions.RequestException as e:
            return False, {"error": f"Erreur de connexion: {str(e)}"}
    
    def predict_pose(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Interface compatible avec TeachableMachinePoseDetector
        """
        detected, result = self.detect_gesture_api(frame)
        
        if "error" in result:
            return "Neutral", 0.0
        
        gesture = result.get("gesture", "Neutral")
        confidence = result.get("confidence", 0.0)
        
        return gesture, confidence
    
    def get_all_predictions(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Interface compatible avec TeachableMachinePoseDetector
        Retourne toutes les prédictions avec leurs confidences
        """
        detected, result = self.detect_gesture_api(frame)
        
        if "error" in result:
            return {"Neutral": 0.8, "Error": 0.2}
        
        gesture = result.get("gesture", "Neutral")
        confidence = result.get("confidence", 0.0)
        
        # Simuler un dictionnaire de prédictions
        predictions = {gesture: confidence}
        
        # Ajouter d'autres gestes avec confiance 0
        all_gestures = [
            "hands_up", "dab", "twerk", "jump", "Neutral"
        ]
        
        for g in all_gestures:
            if g not in predictions:
                predictions[g] = 0.0
        
        return predictions
    
    def draw_keypoints(self, frame: np.ndarray, keypoints=None) -> np.ndarray:
        """
        Interface compatible - l'API ne retourne pas de keypoints à dessiner
        Retourne la frame telle quelle avec un overlay indiquant l'utilisation de l'API
        """
        # Ajouter un indicateur visuel que l'API est utilisée
        frame_copy = frame.copy()
        
        # Badge "API" en haut à droite
        cv2.rectangle(frame_copy, (frame.shape[1]-80, 10), (frame.shape[1]-10, 40), (0, 255, 0), -1)
        cv2.putText(frame_copy, "API", (frame.shape[1]-65, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame_copy
    
    def check_api_health(self) -> Tuple[bool, str]:
        """
        Vérifie si l'API est accessible
        """
        try:
            response = self.session.get(f"{self.base_url}/detect/", timeout=5)
            if response.status_code in [200, 405]:  # 405 = Method Not Allowed (normal pour GET sur POST endpoint)
                return True, "API accessible"
            else:
                return False, f"API retourne le code {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"API non accessible: {str(e)}"