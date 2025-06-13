# app/model_manager.py

from typing import Dict, Tuple, Optional, Any
import numpy as np
import streamlit as st
from enum import Enum

class ModelType(Enum):
    TEACHABLE_MACHINE = "teachable_machine"
    API = "api"

class ModelManager:
    """
    Gestionnaire unifié pour basculer entre les différents modèles
    Fournit une interface commune indépendamment du modèle utilisé
    """
    
    def __init__(self):
        self.current_model_type = ModelType.TEACHABLE_MACHINE
        self.tm_detector = None
        self.api_adapter = None
        self.is_initialized = False
    
    def initialize_teachable_machine(self, model_dir: str) -> bool:
        """Initialise le modèle Teachable Machine"""
        try:
            from teachable_machine_adapter import TeachableMachinePoseDetector
            self.tm_detector = TeachableMachinePoseDetector(model_dir)
            return True
        except Exception as e:
            st.error(f"❌ Erreur initialisation TM: {e}")
            return False
    
    def initialize_api(self, base_url: str) -> bool:
        """Initialise l'adaptateur API"""
        try:
            from api_adapter import APIAdapter
            self.api_adapter = APIAdapter(base_url)
            
            # Vérifier la santé de l'API
            is_healthy, message = self.api_adapter.check_api_health()
            if not is_healthy:
                st.warning(f"⚠️ API Health Check: {message}")
                
            return True
        except Exception as e:
            st.error(f"❌ Erreur initialisation API: {e}")
            return False
    
    def set_model_type(self, model_type: ModelType) -> bool:
        """Change le type de modèle actif"""
        if model_type == ModelType.TEACHABLE_MACHINE and self.tm_detector is None:
            st.error("❌ Modèle Teachable Machine non initialisé")
            return False
        
        if model_type == ModelType.API and self.api_adapter is None:
            st.error("❌ Adaptateur API non initialisé")
            return False
        
        # Pour l'API, vérifier l'authentification
        if model_type == ModelType.API and not self.api_adapter.is_authenticated():
            st.error("❌ Authentification API requise")
            return False
        
        self.current_model_type = model_type
        return True
    
    def get_current_model(self) -> Any:
        """Retourne l'instance du modèle actuel"""
        if self.current_model_type == ModelType.TEACHABLE_MACHINE:
            return self.tm_detector
        elif self.current_model_type == ModelType.API:
            return self.api_adapter
        return None
    
    def predict_pose(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Prédit la pose avec le modèle actuel
        Interface unifiée pour tous les modèles
        """
        current_model = self.get_current_model()
        
        if current_model is None:
            return "Neutral", 0.0
        
        try:
            return current_model.predict_pose(frame)
        except Exception as e:
            st.error(f"❌ Erreur prédiction {self.current_model_type.value}: {e}")
            return "Neutral", 0.0
    
    def get_all_predictions(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Obtient toutes les prédictions avec le modèle actuel
        """
        current_model = self.get_current_model()
        
        if current_model is None:
            return {"Neutral": 0.8}
        
        try:
            return current_model.get_all_predictions(frame)
        except Exception as e:
            st.error(f"❌ Erreur prédictions {self.current_model_type.value}: {e}")
            return {"Neutral": 0.8}
    
    def draw_keypoints(self, frame: np.ndarray, keypoints=None) -> np.ndarray:
        """
        Dessine les keypoints avec le modèle actuel
        """
        current_model = self.get_current_model()
        
        if current_model is None:
            return frame
        
        try:
            return current_model.draw_keypoints(frame, keypoints)
        except Exception as e:
            st.error(f"❌ Erreur dessin keypoints {self.current_model_type.value}: {e}")
            return frame
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur le modèle actuel
        """
        info = {
            "type": self.current_model_type.value,
            "status": "active" if self.get_current_model() is not None else "inactive",
            "authenticated": False
        }
        
        if self.current_model_type == ModelType.API and self.api_adapter:
            info["authenticated"] = self.api_adapter.is_authenticated()
            info["user_id"] = getattr(self.api_adapter, 'current_user_id', None)
        
        return info
    
    def get_available_gestures(self) -> list:
        """
        Retourne la liste des gestes disponibles selon le modèle
        """
        if self.current_model_type == ModelType.TEACHABLE_MACHINE:
            # Gestes du modèle TM (basé sur votre config)
            return [
                "Twerk", "Dab", "Macaréna", "Floss", "Funk", 
                "CrossArm", "V_Signs", "RussianMoove", "PasDuBourré",
                "CrossFeat", "GlassMoove", "Fuck", "JulSign", 
                "Neutral", "Blood"
            ]
        elif self.current_model_type == ModelType.API:
            # Gestes de l'API (basé sur votre pose_detection_model.py)
            return [
                "hands_up", "dab", "jump", "twerk", "Neutral"
            ]
        
        return ["Neutral"]
    
    def authenticate_api(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authentifie auprès de l'API si c'est le modèle actuel
        """
        if self.current_model_type != ModelType.API or self.api_adapter is None:
            return False, "API non sélectionnée ou non initialisée"
        
        return self.api_adapter.authenticate(username, password)
    
    def register_api_user(self, username: str, email: str, password: str) -> Tuple[bool, str]:
        """
        Enregistre un utilisateur via l'API
        """
        if self.api_adapter is None:
            return False, "API non initialisée"
        
        return self.api_adapter.register(username, email, password)
    
    def logout_api(self):
        """
        Déconnecte de l'API
        """
        if self.api_adapter:
            self.api_adapter.logout()
    
    def get_model_specific_settings(self) -> Dict[str, Any]:
        """
        Retourne les paramètres spécifiques au modèle actuel
        """
        base_settings = {
            "confidence_threshold": 0.7,
            "photo_cooldown": 2.0,
            "debug_mode": False
        }
        
        if self.current_model_type == ModelType.TEACHABLE_MACHINE:
            base_settings.update({
                "priority_gestures": [
                    "Twerk", "Dab", "Macaréna", "Floss", "Funk", "CrossArm"
                ],
                "gesture_duration_threshold": 1.0
            })
        elif self.current_model_type == ModelType.API:
            base_settings.update({
                "priority_gestures": [
                    "hands_up", "dab", "jump", "twerk"
                ],
                "gesture_duration_threshold": 0.5,  # API plus réactive
                "confidence_threshold": 0.5  # API moins strict
            })
        
        return base_settings
    
    def is_model_ready(self) -> bool:
        """
        Vérifie si le modèle actuel est prêt à être utilisé
        """
        if self.current_model_type == ModelType.TEACHABLE_MACHINE:
            return self.tm_detector is not None
        elif self.current_model_type == ModelType.API:
            return (self.api_adapter is not None and 
                   self.api_adapter.is_authenticated())
        
        return False