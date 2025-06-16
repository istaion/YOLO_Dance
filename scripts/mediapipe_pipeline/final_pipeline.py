"""
Pipeline temps réel pour détection de poses et capture automatique de photos
Intègre MediaPipe + Feature Engineering + TensorFlow + Détection de sauts
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
from typing import List, Tuple, Optional, Dict
import threading
import queue

# Imports des modules créés
from keypoint_extractor import MediaPipeKeypointExtractor, PersonKeypoints
from feature_engineering import PoseFeatureEngineer, EngineeredFeatures
from tensorflow_model import PoseClassificationModel

class RealTimePosePipeline:
    """
    Pipeline temps réel pour détection de poses et capture automatique
    """
    
    def __init__(self, 
                 model_path: str = "../../models",
                 scaler_path: str = "../../data/features/feature_scaler.joblib",
                 confidence_threshold: float = 0.7,
                 jump_threshold: float = 0.15,
                 smoothing_window: int = 5):
        """
        Initialise le pipeline temps réel
        
        Args:
            model_path: Chemin vers le modèle TensorFlow
            scaler_path: Chemin vers le scaler des features
            confidence_threshold: Seuil de confiance pour déclencher une photo
            jump_threshold: Seuil de détection de saut (changement hauteur visage)
            smoothing_window: Taille de la fenêtre de lissage temporel
        """
        
        self.confidence_threshold = confidence_threshold
        self.jump_threshold = jump_threshold
        self.smoothing_window = smoothing_window
        
        # Initialisation des composants
        print("🚀 Initialisation du pipeline temps réel...")
        
        # 1. Extracteur de keypoints MediaPipe
        self.keypoint_extractor = MediaPipeKeypointExtractor(
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        print("✅ MediaPipe initialisé")
        
        # 2. Feature engineer
        self.feature_engineer = PoseFeatureEngineer()
        print("✅ Feature Engineer initialisé")
        
        # 3. Modèle TensorFlow
        self.model = PoseClassificationModel()
        if not self.model.load_model(model_path):
            raise ValueError(f"Impossible de charger le modèle depuis {model_path}")
        print("✅ Modèle TensorFlow chargé")
        
        # 4. Charger le scaler
        try:
            import joblib
            self.feature_engineer.scaler = joblib.load(scaler_path)
            print("✅ Scaler chargé")
        except Exception as e:
            print(f"⚠️  Scaler non trouvé: {e}. Les features ne seront pas normalisées.")
            self.feature_engineer.scaler = None
        
        # Variables de state
        self.face_positions_history = deque(maxlen=smoothing_window * 2)  # Historique positions visage
        self.predictions_history = deque(maxlen=smoothing_window)          # Historique prédictions
        self.last_photo_time = 0
        self.photo_cooldown = 2.0  # 2 secondes entre photos
        
        # Statistiques
        self.frame_count = 0
        self.fps_counter = deque(maxlen=30)
        self.start_time = time.time()
        
        # Configuration photos
        self.photos_dir = "../../photos"
        os.makedirs(self.photos_dir, exist_ok=True)
        
        # Interface utilisateur
        self.show_keypoints = True
        self.show_features_debug = False
        self.paused = False
        
        print("🎯 Pipeline prêt !")
    
    def update_parameters(self, confidence: float = None, jump: float = None):
        """Met à jour les paramètres en temps réel"""
        if confidence is not None:
            self.confidence_threshold = confidence
        if jump is not None:
            self.jump_threshold = jump
    
    def detect_jump(self, current_face_y: float) -> bool:
        """
        Détecte un saut en analysant l'historique des positions du visage
        
        Args:
            current_face_y: Position Y actuelle du visage (0-1)
            
        Returns:
            True si un saut est détecté
        """
        if len(self.face_positions_history) < self.smoothing_window:
            return False
        
        # Calculer la position moyenne récente (avant le mouvement potentiel)
        recent_positions = list(self.face_positions_history)[-self.smoothing_window:]
        avg_recent_y = np.mean([pos[1] for pos in recent_positions])
        
        # Différence de hauteur (y plus petit = plus haut dans l'image)
        height_change = avg_recent_y - current_face_y
        
        # Saut détecté si mouvement vers le haut significatif
        return height_change > self.jump_threshold
    
    def smooth_predictions(self, current_probs: np.ndarray) -> np.ndarray:
        """
        Lisse les prédictions temporellement pour éviter le flickering
        
        Args:
            current_probs: Probabilités actuelles
            
        Returns:
            Probabilités lissées
        """
        self.predictions_history.append(current_probs)
        
        if len(self.predictions_history) >= 3:
            # Moyenne pondérée avec plus de poids sur les prédictions récentes
            weights = np.linspace(0.5, 1.0, len(self.predictions_history))
            weights = weights / np.sum(weights)
            
            smoothed = np.zeros_like(current_probs)
            for i, pred in enumerate(self.predictions_history):
                smoothed += weights[i] * pred
            
            return smoothed
        
        return current_probs
    
    def should_take_photo(self, pose_confidence: float, predicted_class: str, 
                         face_position: Tuple[float, float]) -> Tuple[bool, str]:
        """
        Détermine s'il faut prendre une photo
        
        Args:
            pose_confidence: Confiance de la prédiction de pose
            predicted_class: Classe prédite
            face_position: Position du visage
            
        Returns:
            (should_take, reason)
        """
        current_time = time.time()
        
        # Vérifier le cooldown
        if current_time - self.last_photo_time < self.photo_cooldown:
            return False, "Cooldown"
        
        # Vérifier la confiance
        if pose_confidence < self.confidence_threshold:
            return False, f"Confiance trop faible: {pose_confidence:.2f}"
        
        # Vérifier que ce n'est pas 'neutral'
        if predicted_class == 'neutral':
            return False, "Pose neutre"
        
        # Détecter un saut (optionnel - peut être activé/désactivé)
        jump_detected = self.detect_jump(face_position[1])
        
        # Prendre photo si pose confiante (avec ou sans saut selon configuration)
        return True, f"Pose détectée: {predicted_class} (confiance: {pose_confidence:.2f})"
    
    def capture_photo(self, frame: np.ndarray, predicted_class: str, 
                     confidence: float, reason: str):
        """
        Capture et sauvegarde une photo
        
        Args:
            frame: Image à sauvegarder
            predicted_class: Classe détectée
            confidence: Confiance de la prédiction
            reason: Raison de la capture
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{predicted_class}_{confidence:.2f}_{timestamp}.jpg"
        filepath = os.path.join(self.photos_dir, filename)
        
        # Sauvegarder l'image
        cv2.imwrite(filepath, frame)
        
        # Mettre à jour le temps de dernière photo
        self.last_photo_time = time.time()
        
        print(f"📸 Photo capturée: {filename} - {reason}")
    
    def draw_interface(self, frame: np.ndarray, persons: List[PersonKeypoints],
                      pose_probs: np.ndarray, face_position: np.ndarray,
                      predicted_class: str, confidence: float, 
                      fps: float, jump_status: str) -> np.ndarray:
        """
        Dessine l'interface utilisateur sur l'image
        
        Args:
            frame: Image de base
            persons: Personnes détectées
            pose_probs: Probabilités des poses
            face_position: Position du visage prédite
            predicted_class: Classe prédite
            confidence: Confiance de la prédiction
            fps: FPS actuels
            jump_status: Statut de détection de saut
            
        Returns:
            Image avec interface
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Dessiner les keypoints si activé
        if self.show_keypoints and persons:
            for person in persons:
                # Points de pose (vert)
                if person.pose_landmarks is not None:
                    for x, y, z in person.pose_landmarks:
                        if z > 0.1:  # Seulement les points visibles
                            px, py = int(x * width), int(y * height)
                            cv2.circle(overlay, (px, py), 3, (0, 255, 0), -1)
                
                # Points des mains (bleu/rouge)
                if person.left_hand_landmarks is not None:
                    for x, y, z in person.left_hand_landmarks:
                        px, py = int(x * width), int(y * height)
                        cv2.circle(overlay, (px, py), 2, (255, 0, 0), -1)
                
                if person.right_hand_landmarks is not None:
                    for x, y, z in person.right_hand_landmarks:
                        px, py = int(x * width), int(y * height)
                        cv2.circle(overlay, (px, py), 2, (0, 0, 255), -1)
                
                # Centre du visage (jaune)
                if person.face_center:
                    center_px = (int(person.face_center[0] * width), 
                               int(person.face_center[1] * height))
                    cv2.circle(overlay, center_px, 8, (0, 255, 255), -1)
        
        # Panel d'information (semi-transparent)
        panel_height = 200
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (0, 0, 0)  # Noir
        
        # Informations principales
        y_offset = 30
        cv2.putText(panel, f"Pose: {predicted_class.upper()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset += 30
        color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 255, 255)
        cv2.putText(panel, f"Confiance: {confidence:.3f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 25
        cv2.putText(panel, f"Seuil: {self.confidence_threshold:.2f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Position du visage
        y_offset += 25
        if face_position is not None:
            cv2.putText(panel, f"Visage: ({face_position[0]:.2f}, {face_position[1]:.2f})", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Statut de saut
        y_offset += 25
        cv2.putText(panel, f"Saut: {jump_status}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Statistiques
        y_offset += 25
        cv2.putText(panel, f"FPS: {fps:.1f} | Frames: {self.frame_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Top 3 des prédictions
        if len(pose_probs) > 0:
            top_indices = np.argsort(pose_probs)[-3:][::-1]
            x_start = width - 200
            
            cv2.putText(panel, "Top 3:", (x_start, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i, idx in enumerate(top_indices):
                class_name = self.model.classes[idx]
                prob = pose_probs[idx]
                y_pos = 60 + i * 25
                
                color = (0, 255, 0) if i == 0 and prob >= self.confidence_threshold else (255, 255, 255)
                cv2.putText(panel, f"{class_name}: {prob:.3f}", (x_start, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions
        instructions = [
            "ESPACE: Pause",
            "K: Keypoints On/Off", 
            "C/V: Confiance +/-",
            "J/H: Saut +/-",
            "Q: Quitter"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(panel, instruction, (width - 200, 120 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Fusionner le panel avec l'image
        alpha = 0.8
        overlay[:panel_height] = cv2.addWeighted(overlay[:panel_height], 1-alpha, panel, alpha, 0)
        
        return overlay
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Traite une frame complète
        
        Args:
            frame: Image d'entrée
            
        Returns:
            (image_with_overlay, info_dict)
        """
        info = {}
        
        # 1. Extraction des keypoints
        persons = self.keypoint_extractor.extract_keypoints_from_image(frame)
        
        if not persons:
            # Aucune personne détectée
            info.update({
                'predicted_class': 'none',
                'confidence': 0.0,
                'face_position': None,
                'jump_detected': False,
                'jump_status': 'No person'
            })
            
            overlay = self.draw_interface(frame, [], np.array([]), None, 
                                        'none', 0.0, self.calculate_fps(), 'No person')
            return overlay, info
        
        # Prendre la première personne détectée
        person = persons[0]
        
        # 2. Feature engineering
        pose_landmarks = person.pose_landmarks
        left_hand = person.left_hand_landmarks  
        right_hand = person.right_hand_landmarks
        
        body_features = self.feature_engineer.extract_body_features(pose_landmarks)
        hand_features = self.feature_engineer.extract_hand_features(left_hand, right_hand)
        combined_features = np.concatenate([body_features, hand_features])
        
        # Normalisation si scaler disponible
        if self.feature_engineer.scaler is not None:
            combined_features = self.feature_engineer.scaler.transform([combined_features])[0]
        
        # 3. Prédiction du modèle
        pose_probs, face_pos_pred = self.model.predict(combined_features.reshape(1, -1))
        pose_probs = pose_probs[0]  # Retirer la dimension batch
        face_pos_pred = face_pos_pred[0]
        
        # Lissage temporel
        pose_probs_smoothed = self.smooth_predictions(pose_probs)
        
        # Classe prédite
        predicted_idx = np.argmax(pose_probs_smoothed)
        predicted_class = self.model.classes[predicted_idx]
        confidence = pose_probs_smoothed[predicted_idx]
        
        # 4. Détection de saut
        face_center = person.face_center
        self.face_positions_history.append(face_center)
        
        jump_detected = self.detect_jump(face_center[1]) if face_center else False
        jump_status = "SAUT!" if jump_detected else "Normal"
        
        # 5. Décision de capture photo
        should_capture, capture_reason = self.should_take_photo(
            confidence, predicted_class, face_center
        )
        
        if should_capture and not self.paused:
            self.capture_photo(frame, predicted_class, confidence, capture_reason)
        
        # 6. Mise à jour des infos
        info.update({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'face_position': face_center,
            'jump_detected': jump_detected,
            'jump_status': jump_status,
            'should_capture': should_capture,
            'capture_reason': capture_reason
        })
        
        # 7. Interface utilisateur
        overlay = self.draw_interface(frame, persons, pose_probs_smoothed, 
                                    face_pos_pred, predicted_class, confidence, 
                                    self.calculate_fps(), jump_status)
        
        return overlay, info
    
    def calculate_fps(self) -> float:
        """Calcule les FPS moyens"""
        current_time = time.time()
        self.fps_counter.append(current_time)
        
        if len(self.fps_counter) > 1:
            time_diff = self.fps_counter[-1] - self.fps_counter[0]
            fps = (len(self.fps_counter) - 1) / time_diff if time_diff > 0 else 0
            return fps
        
        return 0.0
    
    def handle_keyboard(self, key: int) -> bool:
        """
        Gère les entrées clavier
        
        Args:
            key: Code de la touche
            
        Returns:
            False si doit quitter, True sinon
        """
        if key == ord('q') or key == 27:  # 'q' ou Échap
            return False
        
        elif key == ord(' '):  # Espace
            self.paused = not self.paused
            print(f"{'⏸️  Pause' if self.paused else '▶️  Reprise'}")
        
        elif key == ord('k'):  # Toggle keypoints
            self.show_keypoints = not self.show_keypoints
            print(f"Keypoints: {'ON' if self.show_keypoints else 'OFF'}")
        
        elif key == ord('c'):  # Augmenter confiance
            self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
            print(f"Seuil confiance: {self.confidence_threshold:.2f}")
        
        elif key == ord('v'):  # Diminuer confiance
            self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
            print(f"Seuil confiance: {self.confidence_threshold:.2f}")
        
        elif key == ord('j'):  # Augmenter seuil saut
            self.jump_threshold = min(0.5, self.jump_threshold + 0.02)
            print(f"Seuil saut: {self.jump_threshold:.2f}")
        
        elif key == ord('h'):  # Diminuer seuil saut
            self.jump_threshold = max(0.05, self.jump_threshold - 0.02)
            print(f"Seuil saut: {self.jump_threshold:.2f}")
        
        elif key == ord('d'):  # Toggle debug features
            self.show_features_debug = not self.show_features_debug
            print(f"Debug features: {'ON' if self.show_features_debug else 'OFF'}")
        
        return True
    
    def run(self, camera_id: int = 0):
        """
        Lance le pipeline temps réel
        
        Args:
            camera_id: ID de la caméra (0 par défaut)
        """
        print(f"🎥 Démarrage de la caméra {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la caméra {camera_id}")
        
        # Configuration de la caméra
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎯 Pipeline démarré ! Appuyez sur 'q' pour quitter")
        print("📋 Commandes:")
        print("   ESPACE: Pause/Reprise")
        print("   K: Afficher/Masquer keypoints")  
        print("   C/V: Ajuster seuil de confiance")
        print("   J/H: Ajuster seuil de saut")
        print("   Q: Quitter")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur de lecture caméra")
                    break
                
                self.frame_count += 1
                
                # Traiter la frame
                if not self.paused:
                    processed_frame, info = self.process_frame(frame)
                else:
                    processed_frame = frame
                    cv2.putText(processed_frame, "PAUSE", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                # Afficher
                cv2.imshow('Pose Detection Pipeline', processed_frame)
                
                # Gestion clavier
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard(key):
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Arrêt demandé par l'utilisateur")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Pipeline arrêté")
            print(f"📊 Statistiques finales:")
            print(f"   - Frames traitées: {self.frame_count}")
            print(f"   - Temps total: {time.time() - self.start_time:.1f}s")
            print(f"   - Photos dans: {self.photos_dir}")


# Exemple d'utilisation
if __name__ == "__main__":
    
    # Configuration des chemins
    MODEL_PATH = "../../models"
    SCALER_PATH = "../../data/features/feature_scaler.joblib"
    
    try:
        # Initialiser le pipeline
        pipeline = RealTimePosePipeline(
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            confidence_threshold=0.7,  # Ajustable avec C/V
            jump_threshold=0.15,       # Ajustable avec J/H
            smoothing_window=5
        )
        
        # Lancer la détection temps réel
        pipeline.run(camera_id=0)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Vérifiez que:")
        print("  - Le modèle est bien sauvegardé dans ../../models")
        print("  - Le scaler est dans ../../data/features/")
        print("  - Une caméra est connectée")