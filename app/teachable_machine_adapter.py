#app/teachable_machine_adapter_simple.py

import numpy as np
import cv2
import json
import os
from typing import List, Dict, Tuple, Optional
import mediapipe as mp

class TeachableMachinePoseDetector:
    def __init__(self, model_dir: str):
        """
        Adaptateur simplifié pour utiliser un modèle Teachable Machine Pose avec MediaPipe
        Évite les conflits de dépendances en utilisant uniquement MediaPipe
        
        Args:
            model_dir: Chemin vers le dossier contenant metadata.json
        """
        self.model_dir = model_dir
        self.metadata = None
        self.labels = []
        
        # Initialiser MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Charger les métadonnées
        self._load_metadata()
        
        # Règles de détection basées sur les keypoints MediaPipe
        self._setup_gesture_rules()
    
    def _load_metadata(self):
        """Charge les métadonnées du modèle"""
        try:
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.labels = self.metadata['labels']
            print(f"Labels détectés: {self.labels}")
            
        except Exception as e:
            print(f"Erreur lors du chargement des métadonnées: {e}")
            # Labels par défaut basés sur votre modèle
            self.labels = ["Twerk", "RussianMoove", "PasDuBourré", "V_Signs", "Funk", 
                          "Floss", "Macaréna", "Dab", "CrossArm", "CrossFeeat", 
                          "GlassMoove", "Fuck", "JulSign", "Neutral", "Blood"]
    
    def _setup_gesture_rules(self):
        """Configure les règles de détection des gestes basées sur les keypoints"""
        # Index des keypoints MediaPipe importants
        self.KEYPOINT_INDICES = {
            'nose': 0,
            'left_eye': 2, 'right_eye': 5,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
    
    def extract_keypoints(self, frame):
        """Extrait les keypoints avec MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            keypoints = []
            confidences = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y])
                confidences.append(landmark.visibility)
            return np.array(keypoints), np.array(confidences), results.pose_landmarks
        return None, None, None
    
    def _detect_dab(self, keypoints, confidences):
        """Détecte le geste Dab"""
        if len(keypoints) < 34:  # 17 points * 2 coords
            return False, 0.0
        
        # Coordonnées des points clés
        left_shoulder = keypoints[22:24]   # 11 * 2
        right_shoulder = keypoints[24:26]  # 12 * 2
        left_elbow = keypoints[26:28]      # 13 * 2
        right_elbow = keypoints[28:30]     # 14 * 2
        left_wrist = keypoints[30:32]      # 15 * 2
        right_wrist = keypoints[32:34]     # 16 * 2
        
        # Vérifier la confiance des points
        min_confidence = 0.3
        key_confidences = confidences[11:17]  # Épaules, coudes, poignets
        if np.mean(key_confidences) < min_confidence:
            return False, 0.0
        
        # Logique Dab : un bras levé vers le visage, l'autre tendu
        # Bras gauche levé (dab classique)
        left_arm_up = left_wrist[1] < left_shoulder[1] - 0.1
        left_arm_across = abs(left_wrist[0] - left_shoulder[0]) > 0.1
        
        # Bras droit tendu
        right_arm_down = right_wrist[1] > right_shoulder[1]
        
        # Ou bras droit levé (dab inversé)
        right_arm_up = right_wrist[1] < right_shoulder[1] - 0.1
        right_arm_across = abs(right_wrist[0] - right_shoulder[0]) > 0.1
        
        # Bras gauche tendu
        left_arm_down = left_wrist[1] > left_shoulder[1]
        
        # Dab détecté si une des configurations est présente
        dab_left = left_arm_up and left_arm_across and right_arm_down
        dab_right = right_arm_up and right_arm_across and left_arm_down
        
        if dab_left or dab_right:
            confidence = np.mean(key_confidences)
            return True, confidence
        
        return False, 0.0
    
    def _detect_twerk(self, keypoints, confidences):
        """Détecte le geste Twerk"""
        if len(keypoints) < 34:
            return False, 0.0
        
        # Points des hanches et genoux
        left_hip = keypoints[46:48]     # 23 * 2
        right_hip = keypoints[48:50]    # 24 * 2
        left_knee = keypoints[50:52]    # 25 * 2
        right_knee = keypoints[52:54]   # 26 * 2
        
        # Vérifier la confiance
        hip_knee_confidences = confidences[23:27]
        if np.mean(hip_knee_confidences) < 0.3:
            return False, 0.0
        
        # Logique Twerk : hanches en arrière, genoux fléchis
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        knee_center_y = (left_knee[1] + right_knee[1]) / 2
        
        # Les genoux doivent être plus bas que les hanches (position accroupie)
        knees_bent = knee_center_y > hip_center_y + 0.05
        
        if knees_bent:
            confidence = np.mean(hip_knee_confidences)
            return True, confidence
        
        return False, 0.0
    
    def _detect_macarena(self, keypoints, confidences):
        """Détecte le geste Macaréna"""
        if len(keypoints) < 34:
            return False, 0.0
        
        left_shoulder = keypoints[22:24]
        right_shoulder = keypoints[24:26]
        left_wrist = keypoints[30:32]
        right_wrist = keypoints[32:34]
        
        arm_confidences = [confidences[11], confidences[12], confidences[15], confidences[16]]
        if np.mean(arm_confidences) < 0.3:
            return False, 0.0
        
        # Macaréna : bras tendus horizontalement
        shoulder_line_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Les poignets doivent être au niveau des épaules (±0.1)
        left_horizontal = abs(left_wrist[1] - shoulder_line_y) < 0.1
        right_horizontal = abs(right_wrist[1] - shoulder_line_y) < 0.1
        
        # Les bras doivent être écartés
        arms_spread = (right_wrist[0] - left_wrist[0]) > 0.3
        
        if left_horizontal and right_horizontal and arms_spread:
            confidence = np.mean(arm_confidences)
            return True, confidence
        
        return False, 0.0
    
    def _detect_v_signs(self, keypoints, confidences):
        """Détecte les signes V avec les mains"""
        if len(keypoints) < 34:
            return False, 0.0
        
        left_wrist = keypoints[30:32]
        right_wrist = keypoints[32:34]
        left_shoulder = keypoints[22:24]
        right_shoulder = keypoints[24:26]
        
        wrist_confidences = [confidences[15], confidences[16]]
        if np.mean(wrist_confidences) < 0.3:
            return False, 0.0
        
        # V Signs : mains levées près du visage/tête
        # Les poignets doivent être au-dessus des épaules
        left_raised = left_wrist[1] < left_shoulder[1] - 0.05
        right_raised = right_wrist[1] < right_shoulder[1] - 0.05
        
        # Et proches du centre (près du visage)
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        left_near_face = abs(left_wrist[0] - center_x) < 0.2
        right_near_face = abs(right_wrist[0] - center_x) < 0.2
        
        if left_raised and right_raised and (left_near_face or right_near_face):
            confidence = np.mean(wrist_confidences)
            return True, confidence
        
        return False, 0.0
    
    def _detect_cross_arm(self, keypoints, confidences):
        """Détecte les bras croisés"""
        if len(keypoints) < 34:
            return False, 0.0
        
        left_wrist = keypoints[30:32]
        right_wrist = keypoints[32:34]
        left_shoulder = keypoints[22:24]
        right_shoulder = keypoints[24:26]
        
        arm_confidences = [confidences[11], confidences[12], confidences[15], confidences[16]]
        if np.mean(arm_confidences) < 0.3:
            return False, 0.0
        
        # Bras croisés : poignet gauche vers épaule droite et vice versa
        left_crosses = left_wrist[0] > right_shoulder[0] - 0.1
        right_crosses = right_wrist[0] < left_shoulder[0] + 0.1
        
        # Les bras doivent être à hauteur de torse
        torso_level = (left_shoulder[1] + right_shoulder[1]) / 2
        left_at_torso = abs(left_wrist[1] - torso_level) < 0.15
        right_at_torso = abs(right_wrist[1] - torso_level) < 0.15
        
        if left_crosses and right_crosses and left_at_torso and right_at_torso:
            confidence = np.mean(arm_confidences)
            return True, confidence
        
        return False, 0.0
    
    def predict_pose(self, frame):
        """Prédit la pose à partir d'une frame"""
        keypoints, confidences, pose_landmarks = self.extract_keypoints(frame)
        
        if keypoints is None:
            return "Neutral", 0.0
        
        # Tester chaque geste dans l'ordre de priorité
        gestures_to_test = [
            ("Dab", self._detect_dab),
            ("Twerk", self._detect_twerk),
            ("Macaréna", self._detect_macarena),
            ("V_Signs", self._detect_v_signs),
            ("CrossArm", self._detect_cross_arm)
        ]
        
        best_gesture = "Neutral"
        best_confidence = 0.0
        
        for gesture_name, detect_func in gestures_to_test:
            detected, confidence = detect_func(keypoints, confidences)
            if detected and confidence > best_confidence:
                best_gesture = gesture_name
                best_confidence = confidence
        
        return best_gesture, best_confidence
    
    def get_all_predictions(self, frame):
        """Retourne toutes les prédictions avec leurs confidences"""
        keypoints, confidences, pose_landmarks = self.extract_keypoints(frame)
        
        if keypoints is None:
            return {label: 0.0 for label in self.labels}
        
        predictions = {}
        
        # Tester tous les gestes
        gestures_to_test = [
            ("Dab", self._detect_dab),
            ("Twerk", self._detect_twerk),
            ("Macaréna", self._detect_macarena),
            ("V_Signs", self._detect_v_signs),
            ("CrossArm", self._detect_cross_arm)
        ]
        
        for gesture_name, detect_func in gestures_to_test:
            detected, confidence = detect_func(keypoints, confidences)
            predictions[gesture_name] = confidence if detected else 0.0
        
        # Ajouter les autres labels avec confiance 0
        for label in self.labels:
            if label not in predictions:
                predictions[label] = 0.0
        
        # Si aucun geste détecté, Neutral = confiance élevée
        if all(conf == 0.0 for conf in predictions.values()):
            predictions["Neutral"] = 0.8
        
        return predictions
    
    def draw_keypoints(self, frame, keypoints=None):
        """Dessine les keypoints sur la frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            return annotated_frame
        
        return frame