"""
Feature Engineering pour la classification de poses et gestes
Transforme les keypoints MediaPipe en features discriminantes
"""

import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class EngineeredFeatures:
    """Structure pour stocker les features engineerées d'une personne"""
    # Features corporelles
    body_features: np.ndarray
    # Features des mains
    hand_features: np.ndarray
    # Features combinées
    combined_features: np.ndarray
    # Métadonnées
    confidence_score: float
    face_center: Tuple[float, float]
    person_id: int
    class_label: str
    image_path: str

class PoseFeatureEngineer:
    """
    Ingénieur de features pour poses et gestes
    """
    
    def __init__(self):
        # Indices des landmarks clés pour MediaPipe Pose (33 points)
        self.pose_landmarks_indices = {
            'nose': 0,
            'left_eye': 2, 'right_eye': 5,
            'left_ear': 7, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }
        
        # Indices des landmarks des mains (21 points chacune)
        self.hand_landmarks_indices = {
            'wrist': 0,
            'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
            'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
            'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
            'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
            'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
        }
        
        # Classes à classifier
        self.classes = ['hands_up', 'dab', 'twerk', 'mic_drop', 'middle_finger', 
                       'peace', 'heart', 'jule', 'neutral']
        
        # Scaler pour normalisation (sera ajusté pendant l'entraînement)
        self.scaler = None
        
    def _safe_get_landmark(self, landmarks: Optional[np.ndarray], index: int) -> np.ndarray:
        """Récupère un landmark de manière sécurisée"""
        if landmarks is None or index >= len(landmarks):
            return np.array([0.5, 0.5, 0.0])  # Point par défaut au centre
        return landmarks[index]
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calcule l'angle entre trois points (p1-p2-p3)
        Retourne l'angle en degrés
        """
        v1 = p1[:2] - p2[:2]  # Utiliser seulement x,y
        v2 = p3[:2] - p2[:2]
        
        # Éviter la division par zéro
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Éviter les erreurs d'arrondi
        
        angle = np.arccos(cos_angle) * 180 / np.pi
        return angle
    
    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calcule la distance euclidienne entre deux points"""
        return np.linalg.norm(p1[:2] - p2[:2])
    
    def _normalize_landmarks(self, landmarks: np.ndarray, reference_points: List[int]) -> np.ndarray:
        """
        Normalise les landmarks par rapport à des points de référence
        (pour être invariant à la taille de la personne)
        """
        if landmarks is None or len(landmarks) == 0:
            return landmarks
        
        # Calculer la distance de référence (ex: largeur des épaules)
        ref_points = [landmarks[i] for i in reference_points if i < len(landmarks)]
        if len(ref_points) < 2:
            return landmarks
        
        reference_distance = self._calculate_distance(ref_points[0], ref_points[1])
        if reference_distance == 0:
            return landmarks
        
        # Normaliser toutes les coordonnées
        normalized = landmarks.copy()
        normalized[:, :2] = normalized[:, :2] / reference_distance
        
        return normalized
    
    def extract_body_features(self, pose_landmarks: Optional[np.ndarray]) -> np.ndarray:
        """
        Extrait les features corporelles pertinentes pour la classification
        """
        features = []
        
        if pose_landmarks is None:
            return np.zeros(50)  # Features par défaut
        
        # Normaliser par rapport à la largeur des épaules
        normalized_pose = self._normalize_landmarks(
            pose_landmarks, 
            [self.pose_landmarks_indices['left_shoulder'], self.pose_landmarks_indices['right_shoulder']]
        )
        
        # 1. Angles des bras (crucial pour hands_up, dab, etc.)
        left_shoulder = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['left_shoulder'])
        left_elbow = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['left_elbow'])
        left_wrist = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['left_wrist'])
        
        right_shoulder = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['right_shoulder'])
        right_elbow = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['right_elbow'])
        right_wrist = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['right_wrist'])
        
        # Angles des coudes
        left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        features.extend([left_elbow_angle, right_elbow_angle])
        
        # Angles des épaules (bras par rapport au torse)
        left_hip = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['left_hip'])
        right_hip = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['right_hip'])
        
        left_shoulder_angle = self._calculate_angle(left_hip, left_shoulder, left_elbow)
        right_shoulder_angle = self._calculate_angle(right_hip, right_shoulder, right_elbow)
        features.extend([left_shoulder_angle, right_shoulder_angle])
        
        # 2. Positions relatives des mains par rapport au corps
        nose = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['nose'])
        
        # Hauteur des mains par rapport à la tête
        left_hand_height = (nose[1] - left_wrist[1])  # y inversé en image
        right_hand_height = (nose[1] - right_wrist[1])
        features.extend([left_hand_height, right_hand_height])
        
        # Distance des mains par rapport au centre du corps
        body_center = (left_shoulder + right_shoulder) / 2
        left_hand_distance_to_center = self._calculate_distance(left_wrist, body_center)
        right_hand_distance_to_center = self._calculate_distance(right_wrist, body_center)
        features.extend([left_hand_distance_to_center, right_hand_distance_to_center])
        
        # 3. Symétrie corporelle
        symmetry_shoulder_y = abs(left_shoulder[1] - right_shoulder[1])
        symmetry_elbow_y = abs(left_elbow[1] - right_elbow[1])
        symmetry_wrist_y = abs(left_wrist[1] - right_wrist[1])
        features.extend([symmetry_shoulder_y, symmetry_elbow_y, symmetry_wrist_y])
        
        # 4. Largeur et position des bras
        arm_span = self._calculate_distance(left_wrist, right_wrist)
        shoulder_width = self._calculate_distance(left_shoulder, right_shoulder)
        arm_span_ratio = arm_span / (shoulder_width + 1e-6)
        features.append(arm_span_ratio)
        
        # 5. Angles du torse (pour twerk, dab)
        torso_angle = self._calculate_angle(
            (left_shoulder + right_shoulder) / 2,
            (left_hip + right_hip) / 2,
            nose
        )
        features.append(torso_angle)
        
        # 6. Position des hanches (pour twerk)
        left_knee = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['left_knee'])
        right_knee = self._safe_get_landmark(normalized_pose, self.pose_landmarks_indices['right_knee'])
        
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
        features.extend([left_hip_angle, right_hip_angle])
        
        # Compléter avec des zéros si nécessaire
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])  # Limiter à 50 features
    
    def extract_hand_features(self, left_hand: Optional[np.ndarray], 
                            right_hand: Optional[np.ndarray]) -> np.ndarray:
        """
        Extrait les features des mains (crucial pour peace, heart, middle_finger)
        """
        features = []
        
        # Features pour chaque main
        for hand_landmarks, hand_name in [(left_hand, 'left'), (right_hand, 'right')]:
            
            if hand_landmarks is None:
                # Features par défaut si main non détectée
                hand_features = np.zeros(25)
            else:
                hand_features = self._extract_single_hand_features(hand_landmarks)
            
            features.extend(hand_features)
        
        # Features de relation entre les deux mains
        if left_hand is not None and right_hand is not None:
            relation_features = self._extract_hands_relation_features(left_hand, right_hand)
            features.extend(relation_features)
        else:
            features.extend(np.zeros(10))  # Features de relation par défaut
        
        return np.array(features)
    
    def _extract_single_hand_features(self, hand_landmarks: np.ndarray) -> np.ndarray:
        """Extrait les features d'une seule main"""
        features = []
        
        wrist = hand_landmarks[self.hand_landmarks_indices['wrist']]
        
        # 1. Doigts étendus vs pliés (crucial pour peace, middle_finger)
        finger_tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        finger_mcps = ['thumb_cmc', 'index_mcp', 'middle_mcp', 'ring_mcp', 'pinky_mcp']
        
        for tip_name, mcp_name in zip(finger_tips, finger_mcps):
            tip = hand_landmarks[self.hand_landmarks_indices[tip_name]]
            mcp = hand_landmarks[self.hand_landmarks_indices[mcp_name]]
            
            # Distance du bout du doigt au poignet vs articulation au poignet
            tip_to_wrist = self._calculate_distance(tip, wrist)
            mcp_to_wrist = self._calculate_distance(mcp, wrist)
            
            # Ratio > 1 = doigt étendu, < 1 = doigt plié
            extension_ratio = tip_to_wrist / (mcp_to_wrist + 1e-6)
            features.append(extension_ratio)
        
        # 2. Angles entre doigts (pour peace sign)
        index_tip = hand_landmarks[self.hand_landmarks_indices['index_tip']]
        middle_tip = hand_landmarks[self.hand_landmarks_indices['middle_tip']]
        ring_tip = hand_landmarks[self.hand_landmarks_indices['ring_tip']]
        pinky_tip = hand_landmarks[self.hand_landmarks_indices['pinky_tip']]
        
        # Angle entre index et majeur (peace sign)
        index_middle_angle = self._calculate_angle(index_tip, wrist, middle_tip)
        features.append(index_middle_angle)
        
        # Angle entre majeur et annulaire
        middle_ring_angle = self._calculate_angle(middle_tip, wrist, ring_tip)
        features.append(middle_ring_angle)
        
        # 3. Forme globale de la main
        # Bounding box
        x_coords = hand_landmarks[:, 0]
        y_coords = hand_landmarks[:, 1]
        
        hand_width = np.max(x_coords) - np.min(x_coords)
        hand_height = np.max(y_coords) - np.min(y_coords)
        aspect_ratio = hand_width / (hand_height + 1e-6)
        features.append(aspect_ratio)
        
        # 4. Position du pouce par rapport aux autres doigts
        thumb_tip = hand_landmarks[self.hand_landmarks_indices['thumb_tip']]
        
        # Distance du pouce aux autres doigts
        thumb_to_index = self._calculate_distance(thumb_tip, index_tip)
        thumb_to_middle = self._calculate_distance(thumb_tip, middle_tip)
        features.extend([thumb_to_index, thumb_to_middle])
        
        # 5. Courbure des doigts
        for finger in ['index', 'middle', 'ring', 'pinky']:
            mcp = hand_landmarks[self.hand_landmarks_indices[f'{finger}_mcp']]
            pip = hand_landmarks[self.hand_landmarks_indices[f'{finger}_pip']]
            dip = hand_landmarks[self.hand_landmarks_indices[f'{finger}_dip']]
            tip = hand_landmarks[self.hand_landmarks_indices[f'{finger}_tip']]
            
            # Angle de courbure du doigt
            finger_angle = self._calculate_angle(mcp, pip, tip)
            features.append(finger_angle)
        
        # Compléter ou tronquer à 25 features
        while len(features) < 25:
            features.append(0.0)
        
        return np.array(features[:25])
    
    def _extract_hands_relation_features(self, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
        """Extrait les features de relation entre les deux mains"""
        features = []
        
        left_wrist = left_hand[self.hand_landmarks_indices['wrist']]
        right_wrist = right_hand[self.hand_landmarks_indices['wrist']]
        
        # Distance entre les mains
        hands_distance = self._calculate_distance(left_wrist, right_wrist)
        features.append(hands_distance)
        
        # Position relative (pour heart shape)
        relative_x = right_wrist[0] - left_wrist[0]
        relative_y = right_wrist[1] - left_wrist[1]
        features.extend([relative_x, relative_y])
        
        # Symétrie des gestes
        left_index_tip = left_hand[self.hand_landmarks_indices['index_tip']]
        right_index_tip = right_hand[self.hand_landmarks_indices['index_tip']]
        
        # Hauteur relative des index
        index_height_diff = abs(left_index_tip[1] - right_index_tip[1])
        features.append(index_height_diff)
        
        # Orientation similaire des mains
        left_direction = left_hand[self.hand_landmarks_indices['middle_tip']] - left_wrist
        right_direction = right_hand[self.hand_landmarks_indices['middle_tip']] - right_wrist
        
        # Similarité de direction (produit scalaire normalisé)
        left_norm = np.linalg.norm(left_direction[:2])
        right_norm = np.linalg.norm(right_direction[:2])
        
        if left_norm > 0 and right_norm > 0:
            direction_similarity = np.dot(left_direction[:2], right_direction[:2]) / (left_norm * right_norm)
        else:
            direction_similarity = 0.0
        
        features.append(direction_similarity)
        
        # Compléter à 10 features
        while len(features) < 10:
            features.append(0.0)
        
        return np.array(features[:10])
    
    def engineer_features_from_keypoints(self, keypoints_data: Dict) -> List[EngineeredFeatures]:
        """
        Transforme les keypoints bruts en features engineerées
        
        Args:
            keypoints_data: Données du keypoint_extractor
            
        Returns:
            Liste des features engineerées
        """
        
        all_features = []
        
        for class_name, class_data in keypoints_data.items():
            print(f"🔧 Engineering features pour classe: {class_name}")
            
            for sample in class_data:
                for person_data in sample['persons']:
                    
                    # Reconstruction des arrays numpy
                    pose_landmarks = np.array(person_data['pose_landmarks']) if person_data['pose_landmarks'] else None
                    left_hand = np.array(person_data['left_hand_landmarks']) if person_data['left_hand_landmarks'] else None
                    right_hand = np.array(person_data['right_hand_landmarks']) if person_data['right_hand_landmarks'] else None
                    
                    # Extraction des features
                    body_features = self.extract_body_features(pose_landmarks)
                    hand_features = self.extract_hand_features(left_hand, right_hand)
                    
                    # Combinaison des features
                    combined_features = np.concatenate([body_features, hand_features])
                    
                    # Création de l'objet EngineeredFeatures
                    engineered = EngineeredFeatures(
                        body_features=body_features,
                        hand_features=hand_features,
                        combined_features=combined_features,
                        confidence_score=person_data['confidence_score'],
                        face_center=tuple(person_data['face_center']),
                        person_id=person_data['person_id'],
                        class_label=class_name,
                        image_path=sample['image_path']
                    )
                    
                    all_features.append(engineered)
        
        return all_features
    
    def fit_scaler(self, features_list: List[EngineeredFeatures]):
        """Ajuste le scaler sur les données d'entraînement"""
        all_combined_features = np.array([f.combined_features for f in features_list])
        self.scaler = StandardScaler()
        self.scaler.fit(all_combined_features)
        print(f"📏 Scaler ajusté sur {len(features_list)} échantillons")
    
    def normalize_features(self, features_list: List[EngineeredFeatures]) -> List[EngineeredFeatures]:
        """Normalise les features avec le scaler"""
        if self.scaler is None:
            raise ValueError("Le scaler doit être ajusté avant la normalisation (utilisez fit_scaler)")
        
        for features in features_list:
            features.combined_features = self.scaler.transform([features.combined_features])[0]
        
        return features_list
    
    def save_features_and_scaler(self, features_list: List[EngineeredFeatures], 
                                output_path: str):
        """Sauvegarde les features et le scaler"""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Préparer les données pour sauvegarde
        features_data = []
        for f in features_list:
            features_data.append({
                'body_features': f.body_features.tolist(),
                'hand_features': f.hand_features.tolist(),
                'combined_features': f.combined_features.tolist(),
                'confidence_score': f.confidence_score,
                'face_center': f.face_center,
                'person_id': f.person_id,
                'class_label': f.class_label,
                'image_path': f.image_path
            })
        
        # Sauvegarder les features
        features_file = os.path.join(output_path, 'engineered_features.json')
        with open(features_file, 'w') as f:
            json.dump(features_data, f, indent=2)
        
        # Sauvegarder le scaler
        scaler_file = os.path.join(output_path, 'feature_scaler.joblib')
        joblib.dump(self.scaler, scaler_file)
        
        print(f"💾 Features sauvegardées: {features_file}")
        print(f"💾 Scaler sauvegardé: {scaler_file}")
        
        # Statistiques
        print(f"\n📊 Statistiques des features:")
        feature_dims = len(features_list[0].combined_features)
        print(f"  - Dimension des features: {feature_dims}")
        print(f"  - Nombre d'échantillons: {len(features_list)}")
        
        # Distribution par classe
        class_counts = {}
        for f in features_list:
            class_counts[f.class_label] = class_counts.get(f.class_label, 0) + 1
        
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} échantillons")
    
    def load_features_and_scaler(self, features_path: str, scaler_path: str) -> List[EngineeredFeatures]:
        """Charge les features et le scaler depuis les fichiers"""
        
        # Charger le scaler
        self.scaler = joblib.load(scaler_path)
        
        # Charger les features
        with open(features_path, 'r') as f:
            features_data = json.load(f)
        
        features_list = []
        for data in features_data:
            features = EngineeredFeatures(
                body_features=np.array(data['body_features']),
                hand_features=np.array(data['hand_features']),
                combined_features=np.array(data['combined_features']),
                confidence_score=data['confidence_score'],
                face_center=tuple(data['face_center']),
                person_id=data['person_id'],
                class_label=data['class_label'],
                image_path=data['image_path']
            )
            features_list.append(features)
        
        print(f"✅ Features chargées: {len(features_list)} échantillons")
        return features_list


# Exemple d'utilisation
if __name__ == "__main__":
    
    # Chemins
    KEYPOINTS_PATH = "../../data/keypoints/dataset_keypoints.json"
    OUTPUT_PATH = "../../data/features"
    
    # Charger les keypoints
    print("📂 Chargement des keypoints...")
    with open(KEYPOINTS_PATH, 'r') as f:
        keypoints_data = json.load(f)
    
    # Initialiser l'ingénieur de features
    engineer = PoseFeatureEngineer()
    
    # Engineer les features
    print("🔧 Engineering des features...")
    features_list = engineer.engineer_features_from_keypoints(keypoints_data)
    
    # Ajuster et normaliser
    print("📏 Normalisation des features...")
    engineer.fit_scaler(features_list)
    features_list = engineer.normalize_features(features_list)
    
    # Sauvegarder
    engineer.save_features_and_scaler(features_list, OUTPUT_PATH)
    
    print("✅ Feature engineering terminé!")
    
    # Test de rechargement
    print("\n🧪 Test de rechargement...")
    reloaded_features = engineer.load_features_and_scaler(
        os.path.join(OUTPUT_PATH, 'engineered_features.json'),
        os.path.join(OUTPUT_PATH, 'feature_scaler.joblib')
    )
    
    print(f"✅ Rechargement réussi: {len(reloaded_features)} échantillons")