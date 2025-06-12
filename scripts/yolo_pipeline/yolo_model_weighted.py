# scripts/yolo_pipeline/yolo_model_weighted.py
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp
from typing import Dict, Tuple, Optional, List

class WeightedDanceClassifier(nn.Module):
    """Classificateur avec pondération optimisée des features"""
    def __init__(self, pose_features=51, hand_features=126, context_features=15, num_classes=7):
        super().__init__()
        
        # Dimensions des features
        self.pose_features = pose_features
        self.hand_features = hand_features
        self.context_features = context_features
        
        # 🎯 PONDÉRATION DES FEATURES
        # Pose YOLO : Maximum de poids (très important)
        self.pose_weight = 3.0
        
        # Mains : Poids réduit (informatif mais pas crucial)
        self.hand_weight = 0.5
        
        # Context : Poids modéré (complémentaire)
        self.context_weight = 1.0
        
        # Transformation des features avec pondération
        self.pose_transform = nn.Sequential(
            nn.Linear(pose_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.hand_transform = nn.Sequential(
            nn.Linear(hand_features, 64),  # Dimension réduite pour les mains
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # Plus de dropout sur les mains
        )
        
        self.context_transform = nn.Sequential(
            nn.Linear(context_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # Fusion pondérée des features
        fusion_dim = 128 + 64 + 32  # 224 features fusionnées
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialisation des poids avec emphase sur les features de pose"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Amplifier les poids de la transformation pose
        with torch.no_grad():
            for param in self.pose_transform.parameters():
                if param.dim() > 1:  # Poids (pas les biais)
                    param.mul_(1.2)  # Amplification des poids pose
    
    def forward(self, x):
        # Séparer les features
        pose_features = x[:, :self.pose_features]
        hand_features = x[:, self.pose_features:self.pose_features + self.hand_features]
        context_features = x[:, self.pose_features + self.hand_features:]
        
        # Transformer chaque type de feature
        pose_out = self.pose_transform(pose_features)
        hand_out = self.hand_transform(hand_features)
        context_out = self.context_transform(context_features)
        
        # Appliquer la pondération
        pose_weighted = pose_out * self.pose_weight
        hand_weighted = hand_out * self.hand_weight
        context_weighted = context_out * self.context_weight
        
        # Fusion
        fused_features = torch.cat([pose_weighted, hand_weighted, context_weighted], dim=1)
        
        # Classification finale
        output = self.fusion_layer(fused_features)
        
        return output

class YoloDanceSystemWeighted:
    """
    Système YOLO Dance avec pondération optimisée
    - YOLO pose : Poids maximum (3.0x)
    - Mains MediaPipe : Poids réduit (0.5x)
    - Context : Poids modéré (1.0x)
    """
    def __init__(self, pose_model_path='yolov8n-pose.pt', custom_classifier_path=None):
        print("🚀 Initialisation système YOLO Dance pondéré...")
        
        # Classes
        self.classes = [
            'hands_up', 'dab', 'twerk',
            'jul', 'neutral',
            'crossarm'
        ]
        
        # 1. YOLO pour pose corporelle (PRIORITÉ MAXIMALE)
        self.pose_detector = YOLO(pose_model_path)
        print("✅ YOLOv8 pose chargé (poids 3.0x)")
        
        # 2. MediaPipe pour mains (POIDS RÉDUIT)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Seuil réduit car moins important
            min_tracking_confidence=0.3
        )
        print("✅ MediaPipe Hands chargé (poids 0.5x)")
        
        # 3. Classificateur pondéré
        if custom_classifier_path:
            try:
                self.classifier = self._build_weighted_classifier()
                state_dict = torch.load(custom_classifier_path, map_location='cpu')
                
                # Adapter les anciens modèles au nouveau format
                if 'model_state_dict' in state_dict:
                    model_weights = state_dict['model_state_dict']
                else:
                    model_weights = state_dict
                
                # Essayer de charger les poids
                try:
                    self.classifier.load_state_dict(model_weights)
                    print(f"✅ Classificateur pondéré chargé: {custom_classifier_path}")
                except:
                    print("⚠️ Modèle incompatible, utilisation du classificateur non-entraîné")
                
                self.classifier.eval()
            except Exception as e:
                print(f"⚠️ Erreur chargement modèle: {e}")
                self.classifier = self._build_weighted_classifier()
        else:
            self.classifier = self._build_weighted_classifier()
            print("⚠️ Classificateur pondéré non-entraîné")
    
    def _build_weighted_classifier(self):
        """Construit le classificateur pondéré"""
        return WeightedDanceClassifier(
            pose_features=51,      # YOLO pose (poids 3.0x)
            hand_features=126,     # MediaPipe mains (poids 0.5x)
            context_features=15,   # Context (poids 1.0x)
            num_classes=len(self.classes)
        )
    
    def extract_pose_features(self, yolo_results, person_idx=0) -> np.ndarray:
        """Extrait features de pose YOLO (PRIORITÉ MAXIMALE)"""
        if not yolo_results or len(yolo_results) == 0:
            return np.zeros(51)
        
        result = yolo_results[0]
        
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return np.zeros(51)
        
        if person_idx >= len(result.keypoints.data):
            return np.zeros(51)
        
        # Keypoints YOLO (17 points × 3 coordonnées)
        keypoints = result.keypoints.data[person_idx].cpu().numpy()
        
        # Normalisation améliorée pour la pose
        if len(result.boxes) > person_idx:
            bbox = result.boxes.xyxy[person_idx].cpu().numpy()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            if w > 0 and h > 0:
                # Normalisation relative à la bbox
                keypoints[:, 0] = (keypoints[:, 0] - bbox[0]) / w
                keypoints[:, 1] = (keypoints[:, 1] - bbox[1]) / h
                
                # Filtrage des keypoints de faible confiance
                confidence_mask = keypoints[:, 2] > 0.3
                keypoints[~confidence_mask] = 0  # Mettre à zéro les keypoints peu fiables
        
        return keypoints.flatten()
    
    def extract_hand_features(self, image, bbox=None) -> np.ndarray:
        """Extrait features de mains (POIDS RÉDUIT)"""
        hand_features = np.zeros(126)
        
        try:
            # ROI plus large pour compenser la réduction de priorité
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                margin = 80  # Marge plus large
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image.shape[1], x2 + margin)
                y2 = min(image.shape[0], y2 + margin)
                
                roi = image[y1:y2, x1:x2]
                if roi.size == 0:
                    roi = image
            else:
                roi = image
            
            # Détection MediaPipe avec seuils réduits
            rgb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_idx, (hand_landmarks, handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    if hand_idx >= 2:
                        break
                    
                    is_right_hand = handedness.classification[0].label == 'Right'
                    start_idx = 0 if is_right_hand else 63
                    
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        if i < 21:
                            base_idx = start_idx + i * 3
                            if bbox is not None:
                                landmark.x = (landmark.x * (x2 - x1) + x1) / image.shape[1]
                                landmark.y = (landmark.y * (y2 - y1) + y1) / image.shape[0]
                            
                            # Réduction de l'importance des features de mains
                            hand_features[base_idx] = landmark.x * 0.8      # Atténuation
                            hand_features[base_idx + 1] = landmark.y * 0.8  # Atténuation
                            hand_features[base_idx + 2] = landmark.z * 0.5  # Z moins important
        
        except Exception as e:
            print(f"⚠️ Erreur mains (impact réduit): {e}")
        
        return hand_features
    
    def extract_context_features(self, yolo_results, image_shape, hand_results=None, person_idx=0) -> np.ndarray:
        """Features contextuelles (poids modéré)"""
        context = np.zeros(15)
        
        if not yolo_results or len(yolo_results) == 0:
            return context
        
        result = yolo_results[0]
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) <= person_idx:
            return context
        
        bbox = result.boxes.xyxy[person_idx].cpu().numpy()
        h_img, w_img = image_shape[:2]
        
        # Features corporelles (plus importantes)
        context[0] = (bbox[2] - bbox[0]) / w_img  # width ratio
        context[1] = (bbox[3] - bbox[1]) / h_img  # height ratio
        context[2] = bbox[0] / w_img  # x position
        context[3] = bbox[1] / h_img  # y position
        context[4] = (bbox[2] - bbox[0]) / max(bbox[3] - bbox[1], 1)  # aspect ratio
        context[5] = result.boxes.conf[person_idx].cpu().numpy()  # confidence YOLO
        
        # Position du corps dans l'image
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        context[6] = center_x / w_img
        context[7] = center_y / h_img
        
        person_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        context[8] = person_area / (w_img * h_img)
        context[9] = min((bbox[2] - bbox[0]) / w_img, 1.0)
        
        # Features mains (impact réduit)
        if hand_results and hand_results.multi_hand_landmarks:
            # Nombre de mains (moins important)
            context[10] = min(len(hand_results.multi_hand_landmarks) * 0.5, 1.0)  # Atténuation
            
            # Positions moyennes (réduites)
            if hand_results.multi_hand_landmarks:
                avg_hand_x = 0
                avg_hand_y = 0
                hand_count = 0
                
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    center_landmark = hand_landmarks.landmark[9]
                    avg_hand_x += center_landmark.x
                    avg_hand_y += center_landmark.y
                    hand_count += 1
                
                if hand_count > 0:
                    context[11] = (avg_hand_x / hand_count) * 0.7  # Atténuation
                    context[12] = (avg_hand_y / hand_count) * 0.7  # Atténuation
                    
                    if hand_count == 2:
                        hand1_center = hand_results.multi_hand_landmarks[0].landmark[9]
                        hand2_center = hand_results.multi_hand_landmarks[1].landmark[9]
                        distance = np.sqrt(
                            (hand1_center.x - hand2_center.x)**2 + 
                            (hand1_center.y - hand2_center.y)**2
                        )
                        context[13] = distance * 0.6  # Atténuation
                        context[14] = 0.3  # Symétrie réduite
        
        return context
    
    def predict(self, image: np.ndarray) -> Dict:
        """Prédiction avec pondération optimisée"""
        # 1. Détection pose YOLO (PRIORITÉ)
        yolo_results = self.pose_detector(image)
        
        # 2. Détection mains (réduite)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_image)
        
        # 3. Extraction features pondérées
        pose_features = self.extract_pose_features(yolo_results)  # Poids 3.0x
        hand_features = self.extract_hand_features(image)        # Poids 0.5x
        context_features = self.extract_context_features(        # Poids 1.0x
            yolo_results, image.shape, hand_results
        )
        
        # 4. Concaténation (la pondération se fait dans le modèle)
        all_features = np.concatenate([pose_features, hand_features, context_features])
        
        if all_features.size != 192:
            if all_features.size < 192:
                padding = np.zeros(192 - all_features.size)
                all_features = np.concatenate([all_features, padding])
            else:
                all_features = all_features[:192]
        
        # 5. Classification pondérée
        self.classifier.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(all_features).unsqueeze(0)
            logits = self.classifier(features_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze().numpy()
        
        # 6. Résultats
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.classes[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                cls: float(prob) for cls, prob in zip(self.classes, probabilities)
            },
            'debug_info': {
                'pose_features_weight': 3.0,
                'hand_features_weight': 0.5,
                'context_features_weight': 1.0,
                'pose_features_size': pose_features.size,
                'hand_features_size': hand_features.size,
                'context_features_size': context_features.size,
                'hands_detected': len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
            }
        }

# Fonction utilitaire pour convertir un ancien modèle
def convert_old_model_to_weighted(old_model_path, new_model_path):
    """Convertit un ancien modèle vers le format pondéré"""
    print("🔄 Conversion vers modèle pondéré...")
    
    # Charger l'ancien modèle
    old_data = torch.load(old_model_path, map_location='cpu')
    
    # Créer le nouveau modèle
    new_classifier = WeightedDanceClassifier()
    
    # Essayer de mapper les anciens poids (si possible)
    # Sinon, garder l'initialisation aléatoire pondérée
    
    # Sauvegarder le nouveau format
    torch.save({
        'model_state_dict': new_classifier.state_dict(),
        'classes': ['hands_up', 'dab', 'twerk', 'jul', 'neutral', 'crossarm'],
        'config': {
            'pose_features': 51,
            'hand_features': 126,
            'context_features': 15,
            'num_classes': 7,
            'pose_weight': 3.0,
            'hand_weight': 0.5,
            'context_weight': 1.0
        },
        'model_type': 'weighted_dance_classifier'
    }, new_model_path)
    
    print(f"✅ Modèle pondéré sauvegardé: {new_model_path}")

if __name__ == "__main__":
    # Test du système pondéré
    system = YoloDanceSystemWeighted()
    print("🎯 Système de pondération:")
    print("   - Pose YOLO: 3.0x (priorité maximale)")
    print("   - Mains MediaPipe: 0.5x (informatif)")
    print("   - Context: 1.0x (modéré)")