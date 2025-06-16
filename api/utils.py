import cv2
import numpy as np
import math
from typing import Tuple, Optional, Dict

class GestureFilterSystem:
    """Système de filtres visuels pour les gestes détectés"""
    
    def __init__(self):
        self.filters_enabled = True
        self.filter_opacity = 0.8
        
        # Couleurs pour les filtres générés
        self.colors = {
            'peach': (255, 182, 137),  # Couleur pêche
            'pink': (255, 101, 172),   # Rose pour couronne
            'gold': (255, 215, 0),     # Or pour rayons
            'silver': (192, 192, 192), # Argent pour croix
            'blue': (0, 100, 255)      # Bleu pour lunettes
        }
    
    def create_peach_filter(self, size: int = 80) -> np.ndarray:
        """Crée un filtre pêche programmatique"""
        # Créer une image avec canal alpha
        peach = np.zeros((size, size, 4), dtype=np.uint8)
        
        center = size // 2
        radius = size // 3
        
        # Corps de la pêche (cercle principal)
        cv2.circle(peach, (center, center), radius, (*self.colors['peach'], 200), -1)
        
        # Reflet sur la pêche
        cv2.ellipse(peach, (center - 10, center - 10), (radius//3, radius//2), 
                   -30, 0, 180, (255, 200, 150, 150), -1)
        
        # Feuille verte
        leaf_points = np.array([
            [center, center - radius],
            [center + 15, center - radius - 20],
            [center + 25, center - radius - 15],
            [center + 20, center - radius - 5],
            [center + 5, center - radius + 5]
        ], np.int32)
        cv2.fillPoly(peach, [leaf_points], (34, 139, 34, 180))
        
        # Ligne centrale de la pêche
        cv2.line(peach, (center, center - radius//2), (center, center + radius//2), 
                (255, 140, 100, 100), 3)
        
        return peach
    
    def create_crown_filter(self, size: int = 120) -> np.ndarray:
        """Crée un filtre couronne"""
        crown = np.zeros((size, size//2, 4), dtype=np.uint8)
        
        # Base de la couronne
        base_height = size // 8
        cv2.rectangle(crown, (10, size//2 - base_height), (size - 10, size//2), 
                     (*self.colors['pink'], 200), -1)
        
        # Pointes de la couronne
        points = []
        num_points = 5
        for i in range(num_points):
            x = 10 + (size - 20) * i // (num_points - 1)
            if i % 2 == 0:  # Pointe haute
                y = size//4
            else:  # Pointe basse
                y = size//2 - base_height
            points.append([x, y])
        
        # Ajouter les coins de la base
        points.insert(0, [10, size//2 - base_height])
        points.append([size - 10, size//2 - base_height])
        
        points = np.array(points, np.int32)
        cv2.fillPoly(crown, [points], (*self.colors['pink'], 200))
        
        # Décorations (petits cercles)
        for i in range(0, num_points, 2):
            x = 10 + (size - 20) * i // (num_points - 1)
            cv2.circle(crown, (x, size//4 - 10), 6, (255, 255, 255, 255), -1)
        
        # Cœur au centre
        heart_center = (size//2, size//2 - base_height//2)
        cv2.circle(crown, (heart_center[0] - 8, heart_center[1] - 5), 8, (255, 255, 255, 255), -1)
        cv2.circle(crown, (heart_center[0] + 8, heart_center[1] - 5), 8, (255, 255, 255, 255), -1)
        triangle = np.array([[heart_center[0] - 12, heart_center[1] + 3],
                           [heart_center[0] + 12, heart_center[1] + 3],
                           [heart_center[0], heart_center[1] + 15]], np.int32)
        cv2.fillPoly(crown, [triangle], (255, 255, 255, 255))
        
        return crown
    
    def create_light_ray_filter(self, image_shape: Tuple[int, int], 
                               start_point: Tuple[int, int], 
                               end_point: Tuple[int, int]) -> np.ndarray:
        """Crée un rayon lumineux pour le dab"""
        h, w = image_shape[:2]
        ray_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Calculer l'angle du rayon
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = math.atan2(dy, dx)
        
        # Longueur du rayon (jusqu'au bord de l'image)
        ray_length = min(w, h)
        
        # Points du rayon étendu
        extended_end = (
            int(start_point[0] + ray_length * math.cos(angle)),
            int(start_point[1] + ray_length * math.sin(angle))
        )
        
        # Dessiner le rayon principal
        cv2.line(ray_overlay, start_point, extended_end, (*self.colors['gold'], 180), 8)
        
        # Rayons secondaires (effet de diffraction)
        for offset in [-0.3, -0.15, 0.15, 0.3]:
            offset_angle = angle + offset
            offset_end = (
                int(start_point[0] + ray_length * 0.7 * math.cos(offset_angle)),
                int(start_point[1] + ray_length * 0.7 * math.sin(offset_angle))
            )
            cv2.line(ray_overlay, start_point, offset_end, (*self.colors['gold'], 100), 3)
        
        # Halo autour du point de départ
        cv2.circle(ray_overlay, start_point, 20, (255, 255, 255, 120), -1)
        cv2.circle(ray_overlay, start_point, 15, (*self.colors['gold'], 200), -1)
        
        return ray_overlay
    
    def create_cross_filter(self, center: Tuple[int, int], size: int = 100) -> np.ndarray:
        """Crée une croix lumineuse pour crossarm"""
        cross_overlay = np.zeros((size * 2, size * 2, 4), dtype=np.uint8)
        cross_center = (size, size)
        
        # Croix principale
        # Barre horizontale
        cv2.rectangle(cross_overlay, 
                     (size - size//2, size - 10), 
                     (size + size//2, size + 10), 
                     (*self.colors['silver'], 200), -1)
        
        # Barre verticale
        cv2.rectangle(cross_overlay, 
                     (size - 10, size - size//2), 
                     (size + 10, size + size//2), 
                     (*self.colors['silver'], 200), -1)
        
        # Effet lumineux
        cv2.circle(cross_overlay, cross_center, size//3, (255, 255, 255, 80), -1)
        
        # Rayons diagonaux
        for angle in [45, 135, 225, 315]:
            rad = math.radians(angle)
            end_x = int(cross_center[0] + size//2 * math.cos(rad))
            end_y = int(cross_center[1] + size//2 * math.sin(rad))
            cv2.line(cross_overlay, cross_center, (end_x, end_y), 
                    (255, 255, 255, 150), 3)
        
        return cross_overlay, (center[0] - size, center[1] - size)
    
    def create_sunglasses_filter(self, size: int = 100) -> np.ndarray:
        """Crée des lunettes de soleil pour Jul"""
        glasses = np.zeros((size//2, size, 4), dtype=np.uint8)
        
        # Verres
        lens_radius = size // 6
        left_center = (size//4, size//4)
        right_center = (3*size//4, size//4)
        
        # Verres noirs
        cv2.circle(glasses, left_center, lens_radius, (0, 0, 0, 200), -1)
        cv2.circle(glasses, right_center, lens_radius, (0, 0, 0, 200), -1)
        
        # Reflets sur les verres
        cv2.circle(glasses, (left_center[0] - 8, left_center[1] - 8), 
                  lens_radius//3, (255, 255, 255, 180), -1)
        cv2.circle(glasses, (right_center[0] - 8, right_center[1] - 8), 
                  lens_radius//3, (255, 255, 255, 180), -1)
        
        # Pont entre les verres
        cv2.line(glasses, 
                (left_center[0] + lens_radius, left_center[1]), 
                (right_center[0] - lens_radius, right_center[1]), 
                (0, 0, 0, 200), 4)
        
        # Branches
        cv2.line(glasses, 
                (left_center[0] - lens_radius, left_center[1]), 
                (10, left_center[1]), 
                (0, 0, 0, 200), 3)
        cv2.line(glasses, 
                (right_center[0] + lens_radius, right_center[1]), 
                (size - 10, right_center[1]), 
                (0, 0, 0, 200), 3)
        
        return glasses
    
    def get_hip_position(self, keypoints: np.ndarray) -> Optional[Tuple[int, int]]:
        """Obtient la position des hanches depuis les keypoints YOLO"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Keypoints YOLO : 11 = left_hip, 12 = right_hip
        left_hip_idx = 11 * 3  # x, y, conf
        right_hip_idx = 12 * 3
        
        if len(keypoints) <= right_hip_idx + 2:
            return None
        
        left_hip_conf = keypoints[left_hip_idx + 2]
        right_hip_conf = keypoints[right_hip_idx + 2]
        
        if left_hip_conf > 0.3 and right_hip_conf > 0.3:
            # Position moyenne des hanches
            hip_x = int((keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2)
            hip_y = int((keypoints[left_hip_idx + 1] + keypoints[right_hip_idx + 1]) / 2)
            return (hip_x, hip_y)
        
        return None
    
    def get_head_position(self, keypoints: np.ndarray) -> Optional[Tuple[int, int]]:
        """Obtient la position de la tête depuis les keypoints YOLO"""
        if keypoints is None or len(keypoints) == 0:
            return None
        
        # Keypoints YOLO : 0 = nose
        nose_idx = 0
        if len(keypoints) <= nose_idx * 3 + 2:
            return None
        
        nose_conf = keypoints[nose_idx * 3 + 2]
        if nose_conf > 0.3:
            return (int(keypoints[nose_idx * 3]), int(keypoints[nose_idx * 3 + 1]))
        
        return None
    
    def get_shoulder_positions(self, keypoints: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Obtient les positions des épaules"""
        if keypoints is None or len(keypoints) == 0:
            return None, None
        
        # Keypoints YOLO : 5 = left_shoulder, 6 = right_shoulder
        left_shoulder_idx = 5 * 3
        right_shoulder_idx = 6 * 3
        
        if len(keypoints) <= right_shoulder_idx + 2:
            return None, None
        
        left_shoulder = None
        right_shoulder = None
        
        if keypoints[left_shoulder_idx + 2] > 0.3:
            left_shoulder = (int(keypoints[left_shoulder_idx]), int(keypoints[left_shoulder_idx + 1]))
        
        if keypoints[right_shoulder_idx + 2] > 0.3:
            right_shoulder = (int(keypoints[right_shoulder_idx]), int(keypoints[right_shoulder_idx + 1]))
        
        return left_shoulder, right_shoulder
    
    def get_wrist_positions(self, keypoints: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Obtient les positions des poignets pour le dab"""
        if keypoints is None or len(keypoints) == 0:
            return None, None
        
        # Keypoints YOLO : 9 = left_wrist, 10 = right_wrist
        left_wrist_idx = 9 * 3
        right_wrist_idx = 10 * 3
        
        if len(keypoints) <= right_wrist_idx + 2:
            return None, None
        
        left_wrist = None
        right_wrist = None
        
        if keypoints[left_wrist_idx + 2] > 0.3:
            left_wrist = (int(keypoints[left_wrist_idx]), int(keypoints[left_wrist_idx + 1]))
        
        if keypoints[right_wrist_idx + 2] > 0.3:
            right_wrist = (int(keypoints[right_wrist_idx]), int(keypoints[right_wrist_idx + 1]))
        
        return left_wrist, right_wrist
    
    def apply_overlay(self, image: np.ndarray, overlay: np.ndarray, 
                     position: Tuple[int, int], opacity: float = None) -> np.ndarray:
        """Applique un overlay avec transparence sur l'image"""
        if opacity is None:
            opacity = self.filter_opacity
        
        result = image.copy()
        h, w = image.shape[:2]
        oh, ow = overlay.shape[:2]
        
        x, y = position
        
        # Ajuster la position pour centrer l'overlay
        x -= ow // 2
        y -= oh // 2
        
        # Vérifier les limites
        if x < 0 or y < 0 or x + ow > w or y + oh > h:
            return result
        
        # ROI sur l'image de base
        roi = result[y:y+oh, x:x+ow]
        
        if overlay.shape[2] == 4:  # RGBA
            # Utiliser le canal alpha
            alpha = overlay[:, :, 3] / 255.0 * opacity
            alpha = np.stack([alpha] * 3, axis=2)
            
            overlay_rgb = overlay[:, :, :3]
            blended = roi * (1 - alpha) + overlay_rgb * alpha
            result[y:y+oh, x:x+ow] = blended.astype(np.uint8)
        else:  # RGB
            # Blend simple
            blended = cv2.addWeighted(roi, 1-opacity, overlay, opacity, 0)
            result[y:y+oh, x:x+ow] = blended
        
        return result
    
    def apply_filter_for_gesture(self, image: np.ndarray, gesture: str, 
                                keypoints: np.ndarray = None) -> np.ndarray:
        """Applique le filtre approprié selon le geste détecté"""
        if not self.filters_enabled or gesture == 'neutral':
            return image
        
        result = image.copy()
        
        if gesture == 'twerk':
            hip_pos = self.get_hip_position(keypoints)
            if hip_pos:
                peach_filter = self.create_peach_filter(100)
                result = self.apply_overlay(result, peach_filter, hip_pos)
        
        elif gesture == 'hands_up':
            head_pos = self.get_head_position(keypoints)
            if head_pos:
                crown_filter = self.create_crown_filter(140)
                # Positionner la couronne au-dessus de la tête
                crown_pos = (head_pos[0], head_pos[1] - 60)
                result = self.apply_overlay(result, crown_filter, crown_pos)
        
        elif gesture == 'dab':
            left_shoulder, right_shoulder = self.get_shoulder_positions(keypoints)
            left_wrist, right_wrist = self.get_wrist_positions(keypoints)
            
            # Déterminer quel bras est levé pour le dab
            if left_shoulder and left_wrist and right_shoulder and right_wrist:
                left_up = left_wrist[1] < left_shoulder[1] - 30
                right_up = right_wrist[1] < right_shoulder[1] - 30
                
                if left_up and not right_up:
                    # Bras gauche levé
                    ray_filter = self.create_light_ray_filter(
                        image.shape, left_shoulder, left_wrist
                    )
                    result = self.apply_overlay(result, ray_filter, (0, 0), 0.6)
                elif right_up and not left_up:
                    # Bras droit levé
                    ray_filter = self.create_light_ray_filter(
                        image.shape, right_shoulder, right_wrist
                    )
                    result = self.apply_overlay(result, ray_filter, (0, 0), 0.6)
        
        elif gesture == 'crossarm':
            left_shoulder, right_shoulder = self.get_shoulder_positions(keypoints)
            if left_shoulder and right_shoulder:
                # Position centrale des épaules
                center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                         (left_shoulder[1] + right_shoulder[1]) // 2)
                cross_filter, cross_pos = self.create_cross_filter(center)
                result = self.apply_overlay(result, cross_filter, cross_pos, 0.7)
        
        elif gesture == 'jul':
            head_pos = self.get_head_position(keypoints)
            if head_pos:
                glasses_filter = self.create_sunglasses_filter(120)
                # Positionner les lunettes sur les yeux
                glasses_pos = (head_pos[0], head_pos[1] - 20)
                result = self.apply_overlay(result, glasses_filter, glasses_pos)
        
        return result
    
    def toggle_filters(self):
        """Active/désactive les filtres"""
        self.filters_enabled = not self.filters_enabled
        return self.filters_enabled
    
    def set_opacity(self, opacity: float):
        """Définit l'opacité des filtres (0.0 à 1.0)"""
        self.filter_opacity = max(0.0, min(1.0, opacity))

# Dans api/utils.py ou api/evaluation.py

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class DanceFeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def evaluate_model(model, dataloader, device='cpu', class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Accuracy: {acc:.4f}\n")

    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))

    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return y_true, y_pred
