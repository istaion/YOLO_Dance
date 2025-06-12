"""
MediaPipe Holistic Keypoint Extractor
Extrait les keypoints de pose, mains et visage pour chaque personne détectée
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import os
from dataclasses import dataclass

@dataclass
class PersonKeypoints:
    """Structure pour stocker les keypoints d'une personne"""
    pose_landmarks: Optional[np.ndarray] = None
    left_hand_landmarks: Optional[np.ndarray] = None  
    right_hand_landmarks: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None
    face_center: Optional[Tuple[float, float]] = None  # (x, y) normalisées
    confidence_score: float = 0.0
    person_id: int = 0

class MediaPipeKeypointExtractor:
    """
    Extracteur de keypoints utilisant MediaPipe Holistic
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.1,
                 min_tracking_confidence: float = 0.1,
                 model_complexity: int = 1):
        """
        Initialise l'extracteur MediaPipe
        
        Args:
            min_detection_confidence: Seuil de confiance pour la détection (facilement modifiable)
            min_tracking_confidence: Seuil de confiance pour le tracking
            model_complexity: Complexité du modèle (0=lite, 1=full, 2=heavy)
        """
        
        # Configuration facilement modifiable
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # Initialisation MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity
        )
        
        # Points clés pour calculer le centre du visage (plus stable que tous les points)
        self.face_center_landmarks = [1, 2, 5, 6, 10, 151, 9, 10, 151, 337, 299, 333, 298, 301]
        
    def update_confidence_threshold(self, new_threshold: float):
        """Met à jour le seuil de confiance (facilement modifiable pendant l'exécution)"""
        self.min_detection_confidence = new_threshold
        self.holistic.close()
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity
        )
    
    def _landmarks_to_array(self, landmarks) -> Optional[np.ndarray]:
        """Convertit les landmarks MediaPipe en array numpy"""
        if landmarks is None:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    def _calculate_face_center(self, face_landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Calcule le centre du visage à partir des landmarks clés
        Utilise des points stables pour une détection de saut robuste
        """
        if face_landmarks is None or len(face_landmarks) == 0:
            return (0.5, 0.5)  # Centre par défaut
        
        # Utiliser des points clés du visage (nez, front, joues)
        key_points = face_landmarks[[1, 2, 5, 6, 10, 151, 9, 337, 299, 333]]
        
        # Moyenne des coordonnées x et y (déjà normalisées par MediaPipe)
        center_x = np.mean(key_points[:, 0])
        center_y = np.mean(key_points[:, 1]) 
        
        return (float(center_x), float(center_y))
    
    def _calculate_confidence_score(self, pose_landmarks, left_hand, right_hand, face_landmarks) -> float:
        """Calcule un score de confiance global pour la détection"""
        scores = []
        
        # Score basé sur la présence et qualité des landmarks
        if pose_landmarks is not None:
            # Vérifier que les points clés du corps sont détectés
            visible_pose_points = np.sum(pose_landmarks[:, 2] > 0.1)  # z-coordinate comme proxy de visibilité
            scores.append(visible_pose_points / len(pose_landmarks))
        
        if left_hand is not None:
            scores.append(0.8)  # Bonne détection de main
            
        if right_hand is not None:
            scores.append(0.8)  # Bonne détection de main
            
        if face_landmarks is not None:
            scores.append(0.9)  # Très bonne détection de visage
        
        return np.mean(scores) if scores else 0.0
    
    def extract_keypoints_from_image(self, image: np.ndarray) -> List[PersonKeypoints]:
        """
        Extrait les keypoints d'une image
        
        Args:
            image: Image BGR (format OpenCV)
            
        Returns:
            Liste des PersonKeypoints détectées
        """
        
        # Conversion BGR to RGB pour MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # Traitement MediaPipe
        results = self.holistic.process(rgb_image)
        
        persons = []
        
        # MediaPipe Holistic ne gère qu'une personne à la fois
        # Pour multi-personnes, il faudrait utiliser MediaPipe Pose + découpage d'image
        if results.pose_landmarks:
            
            # Conversion des landmarks
            pose_landmarks = self._landmarks_to_array(results.pose_landmarks)
            left_hand_landmarks = self._landmarks_to_array(results.left_hand_landmarks)
            right_hand_landmarks = self._landmarks_to_array(results.right_hand_landmarks)
            face_landmarks = self._landmarks_to_array(results.face_landmarks)
            
            # Calcul du centre du visage
            face_center = self._calculate_face_center(face_landmarks) if face_landmarks is not None else (0.5, 0.5)
            
            # Calcul du score de confiance
            confidence = self._calculate_confidence_score(
                pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks
            )
            
            # Filtrage par seuil de confiance
            if confidence >= self.min_detection_confidence:
                person = PersonKeypoints(
                    pose_landmarks=pose_landmarks,
                    left_hand_landmarks=left_hand_landmarks,
                    right_hand_landmarks=right_hand_landmarks,
                    face_landmarks=face_landmarks,
                    face_center=face_center,
                    confidence_score=confidence,
                    person_id=0  # Une seule personne pour l'instant
                )
                persons.append(person)
        
        return persons
    
    def process_dataset_images(self, dataset_path: str, output_path: str, visualize: bool = False):
        """
        Traite toutes les images du dataset et sauvegarde les keypoints
        
        Args:
            dataset_path: Chemin vers data/dataset/images
            output_path: Chemin de sortie pour les keypoints
            visualize: Sauvegarder des images avec keypoints dessinés
        """
        
        classes = ['hands_up', 'dab', 'twerk', 'mic_drop', 'middle_finger', 
                  'peace', 'heart', 'jule', 'neutral']
        
        os.makedirs(output_path, exist_ok=True)
        if visualize:
            os.makedirs(os.path.join(output_path, 'visualizations'), exist_ok=True)
        
        dataset_keypoints = {}
        processed_count = 0
        failed_count = 0
        
        for class_name in classes:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️  Classe {class_name} introuvable dans {class_path}")
                continue
                
            dataset_keypoints[class_name] = []
            print(f"\n🔄 Traitement de la classe: {class_name}")
            
            # Traiter toutes les images de la classe
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Charger l'image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"❌ Impossible de charger {img_path}")
                        failed_count += 1
                        continue
                    
                    # Extraire les keypoints
                    persons = self.extract_keypoints_from_image(image)
                    
                    if persons:
                        # Sauvegarder les keypoints (convertir en format sérialisable)
                        keypoint_data = {
                            'image_path': img_path,
                            'class': class_name,
                            'persons': []
                        }
                        
                        for person in persons:
                            person_data = {
                                'pose_landmarks': person.pose_landmarks.tolist() if person.pose_landmarks is not None else None,
                                'left_hand_landmarks': person.left_hand_landmarks.tolist() if person.left_hand_landmarks is not None else None,
                                'right_hand_landmarks': person.right_hand_landmarks.tolist() if person.right_hand_landmarks is not None else None,
                                'face_center': person.face_center,
                                'confidence_score': person.confidence_score,
                                'person_id': person.person_id
                            }
                            keypoint_data['persons'].append(person_data)
                        
                        dataset_keypoints[class_name].append(keypoint_data)
                        processed_count += 1
                        
                        # Visualisation optionnelle
                        if visualize and len(dataset_keypoints[class_name]) <= 10:  # Les 10 premières pour déboguer
                            self._save_visualization(image, persons, 
                                                   os.path.join(output_path, 'visualizations', 
                                                              f"{class_name}_{len(dataset_keypoints[class_name])}.jpg"))
                    else:
                        print(f"⚠️  Aucune pose détectée dans {img_file}")
                        failed_count += 1
                        
                except Exception as e:
                    print(f"❌ Erreur lors du traitement de {img_path}: {e}")
                    failed_count += 1
            
            print(f"✅ Classe {class_name}: {len(dataset_keypoints[class_name])} images traitées")
        
        # Sauvegarder le dataset complet
        output_file = os.path.join(output_path, 'dataset_keypoints.json')
        with open(output_file, 'w') as f:
            json.dump(dataset_keypoints, f, indent=2)
        
        print(f"\n📊 Résumé:")
        print(f"✅ Images traitées avec succès: {processed_count}")
        print(f"❌ Images échouées: {failed_count}")
        print(f"💾 Keypoints sauvegardés dans: {output_file}")
        
        return dataset_keypoints
    
    def _save_visualization(self, image: np.ndarray, persons: List[PersonKeypoints], output_path: str):
        """Sauvegarde une image avec les keypoints dessinés"""
        
        # Copier l'image pour éviter de modifier l'originale
        vis_image = image.copy()
        
        for person in persons:
            # Dessiner les keypoints de pose directement
            if person.pose_landmarks is not None:
                # Dessiner les points de pose
                for i, (x, y, z) in enumerate(person.pose_landmarks):
                    if z > 0.1:  # Seulement si le point est visible
                        px = int(x * image.shape[1])
                        py = int(y * image.shape[0])
                        cv2.circle(vis_image, (px, py), 3, (0, 255, 0), -1)
            
            # Dessiner les keypoints des mains
            if person.left_hand_landmarks is not None:
                for x, y, z in person.left_hand_landmarks:
                    px = int(x * image.shape[1])
                    py = int(y * image.shape[0])
                    cv2.circle(vis_image, (px, py), 2, (255, 0, 0), -1)  # Bleu pour main gauche
            
            if person.right_hand_landmarks is not None:
                for x, y, z in person.right_hand_landmarks:
                    px = int(x * image.shape[1])
                    py = int(y * image.shape[0])
                    cv2.circle(vis_image, (px, py), 2, (0, 0, 255), -1)  # Rouge pour main droite
            
            # Dessiner le centre du visage
            if person.face_center:
                center_px = (int(person.face_center[0] * image.shape[1]), 
                           int(person.face_center[1] * image.shape[0]))
                cv2.circle(vis_image, center_px, 10, (255, 255, 0), -1)  # Jaune pour centre visage
                cv2.putText(vis_image, f"Face Center", (center_px[0]-50, center_px[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher le score de confiance
            cv2.putText(vis_image, f"Confidence: {person.confidence_score:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Sauvegarder
        cv2.imwrite(output_path, vis_image)
    
    def __del__(self):
        """Nettoyage"""
        if hasattr(self, 'holistic'):
            self.holistic.close()


# Exemple d'utilisation
if __name__ == "__main__":
    
    # Configuration
    DATASET_PATH = "../../data/dataset/images"  # Ajuster selon votre structure
    OUTPUT_PATH = "../../data/keypoints"
    
    # Initialisation de l'extracteur
    extractor = MediaPipeKeypointExtractor(
        min_detection_confidence=0, 
        min_tracking_confidence=0,
        model_complexity=1
    )
    
    print("🚀 Début de l'extraction des keypoints...")
    
    # Traitement du dataset complet
    keypoints_data = extractor.process_dataset_images(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        visualize=True  # Créer quelques visualisations
    )
    
    print("✅ Extraction terminée!")
    
    # Test sur une seule image pour debug
    test_image_path = "../../data/dataset/images/neutral/test.jpg"  # Ajuster
    if os.path.exists(test_image_path):
        print(f"\n🧪 Test sur une image: {test_image_path}")
        test_image = cv2.imread(test_image_path)
        persons = extractor.extract_keypoints_from_image(test_image)
        
        for i, person in enumerate(persons):
            print(f"Personne {i+1}:")
            print(f"  - Score de confiance: {person.confidence_score:.3f}")
            print(f"  - Centre du visage: {person.face_center}")
            print(f"  - Pose détectée: {'✅' if person.pose_landmarks is not None else '❌'}")
            print(f"  - Main gauche: {'✅' if person.left_hand_landmarks is not None else '❌'}")
            print(f"  - Main droite: {'✅' if person.right_hand_landmarks is not None else '❌'}")