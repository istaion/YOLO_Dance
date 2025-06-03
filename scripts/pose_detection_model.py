import cv2
import numpy as np
from ultralytics import YOLO
import math

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Initialise le détecteur de poses YOLO
        """
        self.model = YOLO(model_path)
        
        # Points clés COCO (17 keypoints)
        self.keypoints_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Connexions pour dessiner le squelette
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
    
    def detect_poses(self, frame):
        """
        Détecte les poses dans l'image
        """
        results = self.model(frame, verbose=False)
        return results
    
    def draw_keypoints(self, frame, results):
        """
        Dessine les keypoints et le squelette sur l'image
        """
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                conf = result.keypoints.conf.cpu().numpy()
                
                for person_kp, person_conf in zip(keypoints, conf):
                    # Dessiner les keypoints
                    for i, (x, y) in enumerate(person_kp):
                        if person_conf[i] > 0.5:  # Seuil de confiance
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.putText(frame, str(i), (int(x), int(y-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    # Dessiner le squelette
                    for connection in self.skeleton:
                        kp1, kp2 = connection[0]-1, connection[1]-1  # COCO est 1-indexé
                        if (person_conf[kp1] > 0.5 and person_conf[kp2] > 0.5):
                            x1, y1 = int(person_kp[kp1][0]), int(person_kp[kp1][1])
                            x2, y2 = int(person_kp[kp2][0]), int(person_kp[kp2][1])
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame
    
    def detect_gesture(self, keypoints, conf, gesture_type="hands_up"):
        """
        Détecte des gestes spécifiques basés sur les keypoints
        """
        if len(keypoints) == 0:
            return False
            
        # Indices des points clés importants (COCO format)
        nose = 0
        left_wrist = 9
        right_wrist = 10
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12
        left_knee = 13
        right_knee = 14
        left_ankle = 15
        right_ankle = 16
        
        kp = keypoints[0]  # Premier personne détectée
        c = conf[0]
        
        if gesture_type == "hands_up":
            # Vérifier si les mains sont levées (mais pas en position dab)
            if (c[left_wrist] > 0.5 and c[right_wrist] > 0.5 and 
                c[left_shoulder] > 0.5 and c[right_shoulder] > 0.5):
                
                left_hand_up = kp[left_wrist][1] < kp[left_shoulder][1] - 20
                right_hand_up = kp[right_wrist][1] < kp[right_shoulder][1] - 20
                
                both_hands_up = left_hand_up and right_hand_up
                
                # Vérifier que ce n'est PAS un dab (les deux mains doivent être à hauteur similaire)
                if both_hands_up and c[nose] > 0.5:
                    hands_height_diff = abs(kp[left_wrist][1] - kp[right_wrist][1])
                    hands_at_similar_height = hands_height_diff < 60  # Tolérance pour "mains en l'air"
                    
                    # Les mains ne doivent pas être trop proches de la tête (éviter confusion avec dab)
                    nose_pos = kp[nose]
                    left_hand_distance = np.sqrt((kp[left_wrist][0] - nose_pos[0])**2 + 
                                                (kp[left_wrist][1] - nose_pos[1])**2)
                    right_hand_distance = np.sqrt((kp[right_wrist][0] - nose_pos[0])**2 + 
                                                 (kp[right_wrist][1] - nose_pos[1])**2)
                    
                    hands_not_near_head = left_hand_distance > 100 and right_hand_distance > 100
                    
                    return hands_at_similar_height and hands_not_near_head
                
                return both_hands_up and not c[nose] > 0.5  # Si pas de nez détecté, accepter quand même
        
        
        elif gesture_type == "dab":
            # Détection simplifiée du dab
            if (c[left_wrist] > 0.5 and c[right_wrist] > 0.5 and 
                c[left_shoulder] > 0.5 and c[right_shoulder] > 0.5):
                
                # Un bras vers le haut, l'autre vers le bas
                left_up = kp[left_wrist][1] < kp[left_shoulder][1] - 30
                right_down = kp[right_wrist][1] > kp[right_shoulder][1] + 30
                left_down = kp[left_wrist][1] > kp[left_shoulder][1] + 30
                right_up = kp[right_wrist][1] < kp[right_shoulder][1] - 30
                
                return (left_up and right_down) or (right_up and left_down)
        
        elif gesture_type == "jump":
            # Détection du saut basée sur la position des genoux et chevilles
            if (c[left_knee] > 0.4 and c[right_knee] > 0.4 and 
                c[left_hip] > 0.4 and c[right_hip] > 0.4):
                
                # Calculer la distance moyenne genoux-hanches
                left_knee_hip_dist = kp[left_hip][1] - kp[left_knee][1]
                right_knee_hip_dist = kp[right_hip][1] - kp[right_knee][1]
                avg_knee_hip_dist = (left_knee_hip_dist + right_knee_hip_dist) / 2
                
                # Saut détecté si genoux très proches des hanches (jambes repliées)
                jump_threshold = 40  # Seuil en pixels
                knees_raised = avg_knee_hip_dist < jump_threshold
                
                # Vérification supplémentaire : les chevilles sont aussi relevées si visibles
                ankles_raised = True
                if c[left_ankle] > 0.4 and c[right_ankle] > 0.4:
                    left_ankle_ground_dist = kp[left_ankle][1]
                    right_ankle_ground_dist = kp[right_ankle][1]
                    
                    # Comparer avec la position "normale" des chevilles (estimation)
                    estimated_ground = max(kp[left_hip][1], kp[right_hip][1]) + 200
                    ankles_raised = (left_ankle_ground_dist < estimated_ground - 50 and 
                                   right_ankle_ground_dist < estimated_ground - 50)
                
                return knees_raised and ankles_raised
        
        elif gesture_type == "twerk":
            # Détection du twerk basée sur position accroupie et hanches proéminentes
            if (c[left_knee] > 0.4 and c[right_knee] > 0.4 and 
                c[left_hip] > 0.4 and c[right_hip] > 0.4 and
                c[left_shoulder] > 0.4 and c[right_shoulder] > 0.4):
                
                # Position accroupie : genoux en dessous des hanches
                left_knee_below_hip = kp[left_knee][1] > kp[left_hip][1] + 30
                right_knee_below_hip = kp[right_knee][1] > kp[right_hip][1] + 30
                is_crouched = left_knee_below_hip and right_knee_below_hip
                
                # Corps penché : épaules en avant par rapport aux hanches
                shoulder_center_x = (kp[left_shoulder][0] + kp[right_shoulder][0]) / 2
                hip_center_x = (kp[left_hip][0] + kp[right_hip][0]) / 2
                body_leaning = abs(shoulder_center_x - hip_center_x) > 20
                
                # Hanches relativement hautes par rapport aux genoux
                hip_center_y = (kp[left_hip][1] + kp[right_hip][1]) / 2
                knee_center_y = (kp[left_knee][1] + kp[right_knee][1]) / 2
                hips_prominent = (knee_center_y - hip_center_y) > 60
                
                return is_crouched and (body_leaning or hips_prominent)
        
        return False
    
    def get_keypoints_data(self, results):
        """
        Extrait les données des keypoints pour traitement ultérieur
        """
        keypoints_data = []
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                conf = result.keypoints.conf.cpu().numpy()
                keypoints_data.append({
                    'keypoints': keypoints,
                    'confidence': conf
                })
        
        return keypoints_data