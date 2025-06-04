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

        # Historique pour détection de saut
        self.position_history = []  # Stocke les positions des dernières frames
        self.max_history_length = 20  # Nombre de frames à conserver
        self.jump_state = {
            'in_jump': False,
            'peak_frame': None,
            'peak_height': 0,
            'start_height': 0,
            'frames_since_peak': 0
        }
    
    def detect_poses(self, frame):
        """
        Détecte les poses dans l'image
        """
        results = self.model(frame, verbose=False)
        return results
    
    def add_position_to_history(self, keypoints, conf, frame=None):
        """Ajoute la position actuelle à l'historique pour analyse temporelle"""
        if len(keypoints) == 0:
            return
            
        kp = keypoints[0]  # Premier personne détectée
        c = conf[0]
        
        # Calculer la position moyenne du corps (bassin + épaules)
        left_shoulder = 5
        right_shoulder = 6
        left_hip = 11
        right_hip = 12
        
        if (c[left_shoulder] > 0.4 and c[right_shoulder] > 0.4 and 
            c[left_hip] > 0.4 and c[right_hip] > 0.4):
            
            # Position moyenne du bassin
            hip_center_y = (kp[left_hip][1] + kp[right_hip][1]) / 2
            
            # Position moyenne des épaules
            shoulder_center_y = (kp[left_shoulder][1] + kp[right_shoulder][1]) / 2
            
            # Position moyenne du corps (pour le saut)
            body_center_y = (hip_center_y + shoulder_center_y) / 2
            
            # Ajouter à l'historique
            position_data = {
                'body_center_y': body_center_y,
                'hip_center_y': hip_center_y,
                'shoulder_center_y': shoulder_center_y,
                'frame': frame,
                'timestamp': cv2.getTickCount()
            }
            
            self.position_history.append(position_data)
            
            # Limiter la taille de l'historique
            if len(self.position_history) > self.max_history_length:
                self.position_history.pop(0)

    def detect_jump_motion(self):
        """
        Détecte un saut basé sur l'analyse temporelle du mouvement
        Retourne: (is_jumping, peak_frame, jump_phase)
        """
        if len(self.position_history) < 10:  # Besoin d'au moins 10 frames
            return False, None, "insufficient_data"
        
        # Extraire les positions Y du corps
        body_positions = [pos['body_center_y'] for pos in self.position_history]
        
        # Calculer les variations de position
        position_changes = np.diff(body_positions)
        
        # Détecter élévation brusque (mouvement vers le haut = valeurs négatives)
        recent_changes = position_changes[-5:]  # 5 dernières frames
        rapid_elevation = np.mean(recent_changes) < -8  # Seuil d'élévation rapide
        
        current_height = body_positions[-1]
        
        # Machine à états pour le saut
        if not self.jump_state['in_jump']:
            # Recherche début de saut (élévation rapide)
            if rapid_elevation:
                self.jump_state['in_jump'] = True
                self.jump_state['start_height'] = current_height
                self.jump_state['peak_height'] = current_height
                self.jump_state['peak_frame'] = self.position_history[-1]['frame']
                self.jump_state['frames_since_peak'] = 0
                return True, None, "jump_start"
        
        else:
            # En cours de saut - suivre le pic
            self.jump_state['frames_since_peak'] += 1
            
            # Nouveau pic détecté
            if current_height < self.jump_state['peak_height']:  # Plus haut (Y plus petit)
                self.jump_state['peak_height'] = current_height
                self.jump_state['peak_frame'] = self.position_history[-1]['frame']
                self.jump_state['frames_since_peak'] = 0
                return True, self.jump_state['peak_frame'], "jump_peak"
            
            # Détection de la redescente
            elif (self.jump_state['frames_since_peak'] > 3 and 
                  current_height > self.jump_state['start_height'] - 20):  # Retour proche position initiale
                
                # Fin du saut
                peak_frame = self.jump_state['peak_frame']
                self.jump_state['in_jump'] = False
                self.jump_state['frames_since_peak'] = 0
                
                return True, peak_frame, "jump_complete"
            
            # Toujours en saut
            elif self.jump_state['frames_since_peak'] < 15:  # Max 15 frames pour un saut
                return True, None, "jump_in_progress"
            
            else:
                # Timeout - fin forcée du saut
                self.jump_state['in_jump'] = False
                return False, None, "jump_timeout"
        
        return False, None, "no_jump"

    def draw_keypoints(self, frame, results):
        """
        Dessine les keypoints et le squelette sur l'image
        """
        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
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
                        if (kp1 < len(person_conf) and kp2 < len(person_conf) and
                            person_conf[kp1] > 0.5 and person_conf[kp2] > 0.5):
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
            # Nouvelle détection du saut basée sur l'analyse temporelle
            # Cette fonction doit être appelée après add_position_to_history()
            is_jumping, peak_frame, jump_phase = self.detect_jump_motion()
            
            # Retourner les informations du saut
            if is_jumping:
                return {
                    'detected': True,
                    'phase': jump_phase,
                    'peak_frame': peak_frame,
                    'should_capture': jump_phase in ['jump_peak', 'jump_complete']
                }
            else:
                return {
                    'detected': False,
                    'phase': jump_phase,
                    'peak_frame': None,
                    'should_capture': False
                }
        
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
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                keypoints = result.keypoints.xy.cpu().numpy()
                conf = result.keypoints.conf.cpu().numpy()
                keypoints_data.append({
                    'keypoints': keypoints,
                    'confidence': conf
                })
        
        return keypoints_data