import cv2
import time
from pose_detection_model import PoseDetector

class WebcamGestureDetector:
    def __init__(self):
        self.pose_detector = PoseDetector()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variables pour la prise de photo
        self.photo_cooldown = 3  # Secondes entre les photos
        self.last_photo_time = 0
        self.photo_count = 0
        
        # Variables spéciales pour le saut
        self.jump_peak_frame = None  # Stocke la frame du pic de saut
        
        # États des gestes
        self.current_gesture = "None"
        self.gesture_start_time = 0
        self.gesture_duration_threshold = 1.0  # Maintenir le geste 1 seconde
        
    def take_photo(self, frame, gesture_type="gesture"):
        """
        Prend une photo et la sauvegarde
        """
        current_time = time.time()
        if current_time - self.last_photo_time > self.photo_cooldown:
            self.photo_count += 1
            
            # Créer le dossier images s'il n'existe pas
            import os
            os.makedirs("../images", exist_ok=True)
            
            filename = f"../images/karaoke_{gesture_type}_{self.photo_count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Photo prise : {filename}")
            self.last_photo_time = current_time
            return True
        return False
    
    def process_frame(self, frame):
        """
        Traite une frame pour détecter les gestes
        """
        # Détection des poses
        results = self.pose_detector.detect_poses(frame)
        
        # Dessiner les keypoints
        frame = self.pose_detector.draw_keypoints(frame, results)
        
        # Extraire les données des keypoints
        keypoints_data = self.pose_detector.get_keypoints_data(results)
        
        gesture_detected = False
        detected_gesture = "None"
        photo_taken = False
        
        if keypoints_data:
            kp_data = keypoints_data[0]  # Premier personne
            keypoints = kp_data['keypoints']
            conf = kp_data['confidence']
            
            # Ajouter position à l'historique pour détection de saut
            self.pose_detector.add_position_to_history(keypoints, conf, frame.copy())
            
            # Tester le saut en premier (priorité car temporel)
            jump_result = self.pose_detector.detect_gesture(keypoints, conf, "jump")
            if isinstance(jump_result, dict) and jump_result['detected']:
                detected_gesture = "Jump"
                gesture_detected = True
                
                # Capturer au pic du saut
                if jump_result['should_capture'] and jump_result['peak_frame'] is not None:
                    photo_taken = self.take_photo(jump_result['peak_frame'], "jump_peak")
                    if photo_taken:
                        print(f"🦘 Saut détecté - Photo au pic! Phase: {jump_result['phase']}")
                        # Ajouter effet visuel
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255), -1)
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Afficher la phase du saut
                cv2.putText(frame, f"JUMP: {jump_result['phase']}", (50, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Tester les autres gestes seulement si pas de saut
            elif self.pose_detector.detect_gesture(keypoints, conf, "dab"):
                detected_gesture = "Dab"
                gesture_detected = True
            elif self.pose_detector.detect_gesture(keypoints, conf, "hands_up"):
                detected_gesture = "Hands Up"
                gesture_detected = True
            elif self.pose_detector.detect_gesture(keypoints, conf, "twerk"):
                detected_gesture = "Twerk"
                gesture_detected = True
        
        # Gestion de la continuité des gestes (sauf saut qui a sa propre logique)
        current_time = time.time()
        
        if detected_gesture != "Jump":  # Les autres gestes utilisent l'ancienne logique
            if gesture_detected and detected_gesture == self.current_gesture:
                # Geste maintenu
                if current_time - self.gesture_start_time > self.gesture_duration_threshold:
                    # Prendre une photo
                    if self.take_photo(frame, detected_gesture.lower().replace(" ", "_")):
                        # Ajouter un effet visuel pour la photo
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                        photo_taken = True
                    
            elif gesture_detected and detected_gesture != self.current_gesture:
                # Nouveau geste détecté
                self.current_gesture = detected_gesture
                self.gesture_start_time = current_time
            elif not gesture_detected:
                # Aucun geste
                self.current_gesture = "None"
        
        return frame, detected_gesture, gesture_detected
    
    def add_ui_elements(self, frame, detected_gesture, gesture_detected):
        """
        Ajoute l'interface utilisateur sur la frame
        """
        # Fond pour le texte
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Informations
        cv2.putText(frame, f"Geste detecte: {detected_gesture}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if gesture_detected else (255, 255, 255), 2)
        
        cv2.putText(frame, f"Photos prises: {self.photo_count}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Gestes: Mains en l'air, Dab, Jump, Twerk", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, "Appuyez 'q' pour quitter", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Indicateur de statut
        status_color = (0, 255, 0) if gesture_detected else (0, 0, 255)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, status_color, -1)
        
        return frame
    
    def run(self):
        """
        Lance la détection en temps réel
        """
        print("🎤 YOLO Dance - Détection de gestes pour karaoke")
        print("Placez-vous devant la caméra et faites des gestes !")
        print("Gestes reconnus : Mains en l'air, Dab, Jump, Twerk")
        print("Appuyez 'q' pour quitter\n")
        
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erreur : Impossible de lire la webcam")
                break
            
            # Miroir de l'image pour une meilleure UX
            frame = cv2.flip(frame, 1)
            
            # Traitement principal
            frame, detected_gesture, gesture_detected = self.process_frame(frame)
            
            # Interface utilisateur
            frame = self.add_ui_elements(frame, detected_gesture, gesture_detected)
            
            # Calcul FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                print(f"FPS: {fps:.1f} | Geste: {detected_gesture}")
                start_time = time.time()
            
            # Affichage
            cv2.imshow('YOLO Dance - Karaoke Gesture Detection', frame)
            
            # Contrôles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):  # Sauvegarde manuelle
                self.take_photo(frame)
        
        # Nettoyage
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎉 Session terminée ! {self.photo_count} photos prises.")

if __name__ == "__main__":
    detector = WebcamGestureDetector()
    detector.run()