# Lance l'app Streamlit, récupère webcam et envoie à l'API
import streamlit as st
import cv2
import numpy as np
import requests
from datetime import datetime
import os
import time
import sys
import asyncio

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())


# Ajouter le dossier parent pour importer vos modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.pose_detection_model import PoseDetector

#Dossier pour sauvegarder les screenshots
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

# #Adresse de l'API backend

# API_URL = "http://localhost:8000/detect/"

#titre de l'app
st.set_page_config(page_title="YOLO_Danse - Détection en Live")
st.title("🕺 YOLO_Dance – Capture automatique de tes meilleurs moves")

# Initialiser le détecteur de poses (une seule fois)
@st.cache_resource
def init_pose_detector():
    return PoseDetector()

# Variables de session pour la gestion des photos
if 'photo_count' not in st.session_state:
    st.session_state.photo_count = 0
if 'last_photo_time' not in st.session_state:
    st.session_state.last_photo_time = 0
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = "None"
if 'gesture_start_time' not in st.session_state:
    st.session_state.gesture_start_time = 0
# Configuration
PHOTO_COOLDOWN = 3.0  # secondes
GESTURE_DURATION_THRESHOLD = 1.0  # maintenir le geste 1 seconde

def take_photo(frame, gesture):
    """Prend une photo si les conditions sont remplies"""
    current_time = time.time()
    if current_time - st.session_state.last_photo_time > PHOTO_COOLDOWN:
        st.session_state.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"karaoke_{gesture}_{ts}_{st.session_state.photo_count:03d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        st.session_state.last_photo_time = current_time
        st.toast(f"📸 {gesture.upper()} détecté ! Photo : {filename}")
        return True
    return False

def process_gesture_detection(frame, pose_detector):
    """Traite la détection de gestes (votre logique)"""
    # Détection des poses
    results = pose_detector.detect_poses(frame)
    
    # Dessiner les keypoints sur la frame
    frame_with_keypoints = pose_detector.draw_keypoints(frame.copy(), results)
    
    # Extraire les données des keypoints
    keypoints_data = pose_detector.get_keypoints_data(results)
    
    gesture_detected = False
    detected_gesture = "None"
    
    if keypoints_data:
        kp_data = keypoints_data[0]  # Premier personne
        keypoints = kp_data['keypoints']
        conf = kp_data['confidence']
        
        # Tester différents gestes dans l'ordre de priorité (votre logique)
        if pose_detector.detect_gesture(keypoints, conf, "jump"):
            detected_gesture = "Jump"
            gesture_detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "dab"):
            detected_gesture = "Dab"
            gesture_detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "hands_up"):
            detected_gesture = "Hands_Up"
            gesture_detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "twerk"):
            detected_gesture = "Twerk"
            gesture_detected = True
    
    # Gestion de la continuité du geste (votre logique)
    current_time = time.time()
    photo_taken = False
    
    if gesture_detected and detected_gesture == st.session_state.current_gesture:
        # Geste maintenu
        if current_time - st.session_state.gesture_start_time > GESTURE_DURATION_THRESHOLD:
            photo_taken = take_photo(frame, detected_gesture)
            
    elif gesture_detected and detected_gesture != st.session_state.current_gesture:
        # Nouveau geste détecté
        st.session_state.current_gesture = detected_gesture
        st.session_state.gesture_start_time = current_time
    elif not gesture_detected:
        # Aucun geste
        st.session_state.current_gesture = "None"
    
    return frame_with_keypoints, detected_gesture, gesture_detected, photo_taken

# Interface utilisateur
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Photos prises", st.session_state.photo_count)
with col2:
    st.metric("Geste actuel", st.session_state.current_gesture)
with col3:
    cooldown_remaining = max(0, PHOTO_COOLDOWN - (time.time() - st.session_state.last_photo_time))
    st.metric("Cooldown", f"{cooldown_remaining:.1f}s")

st.write("**Gestes détectés :** Mains en l'air, Dab, Jump, Twerk")


# button pour démarrer la caméra
if st.button("🎥 Lancer la caméra"):
    # Initialiser le détecteur
    pose_detector = init_pose_detector()
    
    stframe = st.empty()
    status_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la caméra")
    else:
        st.success("✅ Caméra active")
        
        # Bouton d'arrêt
        stop_button = st.button("🛑 Arrêter la caméra")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur lors de la lecture de la caméra.")
                break
            
            # Miroir pour UX
            frame = cv2.flip(frame, 1)
            
            # # Envoi à l'API (commenté comme dans l'original)
            # _, img_encoded = cv2.imencode('.jpg', frame)
            # response = requests.post(
            #     API_URL,
            #     files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            # )
            # if response.ok and response.json().get("detected"):
            #     # 📸 Mouvement détecté → capture d'écran
            #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     filename = f"move_{ts}.jpg"
            #     filepath = os.path.join(SAVE_DIR, filename)
            #     cv2.imwrite(filepath, frame)
            #     st.toast(f"📸 Mouvement détecté ! Photo enregistrée : {filename}")
            
            # INTEGRATION DE VOS FONCTIONS ICI :
            processed_frame, detected_gesture, gesture_detected, photo_taken = process_gesture_detection(frame, pose_detector)
            
            # Ajouter des informations visuelles sur la frame
            if gesture_detected:
                cv2.putText(processed_frame, f"GESTE: {detected_gesture.upper()}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if photo_taken:
                # Effet flash
                overlay = np.ones_like(processed_frame) * 255
                processed_frame = cv2.addWeighted(processed_frame, 0.6, overlay, 0.4, 0)
            
            # 📺 Affichage live
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Mise à jour du statut
            status_placeholder.write(f"**Statut :** {detected_gesture} {'✅' if gesture_detected else '❌'}")
            
            # Petite pause pour éviter la surcharge
            time.sleep(0.03)
        
        cap.release()
        st.success("🎥 Caméra fermée")

# Affichage des dernières photos
if st.session_state.photo_count > 0:
    st.subheader("📸 Dernières photos prises")
    
    # Lister les fichiers dans le dossier images
    if os.path.exists(SAVE_DIR):
        image_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')]
        image_files.sort(reverse=True)  # Plus récents en premier
        
        # Afficher les 3 dernières photos
        cols = st.columns(3)
        for i, img_file in enumerate(image_files[:3]):
            with cols[i]:
                img_path = os.path.join(SAVE_DIR, img_file)
                st.image(img_path, caption=img_file, use_column_width=True)