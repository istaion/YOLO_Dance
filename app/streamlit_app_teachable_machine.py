#app/streamlit_app_teachable_machine.py

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import time

# 🎨 Imports des styles et config
from styles.themes import SOFT_PURPLE
from config.settings import (
    SAVE_DIR, MODEL_DIR, PAGE_CONFIG, 
    DEFAULT_PHOTO_COOLDOWN, DEFAULT_GESTURE_DURATION_THRESHOLD
)
from components.ui_components import (
    render_sidebar_settings, render_metrics_dashboard, 
    render_gestures_status, render_camera_feed_header,
    render_recent_photos, render_debug_info, show_toast_notification
)

# Import de votre adaptateur
from teachable_machine_adapter import TeachableMachinePoseDetector

# 📱 Configuration de l'app
st.set_page_config(**PAGE_CONFIG)

# 🎨 Application du thème
st.markdown(SOFT_PURPLE, unsafe_allow_html=True)

# 🎯 Titre principal
st.title(" YOLO DANCE – On ne capture pas, on éternise l'instant")

# 📁 Créer le dossier d'images
os.makedirs(SAVE_DIR, exist_ok=True)

# 🤖 Initialiser le détecteur
@st.cache_resource
def init_tm_detector():
    try:
        detector = TeachableMachinePoseDetector(MODEL_DIR)
        return detector
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# 💾 Variables de session
if 'photo_count' not in st.session_state:
    st.session_state.photo_count = 0
if 'last_photo_time' not in st.session_state:
    st.session_state.last_photo_time = 0
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = "Neutral"
if 'gesture_start_time' not in st.session_state:
    st.session_state.gesture_start_time = 0

# ⚙️ Rendu de la sidebar
settings = render_sidebar_settings()

# 📊 Tableau de bord
cooldown_remaining = max(0, settings['photo_cooldown'] - (time.time() - st.session_state.last_photo_time))
render_metrics_dashboard(
    st.session_state.photo_count,
    st.session_state.current_gesture,
    cooldown_remaining,
    settings['confidence_threshold']
)

# 🎭 Affichage des gestes disponibles
render_gestures_status(None, settings['priority_gestures'])

def take_photo(frame, gesture, confidence):
    """Prend une photo si les conditions sont remplies"""
    current_time = time.time()
    if current_time - st.session_state.last_photo_time > settings['photo_cooldown']:
        st.session_state.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tm_{gesture}_{confidence:.2f}_{ts}_{st.session_state.photo_count:03d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        st.session_state.last_photo_time = current_time
        show_toast_notification(f"{gesture.upper()} détecté ({confidence:.1%}) ! Photo : {filename}")
        return True
    return False

def process_tm_detection(frame, tm_detector):
    """Traite la détection avec le modèle Teachable Machine"""
    if tm_detector is None:
        return frame, "Neutral", False, False, {}
    
    # Obtenir la prédiction principale
    predicted_gesture, confidence = tm_detector.predict_pose(frame)
    
    # Obtenir toutes les prédictions pour l'affichage
    all_predictions = tm_detector.get_all_predictions(frame)
    
    # Dessiner les keypoints
    frame_with_keypoints = tm_detector.draw_keypoints(frame.copy())
    
    # Vérifier si le geste est détecté avec suffisamment de confiance
    gesture_detected = confidence >= settings['confidence_threshold']
    
    # Prioriser certains gestes
    if gesture_detected and predicted_gesture in settings['priority_gestures']:
        gesture_detected = True
    elif gesture_detected and predicted_gesture not in settings['priority_gestures']:
        # Réduire la sensibilité pour les gestes non prioritaires
        gesture_detected = confidence >= settings['confidence_threshold'] + 0.1
    
    # Gestion de la continuité du geste
    current_time = time.time()
    photo_taken = False
    
    if gesture_detected and predicted_gesture == st.session_state.current_gesture:
        # Geste maintenu
        if current_time - st.session_state.gesture_start_time > DEFAULT_GESTURE_DURATION_THRESHOLD:
            photo_taken = take_photo(frame, predicted_gesture, confidence)
    elif gesture_detected and predicted_gesture != st.session_state.current_gesture:
        # Nouveau geste détecté
        st.session_state.current_gesture = predicted_gesture
        st.session_state.gesture_start_time = current_time
    elif not gesture_detected:
        # Aucun geste détecté avec suffisamment de confiance
        st.session_state.current_gesture = "Neutral"
    
    return frame_with_keypoints, predicted_gesture, gesture_detected, photo_taken, all_predictions

# 🎥 Section caméra
render_camera_feed_header()

tm_detector = init_tm_detector()
if st.button("🎥 Lancer la détection", type="primary"):
    if tm_detector is None:
        st.error("❌ Modèle non chargé. Impossible de démarrer.")
        st.stop()
    
    # Conteneurs pour l'affichage
    video_container = st.empty()
    status_container = st.empty()
    predictions_container = st.empty()
    
    # Bouton d'arrêt
    stop_button = st.button("🛑 Arrêter la caméra", key="stop")
    
    # Initialiser la caméra
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la caméra")
    else:
        st.success("✅ Caméra active - Modèle Teachable Machine prêt")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur lors de la lecture de la caméra.")
                break
            
            # Miroir pour l'UX
            frame = cv2.flip(frame, 1)
            
            # Traitement avec Teachable Machine
            processed_frame, detected_gesture, gesture_detected, photo_taken, all_predictions = process_tm_detection(frame, tm_detector)
            
            # Ajouter des informations visuelles
            if gesture_detected:
                confidence = all_predictions.get(detected_gesture, 0.0) if all_predictions else 0.0
                cv2.putText(processed_frame, f"GESTE: {detected_gesture.upper()}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confiance: {confidence:.1%}", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Effet flash pour les photos
            if photo_taken:
                overlay = np.ones_like(processed_frame) * 255
                processed_frame = cv2.addWeighted(processed_frame, 0.6, overlay, 0.4, 0)
            
            # Affichage vidéo
            video_container.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Affichage du statut
            status_text = f"**Statut:** {detected_gesture} {'✅' if gesture_detected else '❌'}"
            if all_predictions and detected_gesture in all_predictions:
                status_text += f" | Confiance: {all_predictions[detected_gesture]:.1%}"
            status_container.write(status_text)
            
            # Affichage des top prédictions
            if all_predictions:
                sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_predictions[:3]
                
                pred_text = "**Top 3 prédictions:**\n"
                for gesture, conf in top_3:
                    pred_text += f"- {gesture}: {conf:.1%}\n"
                predictions_container.write(pred_text)
            
            # Debug info si activé
            if settings['debug_mode']:
                render_debug_info(all_predictions, detected_gesture, gesture_detected)
            
            time.sleep(0.03)  # Limite FPS
        
        cap.release()
        st.success("🎥 Caméra fermée")

# 📸 Affichage des photos récentes
if st.session_state.photo_count > 0:
    render_recent_photos(SAVE_DIR)