# app/streamlit_app_unified.py

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
    DEFAULT_PHOTO_COOLDOWN, DEFAULT_GESTURE_DURATION_THRESHOLD,
    API_CONFIG, MODEL_CONFIG, GESTURE_EMOJIS, API_GESTURE_EMOJIS
)
from components.ui_components import (
    render_sidebar_settings, render_metrics_dashboard, 
    render_gestures_status, render_camera_feed_header,
    render_recent_photos, render_debug_info, show_toast_notification,
    render_model_status_banner, render_api_specific_info,
    show_model_switch_warning, render_performance_metrics
)
from components.auth_components import (
    render_model_selector, render_api_connection_settings,
    handle_api_authentication, render_auth_status, init_session_state
)

# Import des gestionnaires
from model_manager import ModelManager, ModelType

# 📱 Configuration de l'app
st.set_page_config(**PAGE_CONFIG)

# 🎨 Application du thème
st.markdown(SOFT_PURPLE, unsafe_allow_html=True)

# 🎯 Titre principal
st.title("🕺 YOLO DANCE – Détection Multi-Modèles")

# 📁 Créer le dossier d'images
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔧 Initialisation des états de session
init_session_state()

# 🤖 Initialiser le gestionnaire de modèles
@st.cache_resource
def init_model_manager():
    return ModelManager()

model_manager = init_model_manager()

# 💾 Variables de session pour la prise de photos
if 'photo_count' not in st.session_state:
    st.session_state.photo_count = 0
if 'last_photo_time' not in st.session_state:
    st.session_state.last_photo_time = 0
if 'current_gesture' not in st.session_state:
    st.session_state.current_gesture = "Neutral"
if 'gesture_start_time' not in st.session_state:
    st.session_state.gesture_start_time = 0
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'fps_start_time' not in st.session_state:
    st.session_state.fps_start_time = time.time()

# ⚙️ Configuration API dans la sidebar
api_url = render_api_connection_settings()

# 🤖 Sélection du modèle dans la sidebar
selected_model_type = render_model_selector(model_manager)

# 🔄 Gestion du changement de modèle
if st.session_state.get('selected_model_type') != selected_model_type:
    old_model = st.session_state.get('selected_model_type', 'teachable_machine')
    show_model_switch_warning(old_model, selected_model_type)
    st.session_state['selected_model_type'] = selected_model_type

# 🔧 Initialisation des modèles selon la sélection
if selected_model_type == "teachable_machine":
    if model_manager.tm_detector is None:
        with st.spinner("🤖 Initialisation du modèle Teachable Machine..."):
            success = model_manager.initialize_teachable_machine(MODEL_DIR)
            if success:
                st.success("✅ Modèle Teachable Machine initialisé")
            else:
                st.error("❌ Échec de l'initialisation du modèle TM")
    
    # Basculer vers TM
    model_manager.set_model_type(ModelType.TEACHABLE_MACHINE)

elif selected_model_type == "api":
    if model_manager.api_adapter is None:
        with st.spinner("🔗 Initialisation de l'adaptateur API..."):
            success = model_manager.initialize_api(api_url)
            if success:
                st.success("✅ Adaptateur API initialisé")
            else:
                st.error("❌ Échec de l'initialisation de l'API")

# 🔐 Gestion de l'authentification API
if selected_model_type == "api":
    render_auth_status(model_manager)
    
    if not st.session_state.get('api_authenticated', False):
        st.info("🔐 **Authentification requise pour utiliser l'API**")
        
        authenticated = handle_api_authentication(model_manager)
        
        if not authenticated:
            st.stop()  # Arrêter l'exécution si pas authentifié
    
    # Si authentifié, basculer vers l'API
    if st.session_state.get('api_authenticated', False):
        model_manager.set_model_type(ModelType.API)

# 📊 Banneau de statut du modèle
render_model_status_banner(model_manager)

# ⚙️ Paramètres dans la sidebar
settings = render_sidebar_settings(model_manager)

# 📊 Informations spécifiques à l'API
if selected_model_type == "api":
    render_api_specific_info(model_manager)

# 📊 Tableau de bord avec métriques
cooldown_remaining = max(0, settings['photo_cooldown'] - (time.time() - st.session_state.last_photo_time))
render_metrics_dashboard(
    st.session_state.photo_count,
    st.session_state.current_gesture,
    cooldown_remaining,
    settings['confidence_threshold']
)

# 🎭 Affichage des gestes disponibles
available_gestures = model_manager.get_available_gestures() if model_manager.is_model_ready() else []
render_gestures_status(None, settings['priority_gestures'])

def get_gesture_emoji(gesture, model_type):
    """Obtient l'emoji correspondant au geste selon le modèle"""
    if model_type == "api":
        return API_GESTURE_EMOJIS.get(gesture, "🕺")
    else:
        return GESTURE_EMOJIS.get(gesture, "🕺")

def take_photo(frame, gesture, confidence, model_type):
    """Prend une photo si les conditions sont remplies"""
    current_time = time.time()
    if current_time - st.session_state.last_photo_time > settings['photo_cooldown']:
        st.session_state.photo_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Préfixe selon le modèle
        prefix = "api" if model_type == "api" else "tm"
        filename = f"{prefix}_{gesture}_{confidence:.2f}_{ts}_{st.session_state.photo_count:03d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        cv2.imwrite(filepath, frame)
        st.session_state.last_photo_time = current_time
        
        emoji = get_gesture_emoji(gesture, model_type)
        show_toast_notification(f"{emoji} {gesture.upper()} détecté ({confidence:.1%}) ! Photo : {filename}")
        return True
    return False

def process_detection(frame, model_manager, settings):
    """Traite la détection avec le modèle actuel"""
    if not model_manager.is_model_ready():
        return frame, "Neutral", False, False, {}
    
    # Obtenir la prédiction principale
    predicted_gesture, confidence = model_manager.predict_pose(frame)
    
    # Obtenir toutes les prédictions pour l'affichage
    all_predictions = model_manager.get_all_predictions(frame)
    
    # Dessiner les keypoints/indicateurs
    frame_with_overlay = model_manager.draw_keypoints(frame.copy())
    
    # Vérifier si le geste est détecté avec suffisamment de confiance
    gesture_detected = confidence >= settings['confidence_threshold']
    
    # Prioriser certains gestes
    if gesture_detected and predicted_gesture in settings['priority_gestures']:
        gesture_detected = True
    elif gesture_detected and predicted_gesture not in settings['priority_gestures']:
        # Réduire la sensibilité pour les gestes non prioritaires
        threshold_bonus = 0.1 if model_manager.current_model_type == ModelType.TEACHABLE_MACHINE else 0.05
        gesture_detected = confidence >= settings['confidence_threshold'] + threshold_bonus
    
    # Gestion de la continuité du geste
    current_time = time.time()
    photo_taken = False
    
    gesture_duration_threshold = model_manager.get_model_specific_settings().get(
        'gesture_duration_threshold', DEFAULT_GESTURE_DURATION_THRESHOLD
    )
    
    if gesture_detected and predicted_gesture == st.session_state.current_gesture:
        # Geste maintenu
        if current_time - st.session_state.gesture_start_time > gesture_duration_threshold:
            model_type = model_manager.current_model_type.value
            photo_taken = take_photo(frame, predicted_gesture, confidence, model_type)
    elif gesture_detected and predicted_gesture != st.session_state.current_gesture:
        # Nouveau geste détecté
        st.session_state.current_gesture = predicted_gesture
        st.session_state.gesture_start_time = current_time
    elif not gesture_detected:
        # Aucun geste détecté avec suffisamment de confiance
        st.session_state.current_gesture = "Neutral"
    
    return frame_with_overlay, predicted_gesture, gesture_detected, photo_taken, all_predictions

# 🎥 Section caméra
render_camera_feed_header()

# Vérifier que le modèle est prêt
if not model_manager.is_model_ready():
    if selected_model_type == "teachable_machine":
        st.error("❌ Modèle Teachable Machine non initialisé")
    elif selected_model_type == "api":
        st.error("❌ API non accessible ou authentification requise")
    st.stop()

if st.button("🎥 Lancer la détection", type="primary"):
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
        model_name = "Teachable Machine" if selected_model_type == "teachable_machine" else "API YOLOv8"
        st.success(f"✅ Caméra active - Modèle {model_name} prêt")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur lors de la lecture de la caméra.")
                break
            
            # Miroir pour l'UX
            frame = cv2.flip(frame, 1)
            
            # Traitement avec le modèle actuel
            processed_frame, detected_gesture, gesture_detected, photo_taken, all_predictions = process_detection(
                frame, model_manager, settings
            )
            
            # Ajouter des informations visuelles
            if gesture_detected:
                confidence = all_predictions.get(detected_gesture, 0.0) if all_predictions else 0.0
                cv2.putText(processed_frame, f"GESTE: {detected_gesture.upper()}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Confiance: {confidence:.1%}", (50, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Indiquer le modèle utilisé
            model_indicator = "TM" if selected_model_type == "teachable_machine" else "API"
            cv2.putText(processed_frame, f"Modele: {model_indicator}", (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Effet flash pour les photos
            if photo_taken:
                overlay = np.ones_like(processed_frame) * 255
                processed_frame = cv2.addWeighted(processed_frame, 0.6, overlay, 0.4, 0)
            
            # Affichage vidéo
            video_container.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Affichage du statut
            emoji = get_gesture_emoji(detected_gesture, selected_model_type)
            status_text = f"**Statut:** {emoji} {detected_gesture} {'✅' if gesture_detected else '❌'}"
            if all_predictions and detected_gesture in all_predictions:
                status_text += f" | Confiance: {all_predictions[detected_gesture]:.1%}"
            status_container.write(status_text)
            
            # Affichage des top prédictions
            if all_predictions:
                sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                top_3 = sorted_predictions[:3]
                
                pred_text = "**Top 3 prédictions:**\n"
                for gesture, conf in top_3:
                    emoji = get_gesture_emoji(gesture, selected_model_type)
                    pred_text += f"- {emoji} {gesture}: {conf:.1%}\n"
                predictions_container.write(pred_text)
            
            # Debug info si activé
            if settings['debug_mode']:
                render_debug_info(all_predictions, detected_gesture, gesture_detected, model_manager)
            
            # Métriques de performance
            frame_count += 1
            render_performance_metrics(frame_count, fps_start_time)
            
            time.sleep(0.03)  # Limite FPS
        
        cap.release()
        st.success("🎥 Caméra fermée")

# 📸 Affichage des photos récentes
if st.session_state.photo_count > 0:
    render_recent_photos(SAVE_DIR)

# 📊 Informations techniques dans un expander
with st.expander("🔧 Informations techniques", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modèle actuel")
        model_info = model_manager.get_model_info()
        st.json(model_info)
    
    with col2:
        st.subheader("Configuration")
        st.json({
            "api_url": api_url,
            "available_gestures": len(available_gestures),
            "settings": settings
        })