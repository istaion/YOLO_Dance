# app/main_app.py
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import time
import sys

# Ajouter le chemin vers les scripts YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'yolo_pipeline'))

# Importer les deux pages
from teachable_machine_adapter import TeachableMachinePoseDetector
from yolo_dance_page import show_yolo_dance_page

SOFT_PURPLE = """
<style>
/* SUPPRIMER LA BARRE DU HAUT */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    .stApp {
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 20%, #ddd6fe 40%, #c4b5fd 60%, #a78bfa 80%, #8b5cf6 100%);
    }
    
    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #f3e8ff 0%, #ddd6fe 100%);
        border-right: 2px solid #8b5cf6;
    }
    
    /* TOUT LE TEXTE SIDEBAR EN NOIR */
    .stSidebar .stMarkdown h1, 
    .stSidebar .stMarkdown h2, 
    .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown p,
    .stSidebar .stMarkdown li,
    .stSidebar label,
    .stSidebar .stSlider label,
    .stSidebar .stSelectbox label,
    .stSidebar .stMultiSelect label,
    .stSidebar .stCheckbox label,
    .stSidebar span,
    .stSidebar div {
        color: #000000 !important;
        font-weight: 600;
    }
    
    /* TITRE PRINCIPAL EN NOIR - TOUTES LES VARIANTES */
    .stMarkdown h1,
    h1,
    [data-testid="stHeader"] h1,
    .css-1v0mbdj h1,
    .css-1629p8f h1 {
        color: #000000 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        font-weight: bold !important;
    }
    
    .stMarkdown h2, .stMarkdown h3 {
        color: #374151 !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        color: #111827 !important;
        font-weight: 500;
    }
    
    .stMetric {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid #c4b5fd;
        border-radius: 12px;
        padding: 12px;
    }
    
    .stMetric label {
        color: #374151 !important;
        font-weight: 600;
    }
    
    .stMetric div {
        color: #1f2937 !important;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #8b5cf6, #a78bfa);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #7c3aed, #8b5cf6);
        transform: translateY(-1px);
    }
    
    /* Style pour les boutons de navigation */
    .nav-button {
        background: linear-gradient(45deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: bold;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background: linear-gradient(45deg, #4f46e5, #7c3aed);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .nav-button.active {
        background: linear-gradient(45deg, #059669, #10b981);
    }
</style>
"""

def show_teachable_machine_page():
    """Page pour Teachable Machine (code original)"""
    
    # Configuration des chemins
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "teachable_machine")
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "teachable_machine")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialiser le détecteur Teachable Machine
    @st.cache_resource
    def init_tm_detector():
        try:
            detector = TeachableMachinePoseDetector(MODEL_DIR)
            return detector
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {e}")
            return None
    
    # Variables de session
    if 'photo_count' not in st.session_state:
        st.session_state.photo_count = 0
    if 'last_photo_time' not in st.session_state:
        st.session_state.last_photo_time = 0
    if 'current_gesture' not in st.session_state:
        st.session_state.current_gesture = "Neutral"
    if 'gesture_start_time' not in st.session_state:
        st.session_state.gesture_start_time = 0
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.7
    
    # Configuration
    PHOTO_COOLDOWN = 2.0
    GESTURE_DURATION_THRESHOLD = 1.0
    
    st.title("🎭 TEACHABLE MACHINE – On ne capture pas, on éternise l'instant")
    
    # Paramètres dans la sidebar
    st.sidebar.header("⚙️ Paramètres Teachable Machine")
    confidence_threshold = st.sidebar.slider(
        "Seuil de confiance", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    st.session_state.confidence_threshold = confidence_threshold
    
    photo_cooldown = st.sidebar.slider(
        "Délai entre photos (s)", 
        min_value=1.0, 
        max_value=10.0, 
        value=PHOTO_COOLDOWN, 
        step=0.5
    )
    
    # Gestes prioritaires (modifiez selon vos préférences)
    priority_gestures = st.sidebar.multiselect(
        "Gestes à capturer en priorité",
        ["Twerk", "Dab", "Macaréna", "Floss", "Funk", "CrossArm", "V_Signs"],
        default=["Twerk", "Dab", "Macaréna", "Floss", "Funk","CrossArm"]
    )
    
    def take_photo(frame, gesture, confidence):
        """Prend une photo si les conditions sont remplies"""
        current_time = time.time()
        if current_time - st.session_state.last_photo_time > photo_cooldown:
            st.session_state.photo_count += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tm_{gesture}_{confidence:.2f}_{ts}_{st.session_state.photo_count:03d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            st.session_state.last_photo_time = current_time
            st.toast(f"📸 {gesture.upper()} détecté ({confidence:.1%}) ! Photo : {filename}")
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
        gesture_detected = confidence >= st.session_state.confidence_threshold
        
        # Prioriser certains gestes
        if gesture_detected and predicted_gesture in priority_gestures:
            gesture_detected = True
        elif gesture_detected and predicted_gesture not in priority_gestures:
            # Réduire la sensibilité pour les gestes non prioritaires
            gesture_detected = confidence >= st.session_state.confidence_threshold + 0.1
        
        # Gestion de la continuité du geste
        current_time = time.time()
        photo_taken = False
        
        if gesture_detected and predicted_gesture == st.session_state.current_gesture:
            # Geste maintenu
            if current_time - st.session_state.gesture_start_time > GESTURE_DURATION_THRESHOLD:
                photo_taken = take_photo(frame, predicted_gesture, confidence)
        elif gesture_detected and predicted_gesture != st.session_state.current_gesture:
            # Nouveau geste détecté
            st.session_state.current_gesture = predicted_gesture
            st.session_state.gesture_start_time = current_time
        elif not gesture_detected:
            # Aucun geste détecté avec suffisamment de confiance
            st.session_state.current_gesture = "Neutral"
        
        return frame_with_keypoints, predicted_gesture, gesture_detected, photo_taken, all_predictions
    
    # Interface utilisateur principale
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Photos prises", st.session_state.photo_count)
    with col2:
        st.metric("Geste actuel", st.session_state.current_gesture)
    with col3:
        cooldown_remaining = max(0, photo_cooldown - (time.time() - st.session_state.last_photo_time))
        st.metric("Cooldown", f"{cooldown_remaining:.1f}s")
    with col4:
        st.metric("Seuil confiance", f"{confidence_threshold:.1%}")
    
    # Conseils à l'utilisateur
    st.write("**Lancez la détection pour que le modèle analyse vos poses**")
       
    # Section principale de la caméra
    st.header("🎥 Caméra Live Teachable Machine")
    
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
                
                # Calcul FPS (optionnel)
                frame_count += 1
                if frame_count % 30 == 0:
                    fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                time.sleep(0.03)  # Limite FPS
            
            cap.release()
            st.success("🎥 Caméra fermée")
    
    # Section d'affichage des photos récentes
    if st.session_state.photo_count > 0:
        st.header("📸 Photos récentes Teachable Machine")
        
        if os.path.exists(SAVE_DIR):
            image_files = [f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg') and f.startswith('tm_')]
            image_files.sort(reverse=True)
            
            if image_files:
                # Afficher les 6 dernières photos
                cols = st.columns(3)
                for i, img_file in enumerate(image_files[:6]):
                    with cols[i % 3]:
                        img_path = os.path.join(SAVE_DIR, img_file)
                        st.image(img_path, caption=img_file, use_container_width=True)
            else:
                st.info("Aucune photo prise avec le modèle Teachable Machine")

def main():
    """Fonction principale avec navigation"""
    
    # Configuration de l'app
    st.set_page_config(
        page_title="YOLO Dance & Teachable Machine", 
        page_icon="🕺",
        layout="wide"
    )
    
    # Application du style 
    st.markdown(SOFT_PURPLE, unsafe_allow_html=True)
    
    # Initialisation de la session state pour la navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'teachable_machine'
    
    # Barre de navigation dans la sidebar
    st.sidebar.title("🎮 Navigation")
    st.sidebar.markdown("---")
    
    # Boutons de navigation
    if st.sidebar.button("🎭 Teachable Machine", key="nav_tm", use_container_width=True):
        st.session_state.current_page = 'teachable_machine'
        st.rerun()
    
    if st.sidebar.button("🤖 YOLO Dance (Modèle Entraîné)", key="nav_yolo", use_container_width=True):
        st.session_state.current_page = 'yolo_dance'
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Affichage de la page actuelle dans la sidebar
    if st.session_state.current_page == 'teachable_machine':
        st.sidebar.success("🎭 **Teachable Machine** - Actif")
        st.sidebar.info("Modèle pré-entraîné avec détection MediaPipe")
    else:
        st.sidebar.success("🤖 **YOLO Dance** - Actif")
        st.sidebar.info("Modèle personnalisé entraîné")
    
    # Section de comparaison dans la sidebar
    with st.sidebar.expander("📊 Comparaison des modèles"):
        st.write("**Teachable Machine:**")
        st.write("- ✅ Facile à utiliser")
        st.write("- ✅ Pas d'entraînement requis")
        st.write("- ❌ Gestes prédéfinis")
        st.write("- ❌ Moins précis")
        
        st.write("**YOLO Dance:**")
        st.write("- ✅ Très précis")
        st.write("- ✅ Gestes personnalisés")
        st.write("- ✅ Multi-personnes")
        st.write("- ❌ Modèle à entraîner")
    
    # Affichage de la page selon la sélection
    if st.session_state.current_page == 'teachable_machine':
        show_teachable_machine_page()
    elif st.session_state.current_page == 'yolo_dance':
        show_yolo_dance_page()

if __name__ == "__main__":
    main()