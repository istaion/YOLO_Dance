import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import time
import torch
import sys

# Ajouter le chemin vers les scripts YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'yolo_pipeline'))

# Importer le système pondéré
try:
    from yolo_model_weighted import YoloDanceSystemWeighted
    WEIGHTED_AVAILABLE = True
except ImportError:
    from yolo_model import YoloDanceSystem
    WEIGHTED_AVAILABLE = False
    st.warning("⚠️ Modèle pondéré non disponible, utilisation du modèle standard")

def show_yolo_dance_page():
    """Page YOLO Dance avec support de pondération"""
    
    # Configuration des chemins
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "yolo_dance")
    
    # Chemins des modèles
    WEIGHTED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_weighted_model.pth")
    STANDARD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pth")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Sélection du type de modèle
    st.sidebar.header("🎯 Type de modèle")
    
    model_options = []
    if WEIGHTED_AVAILABLE and os.path.exists(WEIGHTED_MODEL_PATH):
        model_options.append("Pondéré (Recommandé)")
    if os.path.exists(STANDARD_MODEL_PATH):
        model_options.append("Standard")
    
    if not model_options:
        st.error("❌ Aucun modèle YOLO trouvé!")
        st.info("Placez un modèle dans le dossier models/")
        return
    
    selected_model = st.sidebar.selectbox(
        "Sélectionner le modèle",
        model_options,
        index=0
    )
    
    # Affichage des informations du modèle
    if selected_model == "Pondéré (Recommandé)":
        st.sidebar.success("🎯 Modèle pondéré sélectionné")
        st.sidebar.info("""
        **Pondération optimisée:**
        - 🏆 Pose YOLO: 3.0x (priorité max)
        - 📱 Mains MediaPipe: 0.5x (réduit)
        - 📊 Context: 1.0x (standard)
        """)
        model_path = WEIGHTED_MODEL_PATH
        use_weighted = True
    else:
        st.sidebar.info("📊 Modèle standard sélectionné")
        st.sidebar.warning("Toutes les features ont le même poids")
        model_path = STANDARD_MODEL_PATH
        use_weighted = False
    
    # Initialiser le système selon le type
    @st.cache_resource
    def init_yolo_dance(model_path, use_weighted):
        try:
            if use_weighted and WEIGHTED_AVAILABLE:
                dance_system = YoloDanceSystemWeighted(custom_classifier_path=model_path)
                st.success("✅ Modèle YOLO Dance pondéré chargé!")
            else:
                dance_system = YoloDanceSystem(custom_classifier_path=model_path)
                st.success("✅ Modèle YOLO Dance standard chargé!")
            return dance_system
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
            return None
    
    # Variables de session
    session_prefix = 'weighted_' if use_weighted else 'standard_'
    
    for key in ['photo_count', 'last_photo_time', 'current_gesture', 'gesture_start_time', 'confidence_threshold']:
        session_key = f'{session_prefix}{key}'
        if session_key not in st.session_state:
            if key == 'confidence_threshold':
                st.session_state[session_key] = 0.7 if use_weighted else 0.6
            elif key == 'photo_count':
                st.session_state[session_key] = 0
            elif key == 'last_photo_time':
                st.session_state[session_key] = 0
            elif key == 'current_gesture':
                st.session_state[session_key] = "neutral"
            elif key == 'gesture_start_time':
                st.session_state[session_key] = 0
    
    # Configuration
    PHOTO_COOLDOWN = 2.0
    GESTURE_DURATION_THRESHOLD = 1.0
    
    st.title("🤖 YOLO DANCE – Modèle Optimisé")
    
    if use_weighted:
        st.write("**🎯 Détection pondérée : Priorité maximale aux poses corporelles**")
    else:
        st.write("**📊 Détection standard : Toutes features équilibrées**")
    
    # Paramètres dans la sidebar
    st.sidebar.header("⚙️ Paramètres de détection")
    
    confidence_threshold = st.sidebar.slider(
        "Seuil de confiance", 
        min_value=0.1, 
        max_value=1.0, 
        value=st.session_state[f'{session_prefix}confidence_threshold'], 
        step=0.05,
        help="Plus élevé = détection plus stricte"
    )
    st.session_state[f'{session_prefix}confidence_threshold'] = confidence_threshold
    
    photo_cooldown = st.sidebar.slider(
        "Délai entre photos (s)", 
        min_value=1.0, 
        max_value=10.0, 
        value=PHOTO_COOLDOWN, 
        step=0.5
    )
    
    # Classes et gestes prioritaires
    yolo_classes = ['hands_up', 'dab', 'twerk', 'jul', 'neutral', 'crossarm']
    
    if use_weighted:
        # Pour le modèle pondéré, recommander les gestes bien détectés par YOLO pose
        default_priority = ['dab', 'hands_up', 'crossarm', 'twerk']
        st.sidebar.info("💡 Gestes recommandés pour modèle pondéré: dab, hands_up, crossarm")
    else:
        default_priority = ['dab', 'twerk', 'hands_up', 'jul', 'crossarm']
    
    priority_gestures = st.sidebar.multiselect(
        "Gestes à capturer en priorité",
        yolo_classes,
        default=default_priority
    )
    
    # Mode de détection
    detection_mode = st.sidebar.selectbox(
        "Mode de détection",
        ["Single Person", "Multi Person"],
        index=0,
        help="Single Person recommandé pour meilleure précision"
    )
    
    def take_yolo_photo(frame, gesture, confidence, model_type):
        """Prend une photo avec préfixe selon le modèle"""
        current_time = time.time()
        session_key = f'{session_prefix}last_photo_time'
        
        if current_time - st.session_state[session_key] > photo_cooldown:
            st.session_state[f'{session_prefix}photo_count'] += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            prefix = "yolo_weighted" if model_type == "weighted" else "yolo_standard"
            filename = f"{prefix}_{gesture}_{confidence:.2f}_{ts}_{st.session_state[f'{session_prefix}photo_count']:03d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            st.session_state[session_key] = current_time
            
            model_name = "PONDÉRÉ" if model_type == "weighted" else "STANDARD"
            st.toast(f"📸 {model_name} - {gesture.upper()} détecté ({confidence:.1%}) !")
            return True
        return False
    
    def process_yolo_detection(frame, yolo_system, model_type):
        """Traite la détection avec monitoring de pondération"""
        if yolo_system is None:
            return frame, "neutral", False, False, {}, {}
        
        try:
            if detection_mode == "Single Person":
                result = yolo_system.predict(frame)
            else:
                result = yolo_system.predict_multi(frame)
                if 'best_pose' in result:
                    result = result['best_pose']
            
            predicted_gesture = result.get('predicted_class', 'neutral')
            confidence = result.get('confidence', 0.0)
            all_predictions = result.get('all_probabilities', {})
            debug_info = result.get('debug_info', {})
            
            # Frame avec visualisation
            frame_with_detection = frame.copy()
            
            # Couleur selon le modèle et la confiance
            if model_type == "weighted":
                color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)  # Vert ou Orange
                model_text = f"PONDÉRÉ: {predicted_gesture.upper()}"
            else:
                color = (255, 0, 0) if confidence >= confidence_threshold else (0, 0, 255)  # Rouge ou Bleu
                model_text = f"STANDARD: {predicted_gesture.upper()}"
            
            # Dessiner l'interface
            cv2.rectangle(frame_with_detection, (10, 10), (frame.shape[1]-10, 120), color, 3)
            cv2.putText(frame_with_detection, model_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame_with_detection, f"Conf: {confidence:.1%}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Affichage de la pondération si disponible
            if model_type == "weighted" and debug_info:
                pose_weight = debug_info.get('pose_features_weight', 'N/A')
                hand_weight = debug_info.get('hand_features_weight', 'N/A')
                cv2.putText(frame_with_detection, f"Pose:{pose_weight}x Hand:{hand_weight}x", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Logique de détection
            gesture_detected = confidence >= confidence_threshold
            
            if gesture_detected and predicted_gesture in priority_gestures:
                gesture_detected = True
            elif gesture_detected and predicted_gesture not in priority_gestures:
                gesture_detected = confidence >= confidence_threshold + 0.1
            
            # Gestion de continuité
            current_time = time.time()
            photo_taken = False
            current_gesture_key = f'{session_prefix}current_gesture'
            gesture_start_key = f'{session_prefix}gesture_start_time'
            
            if gesture_detected and predicted_gesture == st.session_state[current_gesture_key]:
                if current_time - st.session_state[gesture_start_key] > GESTURE_DURATION_THRESHOLD:
                    photo_taken = take_yolo_photo(frame, predicted_gesture, confidence, model_type)
            elif gesture_detected and predicted_gesture != st.session_state[current_gesture_key]:
                st.session_state[current_gesture_key] = predicted_gesture
                st.session_state[gesture_start_key] = current_time
            elif not gesture_detected:
                st.session_state[current_gesture_key] = "neutral"
            
            return frame_with_detection, predicted_gesture, gesture_detected, photo_taken, all_predictions, debug_info
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")
            return frame, "neutral", False, False, {}, {}
    
    # Interface utilisateur principale
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Photos prises", st.session_state[f'{session_prefix}photo_count'])
    with col2:
        st.metric("Geste détecté", st.session_state[f'{session_prefix}current_gesture'])
    with col3:
        cooldown_remaining = max(0, photo_cooldown - (time.time() - st.session_state[f'{session_prefix}last_photo_time']))
        st.metric("Cooldown", f"{cooldown_remaining:.1f}s")
    with col4:
        st.metric("Seuil", f"{confidence_threshold:.1%}")
    
    # Informations sur le modèle
    model_type_display = "Pondéré 🎯" if use_weighted else "Standard 📊"
    st.info(f"🤖 **Modèle:** {model_type_display} | **Mode:** {detection_mode} | **Classes:** {len(yolo_classes)}")
    
    # Section caméra
    st.header("🎥 Caméra YOLO Dance")
    
    yolo_system = init_yolo_dance(model_path, use_weighted)
    model_type = "weighted" if use_weighted else "standard"
    
    if st.button(f"🤖 Lancer {model_type_display}", type="primary", key="start_yolo"):
        if yolo_system is None:
            st.error("❌ Modèle non chargé. Impossible de démarrer.")
            st.stop()
        
        # Conteneurs d'affichage
        video_container = st.empty()
        status_container = st.empty()
        predictions_container = st.empty()
        debug_container = st.empty()
        
        # Bouton d'arrêt
        stop_button = st.button("🛑 Arrêter la caméra", key="stop_yolo")
        
        # Initialiser la caméra
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Impossible d'accéder à la caméra")
        else:
            success_msg = f"✅ Caméra active - Modèle {model_type_display} prêt"
            if use_weighted:
                success_msg += " (Pose prioritaire)"
            st.success(success_msg)
            
            frame_count = 0
            fps_start_time = time.time()
            fps = 0
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur lors de la lecture de la caméra.")
                    break
                
                # Miroir pour l'UX
                frame = cv2.flip(frame, 1)
                
                # Traitement avec le modèle sélectionné
                processed_frame, detected_gesture, gesture_detected, photo_taken, all_predictions, debug_info = process_yolo_detection(
                    frame, yolo_system, model_type
                )
                
                # Effet flash pour les photos
                if photo_taken:
                    overlay = np.ones_like(processed_frame) * 255
                    processed_frame = cv2.addWeighted(processed_frame, 0.6, overlay, 0.4, 0)
                
                # Ajouter le FPS et device info
                device_info = "GPU" if torch.cuda.is_available() else "CPU"
                cv2.putText(processed_frame, f"FPS: {fps:.1f} | {device_info}", 
                           (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Affichage vidéo
                video_container.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Affichage du statut avec info pondération
                status_icon = "✅" if gesture_detected else "❌"
                confidence_val = all_predictions.get(detected_gesture, 0.0) if all_predictions else 0.0
                
                status_text = f"**Statut {model_type_display}:** {detected_gesture} {status_icon}"
                if confidence_val > 0:
                    status_text += f" | Confiance: {confidence_val:.1%}"
                
                if use_weighted and debug_info:
                    hands_detected = debug_info.get('hands_detected', 0)
                    status_text += f" | Mains: {hands_detected}"
                
                status_container.write(status_text)
                
                # Affichage des prédictions avec pondération
                if all_predictions:
                    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
                    top_5 = sorted_predictions[:5]
                    
                    pred_text = f"**Top 5 prédictions {model_type_display}:**\n"
                    for i, (gesture, conf) in enumerate(top_5):
                        if use_weighted:
                            # Indiquer quels gestes sont favorisés par la pondération
                            if gesture in ['hands_up', 'crossarm']:  # Gestes bien détectés par pose
                                emoji = "🎯"
                            elif gesture in ['dab']:  # Gestes nécessitant les mains
                                emoji = "👋"
                            else:
                                emoji = "📊"
                        else:
                            emoji = "🎯" if conf > 0.5 else "📊"
                        
                        pred_text += f"{emoji} {gesture}: {conf:.1%}\n"
                    predictions_container.write(pred_text)
                
                # Debug info pour modèle pondéré
                if use_weighted and debug_info:
                    debug_text = "**🎯 Info pondération:**\n"
                    debug_text += f"- Pose weight: {debug_info.get('pose_features_weight', 'N/A')}x\n"
                    debug_text += f"- Hand weight: {debug_info.get('hand_features_weight', 'N/A')}x\n"
                    debug_text += f"- Context weight: {debug_info.get('context_features_weight', 'N/A')}x\n"
                    debug_text += f"- Pose features: {debug_info.get('pose_features_size', 'N/A')}\n"
                    debug_text += f"- Hand features: {debug_info.get('hand_features_size', 'N/A')}\n"
                    debug_container.write(debug_text)
                
                # Calcul FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                time.sleep(0.03)  # Limite FPS
            
            cap.release()
            st.success(f"🎥 Caméra {model_type_display} fermée")
    
    # Section photos récentes avec tri par modèle
    if st.session_state[f'{session_prefix}photo_count'] > 0:
        st.header(f"📸 Photos {model_type_display}")
        
        if os.path.exists(SAVE_DIR):
            # Filtrer selon le type de modèle
            if use_weighted:
                pattern = "yolo_weighted_"
            else:
                pattern = "yolo_standard_"
            
            image_files = [f for f in os.listdir(SAVE_DIR) 
                          if f.endswith('.jpg') and f.startswith(pattern)]
            image_files.sort(reverse=True)
            
            if image_files:
                # Afficher les 6 dernières photos
                cols = st.columns(3)
                for i, img_file in enumerate(image_files[:6]):
                    with cols[i % 3]:
                        img_path = os.path.join(SAVE_DIR, img_file)
                        # Extraire des infos du nom de fichier
                        parts = img_file.split('_')
                        if len(parts) >= 4:
                            gesture = parts[2]
                            confidence = parts[3]
                            caption = f"{gesture} ({confidence})"
                        else:
                            caption = img_file
                        
                        st.image(img_path, caption=caption, use_column_width=True)
            else:
                st.info(f"Aucune photo prise avec le modèle {model_type_display}")
    
    # Section statistiques et comparaison
    st.header(f"📊 Statistiques {model_type_display}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            f"Photos {model_type_display}", 
            st.session_state[f'{session_prefix}photo_count'],
            help=f"Nombre total de photos avec modèle {model_type_display}"
        )
    
    with col2:
        device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.metric(
            "Device", 
            device_info,
            help="Processeur utilisé pour l'inférence"
        )
    
    with col3:
        model_size = "Pondéré" if use_weighted else "Standard"
        st.metric(
            "Type modèle",
            model_size,
            help="Architecture du modèle utilisé"
        )
    
    # Section de comparaison des modèles
    if WEIGHTED_AVAILABLE:
        with st.expander("🔍 Comparaison des modèles"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Modèle Pondéré**")
                st.write("✅ Pose YOLO: priorité maximale (3.0x)")
                st.write("✅ Meilleure précision sur gestes corporels")
                st.write("✅ Moins sensible au bruit des mains")
                st.write("✅ Recommandé: dab, hands_up, crossarm")
                st.write("❌ Moins précis sur gestes manuels fins")
            
            with col2:
                st.write("**📊 Modèle Standard**")
                st.write("📊 Toutes features équilibrées")
                st.write("📊 Polyvalent sur tous types de gestes")
                st.write("📊 Sensible aux détails des mains")
                st.write("❌ Plus de faux positifs possibles")
                st.write("❌ Peut être perturbé par le bruit")
    
    # Conseils d'utilisation selon le modèle
    with st.expander("💡 Conseils d'utilisation"):
        if use_weighted:
            st.write("""
            **🎯 Optimisation pour modèle pondéré :**
            - 💃 Privilégiez les gestes corporels marqués (dab, mains levées)
            - 🎯 Les poses YOLO sont prioritaires - soignez votre posture
            - 📏 Restez bien visible dans le cadre (corps entier)
            - ⚡ Réactivité améliorée sur les gestes de pose
            - 🎭 Idéal pour: dab, hands_up, crossarm, twerk
            """)
        else:
            st.write("""
            **📊 Utilisation modèle standard :**
            - 👋 Tous types de gestes équilibrés
            - 🤲 Attention aux détails des mains
            - 🎯 Polyvalent mais potentiellement plus sensible au bruit
            - 💡 Bon éclairage recommandé pour les mains
            - 🎭 Adapté à: tous gestes, y compris ceux nécessitant les mains
            """)
        
        st.write("""
        **Conseils généraux :**
        - 🎯 Éclairage uniforme et suffisant
        - 📏 Distance 1.5-2m de la caméra
        - ⏱️ Maintenez la pose 1-2 secondes
        - 🎬 Arrière-plan dégagé et contrasté
        """)

# Point d'entrée mis à jour
def show_yolo_dance_page_updated():
    """Point d'entrée pour la page YOLO avec support pondération"""
    return show_yolo_dance_page()