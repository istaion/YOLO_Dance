import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import time
import requests
import base64
import json
from io import BytesIO
from PIL import Image

class YOLODanceAPIClient:
    """Client pour l'API YOLO Dance"""
    
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()
        
    def check_api_health(self):
        """Vérifie l'état de l'API"""
        try:
            response = self.session.get(f"{self.api_url}/health/", timeout=5)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_available_models(self):
        """Récupère la liste des modèles disponibles"""
        try:
            response = self.session.get(f"{self.api_url}/models/", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Erreur récupération modèles: {e}")
            return None
    
    def detect_gesture(self, image, model_type="weighted", apply_filters=False, 
                      custom_thresholds=None, return_image=False):
        """Détection de geste via l'API"""
        try:
            # Encoder l'image
            _, buffer = cv2.imencode('.jpg', image)
            
            # Préparer les données
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            data = {
                'model_type': model_type,
                'apply_filters': apply_filters,
                'return_image': return_image,
                'return_debug': True
            }
            
            if custom_thresholds:
                data['custom_thresholds'] = json.dumps(custom_thresholds)
            
            # Requête API
            response = self.session.post(
                f"{self.api_url}/detect_advanced/",
                files=files,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Décoder l'image si présente
                if 'annotated_image' in result:
                    img_data = base64.b64decode(result['annotated_image'])
                    nparr = np.frombuffer(img_data, np.uint8)
                    annotated_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    result['annotated_image_array'] = annotated_img
                
                return True, result
            else:
                return False, {"error": f"API Error: {response.status_code}"}
                
        except Exception as e:
            return False, {"error": str(e)}

def show_yolo_dance_page():
    """Page YOLO Dance utilisant l'API FastAPI"""
    
    # Configuration des chemins
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "images", "yolo_dance")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Configuration de l'API
    st.sidebar.header("🔌 Configuration API")
    api_url = st.sidebar.text_input(
        "URL de l'API", 
        value="http://localhost:8000",
        help="Adresse de votre API YOLO Dance"
    )
    
    # Initialiser le client API
    api_client = YOLODanceAPIClient(api_url)
    
    # Vérifier la connexion API
    is_connected, health_data = api_client.check_api_health()
    
    if is_connected:
        st.sidebar.success("✅ API connectée")
        
        # Afficher l'état des modèles
        models_status = health_data.get('models', {})
        for model_name, is_available in models_status.items():
            status_icon = "✅" if is_available else "❌"
            st.sidebar.write(f"{status_icon} {model_name.title()}")
        
        filters_status = "✅" if health_data.get('filters', False) else "❌"
        st.sidebar.write(f"{filters_status} Filtres")
        
    else:
        st.sidebar.error("❌ API non disponible")
        st.error("🚨 Impossible de se connecter à l'API YOLO Dance")
        st.info(f"""
        **Vérifiez que l'API est démarrée :**
        ```bash
        cd api/
        python main.py
        ```
        URL configurée: {api_url}
        """)
        return
    
    # Récupérer les modèles disponibles
    models_info = api_client.get_available_models()
    if not models_info:
        st.error("Impossible de récupérer les informations des modèles")
        return
    
    available_models = list(models_info.get('models', {}).keys())
    default_thresholds = models_info.get('default_thresholds', {})
    
    if not available_models:
        st.error("Aucun modèle disponible sur l'API")
        return
    
    # Sélection du type de modèle
    st.sidebar.header("🎯 Type de modèle")
    
    selected_model = st.sidebar.selectbox(
        "Sélectionner le modèle",
        available_models,
        index=0 if "weighted" in available_models else 0
    )
    
    # Affichage des informations du modèle
    model_info = models_info['models'][selected_model]
    if selected_model == "weighted":
        st.sidebar.success("🎯 Modèle pondéré sélectionné")
        st.sidebar.info("""
        **Pondération optimisée:**
        - 🏆 Pose YOLO: 3.0x (priorité max)
        - 📱 Mains MediaPipe: 0.5x (réduit)
        - 📊 Context: 1.0x (standard)
        """)
        use_weighted = True
    else:
        st.sidebar.info("📊 Modèle standard sélectionné")
        st.sidebar.warning("Toutes les features ont le même poids")
        use_weighted = False
    
    # Variables de session
    session_prefix = f'api_{selected_model}_'
    
    # Classes disponibles
    yolo_classes = model_info.get('classes', list(default_thresholds.keys()))
    
    # Initialiser les seuils par classe
    thresholds_key = f'{session_prefix}class_thresholds'
    if thresholds_key not in st.session_state:
        st.session_state[thresholds_key] = default_thresholds.copy()
    
    for key in ['photo_count', 'last_photo_time', 'current_gesture', 'gesture_start_time']:
        session_key = f'{session_prefix}{key}'
        if session_key not in st.session_state:
            if key == 'photo_count':
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
    
    st.title("🤖 YOLO DANCE – Modèle API")
    
    if use_weighted:
        st.write("**🎯 Détection pondérée via API : Priorité maximale aux poses corporelles**")
    else:
        st.write("**📊 Détection standard via API : Toutes features équilibrées**")
    
    # Paramètres dans la sidebar
    st.sidebar.header("⚙️ Paramètres de détection")
    
    # Section filtres visuels
    st.sidebar.subheader("🎨 Filtres visuels")
    filters_enabled = st.sidebar.checkbox(
        "Activer les filtres", 
        value=True,
        help="Applique des filtres visuels via l'API selon le geste détecté"
    )
    
    if filters_enabled:
        st.sidebar.success("Filtres activés via API")
        with st.sidebar.expander("🎭 Filtres par geste", expanded=False):
            st.write("🍑 **Twerk** : Pêche sur les hanches")
            st.write("👑 **Hands Up** : Couronne sur la tête")
            st.write("⚡ **Dab** : Rayon lumineux")
            st.write("✝️ **CrossArm** : Croix lumineuse")
            st.write("🕶️ **Jul** : Lunettes de soleil")
            st.write("😐 **Neutral** : Aucun filtre")
    
    # Seuil global (pour référence)
    global_threshold = st.sidebar.slider(
        "Seuil global (référence)", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Applique ce seuil à toutes les classes"
    )
    
    # Bouton pour appliquer le seuil global
    if st.sidebar.button("🎯 Appliquer seuil global à toutes les classes"):
        for cls in yolo_classes:
            st.session_state[thresholds_key][cls] = global_threshold
        st.sidebar.success(f"Seuil {global_threshold:.2f} appliqué à toutes les classes")
    
    # Seuils individuels par classe
    st.sidebar.subheader("🎛️ Seuils par classe")
    st.sidebar.caption("Personnalisez la sensibilité de chaque geste")
    
    # Grouper les classes pour un affichage organisé
    gesture_groups = {
        "🕺 Gestes principaux": ['dab', 'hands_up', 'twerk'],
        "🎭 Gestes spéciaux": ['jul', 'crossarm'],
        "⚖️ État neutre": ['neutral']
    }
    
    for group_name, group_classes in gesture_groups.items():
        # Filtrer les classes qui existent réellement
        existing_classes = [cls for cls in group_classes if cls in yolo_classes]
        if existing_classes:
            with st.sidebar.expander(group_name, expanded=True):
                for cls in existing_classes:
                    current_threshold = st.session_state[thresholds_key][cls]
                    
                    # Emoji pour chaque classe
                    class_emojis = {
                        'dab': '🕺',
                        'hands_up': '🙌',
                        'twerk': '💃',
                        'jul': '🎤',
                        'crossarm': '🤷',
                        'neutral': '😐'
                    }
                    
                    emoji = class_emojis.get(cls, '🎯')
                    
                    new_threshold = st.slider(
                        f"{emoji} {cls}",
                        min_value=0.1,
                        max_value=1.0,
                        value=current_threshold,
                        step=0.05,
                        key=f"threshold_{cls}_{session_prefix}",
                        help=f"Seuil de confiance pour détecter '{cls}'"
                    )
                    
                    st.session_state[thresholds_key][cls] = new_threshold
                    
                    # Indicateur visuel du niveau
                    if new_threshold >= 0.8:
                        st.caption("🟢 Très strict")
                    elif new_threshold >= 0.6:
                        st.caption("🟡 Équilibré")
                    else:
                        st.caption("🔴 Permissif")
    
    # Bouton reset des seuils
    if st.sidebar.button("🔄 Reset seuils (0.7 partout)"):
        for cls in yolo_classes:
            st.session_state[thresholds_key][cls] = 0.7
        st.sidebar.success("Seuils remis à 0.7 pour toutes les classes")
    
    photo_cooldown = st.sidebar.slider(
        "Délai entre photos (s)", 
        min_value=1.0, 
        max_value=10.0, 
        value=PHOTO_COOLDOWN, 
        step=0.5
    )
    
    def take_yolo_photo(frame, gesture, confidence, model_type):
        """Prend une photo avec préfixe selon le modèle (SANS FILTRE via API)"""
        # Exclure neutral des photos
        if gesture == 'neutral':
            return False
            
        current_time = time.time()
        session_key = f'{session_prefix}last_photo_time'
        
        if current_time - st.session_state[session_key] > photo_cooldown:
            st.session_state[f'{session_prefix}photo_count'] += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            prefix = f"api_{model_type}"
            filename = f"{prefix}_{gesture}_{confidence:.2f}_{ts}_{st.session_state[f'{session_prefix}photo_count']:03d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            # IMPORTANT: Sauvegarder la frame ORIGINALE sans filtre
            cv2.imwrite(filepath, frame)
            st.session_state[session_key] = current_time
            
            model_name = "PONDÉRÉ" if model_type == "weighted" else "STANDARD"
            st.toast(f"📸 API {model_name} - {gesture.upper()} détecté ({confidence:.1%}) !")
            return True
        return False
    
    # Interface utilisateur principale
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Photos prises", st.session_state[f'{session_prefix}photo_count'])
    with col2:
        current_gesture = st.session_state[f'{session_prefix}current_gesture']
        st.metric("Geste détecté", current_gesture)
    with col3:
        cooldown_remaining = max(0, photo_cooldown - (time.time() - st.session_state[f'{session_prefix}last_photo_time']))
        st.metric("Cooldown", f"{cooldown_remaining:.1f}s")
    with col4:
        # Afficher le seuil de la classe actuelle
        current_threshold = st.session_state[thresholds_key].get(current_gesture, 0.7)
        st.metric("Seuil actuel", f"{current_threshold:.1%}")
    
    # Affichage des seuils configurés
    with st.expander("🎛️ Seuils configurés", expanded=False):
        threshold_cols = st.columns(3)
        for i, (cls, threshold) in enumerate(st.session_state[thresholds_key].items()):
            with threshold_cols[i % 3]:
                emoji = {'dab': '🕺', 'hands_up': '🙌', 'twerk': '💃', 'jul': '🎤', 'crossarm': '🤷', 'neutral': '😐'}.get(cls, '🎯')
                st.metric(f"{emoji} {cls}", f"{threshold:.1%}")
    
    # Informations sur le modèle
    model_type_display = "Pondéré 🎯" if use_weighted else "Standard 📊"
    st.info(f"🔗 **API:** {api_url} | **Modèle:** {model_type_display} | **Filtres:** {'✅' if filters_enabled else '❌'}")
    
    # Section caméra
    st.header("🎥 Caméra YOLO Dance (API)")
    
    if st.button(f"🤖 Lancer {model_type_display} via API", type="primary", key="start_yolo_api"):
        # Conteneurs d'affichage
        video_container = st.empty()
        status_container = st.empty()
        api_info_container = st.empty()
        predictions_container = st.empty()
        
        # Bouton d'arrêt
        stop_button = st.button("🛑 Arrêter la caméra", key="stop_yolo_api")
        
        # Initialiser la caméra
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Impossible d'accéder à la caméra")
        else:
            success_msg = f"✅ Caméra active - API {model_type_display} prêt"
            st.success(success_msg)
            
            frame_count = 0
            api_call_count = 0
            fps_start_time = time.time()
            fps = 0
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur lors de la lecture de la caméra.")
                    break
                
                # Miroir pour l'UX
                frame = cv2.flip(frame, 1)
                original_frame = frame.copy()
                
                try:
                    # Appel API pour détection
                    success, result = api_client.detect_gesture(
                        frame,
                        model_type=selected_model,
                        apply_filters=filters_enabled,
                        custom_thresholds=st.session_state[thresholds_key],
                        return_image=filters_enabled
                    )
                    
                    api_call_count += 1
                    
                    if success:
                        detected = result.get('detected', False)
                        predicted_class = result.get('predicted_class', 'neutral')
                        confidence = result.get('confidence', 0.0)
                        all_probabilities = result.get('all_probabilities', {})
                        threshold_used = result.get('threshold_used', 0.7)
                        debug_info = result.get('debug_info', {})
                        
                        # Frame à afficher
                        if filters_enabled and 'annotated_image_array' in result:
                            display_frame = result['annotated_image_array']
                        else:
                            display_frame = frame.copy()
                            # Ajouter annotations basiques
                            color = (0, 255, 0) if detected else (0, 0, 255)
                            cv2.putText(display_frame, f"API {selected_model.upper()}: {predicted_class}: {confidence:.1%}", 
                                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        
                        # Gestion de continuité
                        current_time = time.time()
                        photo_taken = False
                        current_gesture_key = f'{session_prefix}current_gesture'
                        gesture_start_key = f'{session_prefix}gesture_start_time'
                        
                        if detected and predicted_class == st.session_state[current_gesture_key]:
                            if current_time - st.session_state[gesture_start_key] > GESTURE_DURATION_THRESHOLD:
                                # IMPORTANT: Utiliser original_frame (sans filtre) pour la photo
                                photo_taken = take_yolo_photo(original_frame, predicted_class, confidence, selected_model)
                        elif detected and predicted_class != st.session_state[current_gesture_key]:
                            st.session_state[current_gesture_key] = predicted_class
                            st.session_state[gesture_start_key] = current_time
                        elif not detected:
                            st.session_state[current_gesture_key] = "neutral"
                        
                        # Effet flash pour les photos
                        if photo_taken:
                            overlay = np.ones_like(display_frame) * 255
                            display_frame = cv2.addWeighted(display_frame, 0.6, overlay, 0.4, 0)
                        
                        # Ajouter le FPS et device info
                        cv2.putText(display_frame, f"FPS: {fps:.1f} | API calls: {api_call_count}", 
                                   (frame.shape[1]-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Affichage vidéo
                        video_container.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        
                        # Affichage du statut avec seuil utilisé
                        status_icon = "✅" if detected else "❌"
                        status_text = f"**Statut API {model_type_display}:** {predicted_class} {status_icon}"
                        if confidence > 0:
                            status_text += f" | Confiance: {confidence:.1%} (seuil: {threshold_used:.1%})"
                        status_container.write(status_text)
                        
                        # Informations API
                        api_info_text = f"**API Info:** Modèle: {result.get('model_used', 'N/A')} | Calls: {api_call_count}"
                        if debug_info and selected_model == "weighted":
                            pose_weight = debug_info.get('pose_features_weight', 'N/A')
                            hand_weight = debug_info.get('hand_features_weight', 'N/A')
                            api_info_text += f" | Pondération: Pose {pose_weight}x, Mains {hand_weight}x"
                        api_info_container.write(api_info_text)
                        
                        # Affichage des prédictions avec indication des seuils
                        if all_probabilities:
                            sorted_predictions = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                            top_5 = sorted_predictions[:5]
                            
                            pred_text = f"**Top 5 prédictions API {model_type_display}:**\n"
                            for i, (gesture, conf) in enumerate(top_5):
                                gesture_threshold = st.session_state[thresholds_key].get(gesture, 0.7)
                                
                                if conf >= gesture_threshold:
                                    status_emoji = "✅"
                                elif conf >= gesture_threshold - 0.1:
                                    status_emoji = "⚠️"
                                else:
                                    status_emoji = "❌"
                                
                                pred_text += f"{status_emoji} {gesture}: {conf:.1%} (seuil: {gesture_threshold:.1%})\n"
                            predictions_container.write(pred_text)
                    
                    else:
                        # Erreur API
                        error_msg = result.get('error', 'Erreur inconnue')
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"API ERROR: {error_msg[:50]}", 
                                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        video_container.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                        status_container.error(f"Erreur API: {error_msg}")
                
                except Exception as e:
                    st.error(f"Erreur communication API: {e}")
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "CONNECTION ERROR", 
                               (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    video_container.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Calcul FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                
                # Limiter FPS pour éviter la surcharge API
                time.sleep(0.1)  # 10 FPS max pour éviter la surcharge
            
            cap.release()
            st.success(f"🎥 Caméra {model_type_display} fermée")
    
    # Section photos récentes avec tri par modèle
    if st.session_state[f'{session_prefix}photo_count'] > 0:
        st.header(f"📸 Photos API {model_type_display}")
        
        if os.path.exists(SAVE_DIR):
            # Filtrer selon le type de modèle
            pattern = f"api_{selected_model}_"
            
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
                            model_type = parts[1]
                            gesture = parts[2]
                            confidence = parts[3]
                            used_threshold = st.session_state[thresholds_key].get(gesture, 0.7)
                            caption = f"API {model_type} - {gesture} (conf: {confidence}, seuil: {used_threshold:.1%})"
                        else:
                            caption = img_file
                        
                        st.image(img_path, caption=caption, use_column_width=True)
            else:
                st.info(f"Aucune photo prise avec le modèle API {model_type_display}")
    
    # Conseils d'utilisation selon le modèle
    with st.expander("💡 Conseils d'utilisation, seuils et API"):
        st.write("**🔗 Architecture API :**")
        st.write("""
        - **🚀 Performance** : Modèles chargés une fois côté serveur
        - **🎨 Filtres** : Traitement des filtres visuels via API
        - **📊 Scalabilité** : Plusieurs clients peuvent utiliser la même API
        - **🔧 Maintenance** : Mise à jour des modèles centralisée
        """)
        
        st.write("**🎛️ Configuration des seuils par classe :**")
        st.write("""
        - **Seuil élevé (0.8+)** : Détection très stricte, moins de faux positifs
        - **Seuil moyen (0.6-0.8)** : Équilibre entre précision et sensibilité  
        - **Seuil bas (0.4-0.6)** : Détection permissive, plus de vrais positifs mais plus de bruit
        
        **💡 Recommandations par geste :**
        - **Dab, Hands_up** : Seuil élevé (0.7-0.8) - gestes distinctifs
        - **Twerk, Jul** : Seuil moyen (0.6-0.7) - plus de nuances
        - **Crossarm** : Seuil moyen (0.6-0.7) - peut être confondu
        - **Neutral** : Seuil bas (0.4-0.6) - état par défaut (AUCUNE PHOTO)
        """)
        
        st.write("**🎨 Système de filtres visuels (via API) :**")
        st.write("""
        - **🍑 Twerk** : Pêche positionnée automatiquement sur les hanches
        - **👑 Hands Up** : Couronne royale au-dessus de la tête
        - **⚡ Dab** : Rayon lumineux dans l'alignement du bras levé
        - **✝️ CrossArm** : Croix lumineuse sur le torse
        - **🕶️ Jul** : Lunettes de soleil sur les yeux
        - **😐 Neutral** : Aucun filtre appliqué
        
        ⚠️ **Important** : Les filtres sont appliqués côté API.
        Les photos sauvegardées sont SANS filtre (image originale).
        """)
        
        st.write("**🔧 Avantages de l'API :**")
        st.write("""
        - **Performance** : Pas de rechargement de modèles
        - **Flexibilité** : Changement de modèle sans redémarrage
        - **Monitoring** : Logs centralisés des prédictions
        - **Multi-clients** : Partage des ressources
        """)

# Point d'entrée principal
if __name__ == "__main__":
    show_yolo_dance_page()