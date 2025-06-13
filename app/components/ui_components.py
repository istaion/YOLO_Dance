# app/components/ui_components.py

"""
Composants UI réutilisables pour l'application YOLO Dance
"""
import streamlit as st
import os
from config.settings import GESTURE_EMOJIS, DEFAULT_PRIORITY_GESTURES

# Fallback pour ALL_GESTURES si pas dans settings
try:
    from config.settings import ALL_GESTURES
except ImportError:
    ALL_GESTURES = [
        "Twerk", "Dab", "Macaréna", "Floss", "Funk", 
        "CrossArm", "V_Signs", "RussianMoove", "PasDuBourré",
        "CrossFeat", "GlassMoove", "Fuck", "JulSign", 
        "Neutral", "Blood"
    ]

def render_sidebar_settings(model_manager=None):
    """Rendu de la sidebar avec tous les paramètres"""
    st.sidebar.header("⚙️ Paramètres")
    
    # Obtenir les paramètres spécifiques au modèle
    if model_manager and model_manager.is_model_ready():
        model_settings = model_manager.get_model_specific_settings()
        available_gestures = model_manager.get_available_gestures()
        default_confidence = model_settings.get('confidence_threshold', 0.7)
        default_gestures = model_settings.get('priority_gestures', [])
    else:
        available_gestures = ALL_GESTURES
        default_confidence = 0.7
        default_gestures = DEFAULT_PRIORITY_GESTURES
    
    # Seuil de confiance
    confidence_threshold = st.sidebar.slider(
        "Seuil de confiance", 
        min_value=0.1, 
        max_value=1.0, 
        value=default_confidence, 
        step=0.05,
        help="Plus élevé = détection plus stricte"
    )
    
    # Délai entre photos
    photo_cooldown = st.sidebar.slider(
        "Délai entre photos (s)", 
        min_value=1.0, 
        max_value=10.0, 
        value=2.0, 
        step=0.5,
        help="Temps d'attente minimum entre 2 photos"
    )
    
    # Gestes prioritaires - adapté au modèle
    st.sidebar.subheader("🎯 Gestes à capturer")
    priority_gestures = st.sidebar.multiselect(
        "Sélectionnez les gestes prioritaires",
        available_gestures,
        default=[g for g in default_gestures if g in available_gestures],
        help="Ces gestes seront détectés plus facilement"
    )
    
    # Mode debug
    debug_mode = st.sidebar.checkbox(
        "Mode debug",
        value=False,
        help="Affiche des informations supplémentaires"
    )
    
    return {
        'confidence_threshold': confidence_threshold,
        'photo_cooldown': photo_cooldown,
        'priority_gestures': priority_gestures,
        'debug_mode': debug_mode
    }

def render_metrics_dashboard(photo_count, current_gesture, cooldown_remaining, confidence_threshold):
    """Affichage du tableau de bord avec métriques"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emoji = GESTURE_EMOJIS.get(current_gesture, "📸")
        st.metric(
            label="Photos prises",
            value=photo_count,
            delta=None
        )
    
    with col2:
        gesture_emoji = GESTURE_EMOJIS.get(current_gesture, "😐")
        st.metric(
            label="Geste actuel",
            value=f"{gesture_emoji} {current_gesture}"
        )
    
    with col3:
        st.metric(
            label="Cooldown",
            value=f"{cooldown_remaining:.1f}s",
            delta=f"-{2.0-cooldown_remaining:.1f}s" if cooldown_remaining < 2.0 else None
        )
    
    with col4:
        st.metric(
            label="Seuil confiance",
            value=f"{confidence_threshold:.0%}"
        )

def render_gestures_status(all_predictions, priority_gestures):
    """Affiche le statut de tous les gestes disponibles"""
    st.subheader("🎭 Gestes actuellement disponibles :")
    
    if priority_gestures:
        # Créer une chaîne avec les gestes et leurs émojis
        gestures_display = []
        for gesture in priority_gestures:
            emoji = GESTURE_EMOJIS.get(gesture, "🕺")
            gestures_display.append(f"{emoji} {gesture}")
        
        gesture_text = ", ".join(gestures_display)
        st.write(gesture_text)
    else:
        st.write("Aucun geste sélectionné")

def render_camera_feed_header():
    """En-tête pour la section caméra"""
    st.header("🎥 Caméra Live")
    
    # Instructions
    st.info("""
    💡 **Instructions :**
    - Placez-vous devant la caméra
    - Effectuez vos gestes de danse
    - Les photos seront prises automatiquement lors de la détection
    """)

def render_recent_photos(save_dir, max_photos=6):
    """Affichage des photos récentes"""
    if not os.path.exists(save_dir):
        st.warning("📁 Dossier d'images non trouvé")
        return
        
    image_files = [
        f for f in os.listdir(save_dir) 
        if f.endswith('.jpg') and (f.startswith('tm_') or f.startswith('api_'))
    ]
    image_files.sort(reverse=True)  # Plus récentes en premier
    
    if not image_files:
        st.info("📸 Aucune photo prise pour le moment")
        return
    
    st.header("📸 Photos récentes")
    
    # Afficher en grille
    cols = st.columns(3)
    for i, img_file in enumerate(image_files[:max_photos]):
        with cols[i % 3]:
            img_path = os.path.join(save_dir, img_file)
            
            # Extraire les infos du nom de fichier
            parts = img_file.replace('.jpg', '').split('_')
            if len(parts) >= 3:
                model_type = parts[0]  # tm ou api
                gesture = parts[1]
                confidence = parts[2]
                emoji = GESTURE_EMOJIS.get(gesture, "🕺")
                caption = f"{emoji} {gesture} ({confidence}) [{model_type.upper()}]"
            else:
                caption = img_file
                
            st.image(img_path, caption=caption, use_column_width=True)

def show_toast_notification(message, icon="📸"):
    """Affichage de notification toast"""
    st.toast(f"{icon} {message}", icon="📸")

def render_debug_info(all_predictions, current_gesture, gesture_detected, model_manager=None):
    """Affichage des informations de debug"""
    with st.expander("🔍 Debug Info", expanded=False):
        
        # Informations du modèle
        if model_manager:
            model_info = model_manager.get_model_info()
            st.json({
                "model_type": model_info["type"],
                "model_status": model_info["status"],
                "authenticated": model_info.get("authenticated", "N/A"),
                "current_gesture": current_gesture,
                "gesture_detected": gesture_detected,
                "model_ready": model_manager.is_model_ready()
            })
        
        # Prédictions
        if all_predictions:
            st.subheader("📊 Toutes les prédictions")
            st.json(all_predictions)

def render_model_status_banner(model_manager):
    """Affiche un bandeau avec le statut du modèle actuel"""
    model_info = model_manager.get_model_info()
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        model_type_display = {
            "teachable_machine": "🤖 Teachable Machine",
            "api": "🔗 API YOLOv8"
        }
        st.write(f"**Modèle actuel:** {model_type_display.get(model_info['type'], 'Inconnu')}")
    
    with col2:
        if model_info["status"] == "active":
            if model_info["type"] == "api":
                if model_info.get("authenticated", False):
                    st.success("✅ Connecté et prêt")
                else:
                    st.error("❌ Authentification requise")
            else:
                st.success("✅ Modèle chargé")
        else:
            st.error("❌ Modèle non initialisé")
    
    with col3:
        is_ready = model_manager.is_model_ready()
        if is_ready:
            st.write("🟢 **PRÊT**")
        else:
            st.write("🔴 **NON PRÊT**")

def render_gesture_comparison(tm_predictions=None, api_predictions=None):
    """Compare les prédictions des deux modèles (si disponibles)"""
    if tm_predictions and api_predictions:
        with st.expander("🔄 Comparaison des modèles", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Teachable Machine")
                tm_sorted = sorted(tm_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
                for gesture, conf in tm_sorted:
                    st.write(f"- {gesture}: {conf:.2%}")
            
            with col2:
                st.subheader("API YOLOv8")
                api_sorted = sorted(api_predictions.items(), key=lambda x: x[1], reverse=True)[:3]
                for gesture, conf in api_sorted:
                    st.write(f"- {gesture}: {conf:.2%}")

def render_api_specific_info(model_manager):
    """Affiche des informations spécifiques à l'API dans la sidebar"""
    if (model_manager.current_model_type.value == "api" and 
        model_manager.api_adapter and 
        model_manager.api_adapter.is_authenticated()):
        
        st.sidebar.subheader("📊 Informations API")
        
        # Test de latence
        if st.sidebar.button("🏃 Tester la latence"):
            import time
            import numpy as np
            
            # Créer une image test
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            start_time = time.time()
            try:
                detected, result = model_manager.api_adapter.detect_gesture_api(test_frame)
                latency = (time.time() - start_time) * 1000
                st.sidebar.success(f"⚡ Latence: {latency:.0f}ms")
            except Exception as e:
                st.sidebar.error(f"❌ Erreur: {str(e)}")

def show_model_switch_warning(old_model, new_model):
    """Affiche un avertissement lors du changement de modèle"""
    if old_model != new_model:
        model_names = {
            "teachable_machine": "Teachable Machine",
            "api": "API YOLOv8"
        }
        
        st.warning(f"""
        🔄 **Changement de modèle détecté**
        
        Passage de **{model_names.get(old_model, old_model)}** vers **{model_names.get(new_model, new_model)}**
        
        Les gestes disponibles peuvent être différents.
        """)

def render_performance_metrics(frame_count, start_time):
    """Affiche les métriques de performance"""
    import time
    
    if frame_count > 0 and frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = 30 / elapsed
        
        # Afficher dans la sidebar
        st.sidebar.metric(
            label="🎥 FPS",
            value=f"{fps:.1f}",
            delta=f"{fps-25:.1f}" if fps > 25 else None
        )