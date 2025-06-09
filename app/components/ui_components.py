"""
Composants UI réutilisables pour l'application YOLO Dance
"""
import streamlit as st
import os
from config.settings import GESTURE_EMOJIS, ALL_GESTURES, DEFAULT_PRIORITY_GESTURES

def render_sidebar_settings():
    """Rendu de la sidebar avec tous les paramètres"""
    st.sidebar.header("⚙️ Paramètres")
    
    # Seuil de confiance
    confidence_threshold = st.sidebar.slider(
        "Seuil de confiance", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
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
    
    # Gestes prioritaires
    st.sidebar.subheader("🎯 Gestes à capturer")
    priority_gestures = st.sidebar.multiselect(
        "Sélectionnez les gestes prioritaires",
        ALL_GESTURES,
        default=DEFAULT_PRIORITY_GESTURES,
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
    st.header("🎥Caméra Live")
    
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
        if f.endswith('.jpg') and f.startswith('tm_')
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
                gesture = parts[1]
                confidence = parts[2]
                emoji = GESTURE_EMOJIS.get(gesture, "🕺")
                caption = f"{emoji} {gesture} ({confidence})"
            else:
                caption = img_file
                
            st.image(img_path, caption=caption, use_column_width=True)

def show_toast_notification(message, icon="📸"):
    """Affichage de notification toast"""
    st.toast(f"{icon} {message}", icon="📸")

def render_debug_info(all_predictions, current_gesture, gesture_detected):
    """Affichage des informations de debug"""
    with st.expander("🔍 Debug Info", expanded=False):
        st.json({
            "current_gesture": current_gesture,
            "gesture_detected": gesture_detected,
            "all_predictions": all_predictions
        })