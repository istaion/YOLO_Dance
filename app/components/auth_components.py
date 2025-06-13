# app/components/auth_components.py

import streamlit as st
from typing import Tuple, Optional
import re

def validate_email(email: str) -> bool:
    """Valide le format de l'email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Tuple[bool, str]:
    """
    Valide la force du mot de passe
    Returns: (is_valid, message)
    """
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères"
    
    if not any(c.isalpha() for c in password):
        return False, "Le mot de passe doit contenir au moins une lettre"
    
    if not any(c.isdigit() for c in password):
        return False, "Le mot de passe doit contenir au moins un chiffre"
    
    return True, "Mot de passe valide"

def render_login_form(model_manager) -> bool:
    """
    Affiche le formulaire de connexion
    Returns: True si connexion réussie
    """
    st.subheader("🔐 Connexion API")
    
    with st.form("login_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            username = st.text_input("👤 Nom d'utilisateur", key="login_username")
            password = st.text_input("🔒 Mot de passe", type="password", key="login_password")
        
        with col2:
            st.write("")  # Espacement
            submit_button = st.form_submit_button("Se connecter", type="primary")
            show_register = st.form_submit_button("S'inscrire", type="secondary")
    
    if submit_button:
        if not username or not password:
            st.error("❌ Veuillez remplir tous les champs")
            return False
        
        success, message = model_manager.authenticate_api(username, password)
        
        if success:
            st.success(message)
            st.session_state['api_authenticated'] = True
            st.session_state['api_username'] = username
            st.rerun()
            return True
        else:
            st.error(message)
            return False
    
    if show_register:
        st.session_state['show_register'] = True
        st.rerun()
    
    return False

def render_register_form(model_manager) -> bool:
    """
    Affiche le formulaire d'inscription
    Returns: True si inscription réussie
    """
    st.subheader("📝 Inscription API")
    
    with st.form("register_form"):
        username = st.text_input("👤 Nom d'utilisateur", key="register_username")
        email = st.text_input("📧 Email", key="register_email")
        password = st.text_input("🔒 Mot de passe", type="password", key="register_password")
        password_confirm = st.text_input("🔒 Confirmer le mot de passe", type="password", key="register_password_confirm")
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("S'inscrire", type="primary")
        with col2:
            back_button = st.form_submit_button("Retour à la connexion")
    
    if submit_button:
        # Validation des champs
        if not all([username, email, password, password_confirm]):
            st.error("❌ Veuillez remplir tous les champs")
            return False
        
        if password != password_confirm:
            st.error("❌ Les mots de passe ne correspondent pas")
            return False
        
        if not validate_email(email):
            st.error("❌ Format d'email invalide")
            return False
        
        is_valid_password, password_message = validate_password(password)
        if not is_valid_password:
            st.error(f"❌ {password_message}")
            return False
        
        # Tentative d'inscription
        success, message = model_manager.register_api_user(username, email, password)
        
        if success:
            st.success(message)
            st.session_state['show_register'] = False
            st.rerun()
            return True
        else:
            st.error(message)
            return False
    
    if back_button:
        st.session_state['show_register'] = False
        st.rerun()
    
    return False

def render_auth_status(model_manager) -> None:
    """Affiche le statut d'authentification dans la sidebar"""
    model_info = model_manager.get_model_info()
    
    if model_info["type"] == "api":
        st.sidebar.subheader("🔐 Statut API")
        
        if model_info["authenticated"]:
            st.sidebar.success("✅ Connecté")
            if 'api_username' in st.session_state:
                st.sidebar.write(f"👤 **{st.session_state['api_username']}**")
            
            if st.sidebar.button("🚪 Se déconnecter"):
                model_manager.logout_api()
                st.session_state['api_authenticated'] = False
                if 'api_username' in st.session_state:
                    del st.session_state['api_username']
                st.rerun()
        else:
            st.sidebar.error("❌ Non connecté")
            st.sidebar.write("Connexion requise pour utiliser l'API")

def handle_api_authentication(model_manager) -> bool:
    """
    Gère l'authentification API dans l'interface principale
    Returns: True si l'utilisateur est authentifié
    """
    # Vérifier l'état de la session
    if st.session_state.get('api_authenticated', False):
        return True
    
    # Afficher le formulaire approprié
    if st.session_state.get('show_register', False):
        success = render_register_form(model_manager)
        return success
    else:
        success = render_login_form(model_manager)
        return success

def render_model_selector(model_manager) -> str:
    """
    Affiche le sélecteur de modèle dans la sidebar
    Returns: Le type de modèle sélectionné
    """
    st.sidebar.subheader("🤖 Sélection du modèle")
    
    model_options = {
        "Teachable Machine": "teachable_machine",
        "API (YOLOv8)": "api"
    }
    
    current_model = model_manager.current_model_type.value
    current_index = list(model_options.values()).index(current_model)
    
    selected_model_name = st.sidebar.selectbox(
        "Choisir le modèle :",
        options=list(model_options.keys()),
        index=current_index,
        help="Sélectionnez le modèle à utiliser pour la détection"
    )
    
    selected_model_type = model_options[selected_model_name]
    
    # Afficher des informations sur le modèle sélectionné
    if selected_model_type == "teachable_machine":
        st.sidebar.info("""
        **Teachable Machine :**
        - Modèle personnalisé pré-entraîné
        - Détection locale (rapide)
        - 15 gestes disponibles
        """)
    elif selected_model_type == "api":
        st.sidebar.info("""
        **API YOLOv8 :**
        - Modèle YOLOv8 pose detection
        - Traitement serveur
        - Authentification requise
        - 5 gestes principaux
        """)
        
        # Afficher le statut de l'API
        if model_manager.api_adapter:
            is_healthy, health_message = model_manager.api_adapter.check_api_health()
            if is_healthy:
                st.sidebar.success("🟢 API accessible")
            else:
                st.sidebar.error(f"🔴 {health_message}")
    
    return selected_model_type

def render_api_connection_settings() -> str:
    """
    Affiche les paramètres de connexion API dans la sidebar
    """
    with st.sidebar.expander("⚙️ Configuration API"):
        api_url = st.text_input(
            "URL de l'API",
            value="http://localhost:8000",
            help="URL de base de votre API FastAPI"
        )
        
        if st.button("Tester la connexion"):
            try:
                import requests
                response = requests.get(f"{api_url.rstrip('/')}/detect/", timeout=5)
                if response.status_code in [200, 405]:
                    st.success("✅ API accessible")
                else:
                    st.error(f"❌ Code de réponse: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")
    
    return api_url

def init_session_state():
    """Initialise les variables de session pour l'authentification"""
    if 'api_authenticated' not in st.session_state:
        st.session_state['api_authenticated'] = False
    
    if 'show_register' not in st.session_state:
        st.session_state['show_register'] = False
    
    if 'selected_model_type' not in st.session_state:
        st.session_state['selected_model_type'] = 'teachable_machine'