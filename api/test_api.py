#!/usr/bin/env python3
"""
Test simple de l'API Dance Detection
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test simple du health check"""
    print("🏥 Test Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API accessible")
            print(f"   Réponse: {response.json()}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connexion impossible: {e}")
        return False

def test_register_and_login():
    """Test inscription + connexion"""
    print("\n👤 Test Inscription + Connexion...")
    
    # Données utilisateur test
    timestamp = int(time.time())
    user_data = {
        "username": f"test_{timestamp}",
        "email": f"test_{timestamp}@example.com", 
        "password": "test123"
    }
    
    # 1. Inscription
    try:
        response = requests.post(f"{API_BASE_URL}/auth/register", json=user_data, timeout=10)
        if response.status_code == 200:
            print("✅ Inscription réussie")
        else:
            print(f"❌ Inscription échouée: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur inscription: {e}")
        return None
    
    # 2. Connexion
    try:
        login_data = {"username": user_data["username"], "password": user_data["password"]}
        response = requests.post(f"{API_BASE_URL}/auth/token", json=login_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            token = data["access_token"]
            print("✅ Connexion réussie")
            print(f"   Token: {token[:20]}...")
            return token
        else:
            print(f"❌ Connexion échouée: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return None

def test_protected_endpoint(token):
    """Test endpoint protégé"""
    print("\n🔒 Test Endpoint Protégé...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{API_BASE_URL}/auth/me", headers=headers, timeout=10)
        if response.status_code == 200:
            user_info = response.json()
            print("✅ Accès autorisé")
            print(f"   Utilisateur: {user_info['username']}")
            return True
        else:
            print(f"❌ Accès refusé: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_gestures(token):
    """Test endpoint des gestes"""
    print("\n🎭 Test Gestes Disponibles...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/gestures", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Gestes récupérés")
            print(f"   Nombre: {len(data['available_gestures'])}")
            for gesture in data['available_gestures']:
                print(f"   {gesture['emoji']} {gesture['name']}")
            return True
        else:
            print(f"❌ Erreur: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Test principal"""
    print("🧪 === TEST SIMPLE API DANCE DETECTION ===\n")
    
    # 1. Test santé
    if not test_health():
        print("\n❌ API non accessible - Vérifiez qu'elle tourne sur localhost:8000")
        return False
    
    # 2. Test authentification  
    token = test_register_and_login()
    if not token:
        print("\n❌ Authentification échouée")
        return False
    
    # 3. Test endpoint protégé
    if not test_protected_endpoint(token):
        print("\n❌ Problème avec les endpoints protégés")
        return False
    
    # 4. Test gestes
    if not test_gestures(token):
        print("\n❌ Problème avec l'endpoint des gestes")
        return False
    
    print("\n🎉 TOUS LES TESTS RÉUSSIS ! L'API fonctionne correctement.")
    print("\n💡 Vous pouvez maintenant:")
    print("   - Aller sur http://localhost:8000/docs pour la documentation")
    print("   - Utiliser l'app Streamlit avec cette API")
    print(f"   - Utiliser le token: {token[:30]}...")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)