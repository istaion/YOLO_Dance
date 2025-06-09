# setup_teachable_machine.py
import os
import shutil
import zipfile
import json
import sys
import subprocess

def install_requirements():
    """Installe les dépendances nécessaires"""
    requirements = [
        "tensorflow>=2.8.0",
        "tensorflowjs>=3.18.0", 
        "tensorflow-hub>=0.12.0",
        "mediapipe>=0.8.10",
        "opencv-python>=4.5.0",
        "streamlit>=1.20.0",
        "numpy>=1.21.0"
    ]
    
    print("🚀 Installation des dépendances...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ {req} installé avec succès")
        except subprocess.CalledProcessError:
            print(f"❌ Erreur lors de l'installation de {req}")
            return False
    
    return True

def setup_model_directory(project_root, zip_path=None):
    """Configure le répertoire du modèle"""
    model_dir = os.path.join(project_root, "model", "teachable_machine")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📁 Répertoire créé: {model_dir}")
    
    if zip_path and os.path.exists(zip_path):
        print(f"📦 Extraction du modèle depuis {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            print("✅ Modèle extrait avec succès")
            
            # Vérifier les fichiers nécessaires
            required_files = ["model.json", "metadata.json", "weights.bin"]
            for file in required_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    print(f"✅ {file} trouvé")
                else:
                    print(f"❌ {file} manquant")
                    return False
            
            # Afficher les informations du modèle
            display_model_info(model_dir)
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {e}")
            return False
    else:
        print("⚠️ Aucun fichier ZIP fourni. Vous devrez copier manuellement les fichiers.")
        print(f"Copiez model.json, metadata.json et weights.bin dans: {model_dir}")
        return True

def display_model_info(model_dir):
    """Affiche les informations du modèle"""
    try:
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("\n📊 Informations du modèle:")
        print(f"   Nom: {metadata.get('modelName', 'N/A')}")
        print(f"   Version TM: {metadata.get('tmVersion', 'N/A')}")
        print(f"   Labels: {', '.join(metadata.get('labels', []))}")
        print(f"   Nombre de classes: {len(metadata.get('labels', []))}")
        
    except Exception as e:
        print(f"⚠️ Impossible de lire les métadonnées: {e}")

def create_launch_script(project_root):
    """Crée un script de lancement simple"""
    launch_script = os.path.join(project_root, "launch_tm_app.py")
    
    script_content = f'''#!/usr/bin/env python3
"""
Script de lancement pour l'application Teachable Machine
Généré automatiquement par setup_teachable_machine.py
"""

import streamlit as st
import sys
import os

# Ajouter le répertoire du projet au PYTHONPATH
project_root = "{project_root}"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Lancer l'application
if __name__ == "__main__":
    import subprocess
    app_path = os.path.join(project_root, "app", "streamlit_app_teachable_machine.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
'''
    
    with open(launch_script, 'w') as f:
        f.write(script_content)
    
    print(f"🚀 Script de lancement créé: {launch_script}")
    return launch_script

def main():
    print("🎯 Configuration de l'intégration Teachable Machine")
    print("=" * 50)
    
    # Détecter le répertoire du projet
    current_dir = os.getcwd()
    print(f"📍 Répertoire actuel: {current_dir}")
    
    # Demander le chemin du ZIP (optionnel)
    zip_path = input("\n📦 Chemin vers votre fichier my-pose-model.zip (optionnel): ").strip()
    if not zip_path:
        zip_path = None
    elif not os.path.exists(zip_path):
        print(f"❌ Fichier non trouvé: {zip_path}")
        zip_path = None
    
    # Installation des dépendances
    if input("\n🔧 Installer les dépendances Python ? (y/N): ").lower() == 'y':
        if not install_requirements():
            print("❌ Échec de l'installation des dépendances")
            return
    
    # Configuration du modèle
    print("\n📁 Configuration du répertoire du modèle...")
    if not setup_model_directory(current_dir, zip_path):
        print("❌ Échec de la configuration du modèle")
        return
    
    # Copier les fichiers de l'adaptateur
    app_dir = os.path.join(current_dir, "app")
    os.makedirs(app_dir, exist_ok=True)
    
    print(f"📋 Les fichiers suivants doivent être copiés dans {app_dir}:")
    print("   - teachable_machine_adapter.py")
    print("   - streamlit_app_teachable_machine.py")
    
    # Créer le script de lancement
    launch_script = create_launch_script(current_dir)
    
    print("\n✅ Configuration terminée!")
    print("\n📋 Prochaines étapes:")
    print("1. Copiez teachable_machine_adapter.py dans le dossier app/")
    print("2. Copiez streamlit_app_teachable_machine.py dans le dossier app/")
    print("3. Si vous n'avez pas fourni le ZIP, copiez manuellement les fichiers du modèle")
    print(f"4. Lancez l'application avec: python {launch_script}")
    print("   Ou directement: streamlit run app/streamlit_app_teachable_machine.py")
    
    print("\n🎨 Structure finale attendue:")
    print("""
    votre_projet/
    ├── app/
    │   ├── teachable_machine_adapter.py
    │   ├── streamlit_app_teachable_machine.py
    │   └── (vos autres fichiers app)
    ├── model/
    │   └── teachable_machine/
    │       ├── model.json
    │       ├── metadata.json
    │       └── weights.bin
    ├── images/
    └── launch_tm_app.py
    """)

if __name__ == "__main__":
    main()