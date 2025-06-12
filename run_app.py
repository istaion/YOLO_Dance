import os
import sys
import subprocess
from pathlib import Path

def check_structure():
    """Vérifie la structure du projet"""
    base_dir = Path(__file__).parent
    required_dirs = [
        base_dir / "app",
        base_dir / "scripts" / "yolo_pipeline", 
        base_dir / "models",
        base_dir / "images"
    ]
    
    required_files = [
        base_dir / "app" / "main_app.py",
        base_dir / "scripts" / "yolo_pipeline" / "yolo_model.py"
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    missing_files = [f for f in required_files if not f.exists()]
    
    if missing_dirs:
        print(f"❌ Répertoires manquants: {[str(d) for d in missing_dirs]}")
        for d in missing_dirs:
            d.mkdir(parents=True, exist_ok=True)
        print("✅ Répertoires créés")
    
    if missing_files:
        print(f"❌ Fichiers manquants: {[str(f) for f in missing_files]}")
        return False
    
    print("✅ Structure du projet correcte")
    return True

def main():
    """Lance l'application Streamlit"""
    print("🚀 Lancement de YOLO Dance & Teachable Machine")
    print("=" * 50)
    
    # Vérifications
    if not check_structure():
        sys.exit(1)
    
    # Lancer Streamlit
    app_path = Path(__file__).parent / "app" / "main_app.py"
    
    print(f"🎬 Démarrage de Streamlit...")
    print(f"📂 Application: {app_path}")
    print("🌐 L'application s'ouvrira dans votre navigateur")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application fermée")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    main()