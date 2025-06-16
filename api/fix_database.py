#!/usr/bin/env python3
"""
Script pour réparer et initialiser correctement la base de données
"""

import os
import sys
import sqlite3
from pathlib import Path

# Détecter le répertoire API (le script peut être lancé depuis api/ ou depuis la racine)
current_dir = Path(__file__).parent
if current_dir.name == "api":
    # Script lancé depuis le dossier api/
    api_dir = current_dir
else:
    # Script lancé depuis la racine du projet
    api_dir = current_dir / "api"

sys.path.insert(0, str(api_dir))

def check_and_fix_database():
    """Vérifie et répare la base de données"""
    print("🔧 Vérification et réparation de la base de données...")
    
    db_path = api_dir / "dance_app.db"
    
    try:
        # Importer les modules SQLAlchemy
        from database import engine, Base
        from models import User, PredictionLog
        
        print(f"📁 Chemin de la base: {db_path}")
        
        # Supprimer l'ancienne base si elle existe et est corrompue
        if db_path.exists():
            try:
                # Tester la connexion
                with sqlite3.connect(str(db_path)) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"📋 Tables existantes: {[table[0] for table in tables]}")
                    
                    # Vérifier la structure des tables
                    for table_name in ['users', 'prediction_logs']:
                        try:
                            cursor.execute(f"PRAGMA table_info({table_name});")
                            columns = cursor.fetchall()
                            print(f"   {table_name}: {len(columns)} colonnes")
                        except sqlite3.Error as e:
                            print(f"❌ Erreur table {table_name}: {e}")
                            # Supprimer et recréer
                            if db_path.exists():
                                os.remove(str(db_path))
                                print("🗑️ Base corrompue supprimée")
                            break
                            
            except sqlite3.Error as e:
                print(f"❌ Base corrompue: {e}")
                if db_path.exists():
                    os.remove(str(db_path))
                    print("🗑️ Base corrompue supprimée")
        
        # Créer/recréer toutes les tables
        print("🔨 Création des tables...")
        Base.metadata.create_all(bind=engine)
        
        # Vérifier que tout fonctionne
        from sqlalchemy import text
        with engine.connect() as conn:
            # Test des tables
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM prediction_logs"))
            log_count = result.scalar()
            
            print(f"✅ Base de données opérationnelle:")
            print(f"   - {user_count} utilisateurs")
            print(f"   - {log_count} logs de prédiction")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la réparation: {e}")
        return False

def create_test_user():
    """Crée un utilisateur de test"""
    print("\n👤 Création d'un utilisateur de test...")
    
    try:
        from sqlalchemy.orm import sessionmaker
        from database import engine
        from models import User
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = SessionLocal()
        
        # Vérifier si l'utilisateur test existe déjà
        existing_user = db.query(User).filter(User.username == "testuser").first()
        
        if existing_user:
            print("✅ Utilisateur de test existe déjà")
            db.close()
            return True
        
        # Créer l'utilisateur de test
        hashed_password = pwd_context.hash("testpass123")
        test_user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=hashed_password
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print(f"✅ Utilisateur de test créé: {test_user.username} (ID: {test_user.id})")
        db.close()
        return True
        
    except Exception as e:
        print(f"❌ Erreur création utilisateur test: {e}")
        return False

def check_required_files():
    """Vérifie que tous les fichiers requis existent"""
    print("\n📁 Vérification des fichiers requis...")
    
    required_files = [
        "database.py",
        "models.py",
        "config.py",
        "authentication.py",
        "main.py",
        "routes/log.py",
        "routes/inference.py",
        "schemas.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = api_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MANQUANT")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Fichiers manquants: {missing_files}")
        return False
    
    print("✅ Tous les fichiers requis sont présents")
    return True

def check_yolo_model():
    """Vérifie que le modèle YOLO est présent"""
    print("\n🤖 Vérification du modèle YOLO...")
    
    model_path = api_dir / "yolov8n-pose.pt"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Modèle YOLO trouvé: {size_mb:.1f} MB")
        return True
    else:
        print(f"❌ Modèle YOLO manquant: {model_path}")
        print("💡 Téléchargez le modèle avec: python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt')\"")
        return False

def main():
    """Fonction principale"""
    print("🔧 === RÉPARATION DE L'API DANCE DETECTION ===\n")
    
    # Vérifier que nous sommes dans le bon répertoire
    if not api_dir.exists():
        print(f"❌ Répertoire API non trouvé: {api_dir}")
        print("💡 Exécutez ce script depuis la racine du projet")
        return False
    
    # Étapes de réparation
    steps = [
        ("Fichiers requis", check_required_files),
        ("Base de données", check_and_fix_database),
        ("Utilisateur de test", create_test_user),
        ("Modèle YOLO", check_yolo_model)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*50}")
        print(f"ÉTAPE: {step_name}")
        print('='*50)
        
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ Erreur dans l'étape {step_name}: {e}")
            results[step_name] = False
    
    # Résumé
    print("\n" + "="*50)
    print("📊 RÉSUMÉ DE LA RÉPARATION:")
    print("="*50)
    
    all_success = True
    for step_name, success in results.items():
        status = "✅ OK" if success else "❌ ÉCHEC"
        print(f"{step_name.ljust(20)}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n🎉 Réparation terminée avec succès !")
        print("💡 Vous pouvez maintenant lancer l'API avec:")
        print("   cd api && python main.py")
    else:
        print("\n⚠️ Certaines étapes ont échoué.")
        print("💡 Vérifiez les erreurs ci-dessus et réessayez.")
    
    return all_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)