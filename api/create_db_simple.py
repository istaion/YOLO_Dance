import os
import sqlite3
from database import engine, Base
from models import User, PredictionLog

def create_tables():
    """Crée toutes les tables nécessaires"""
    print("🔄 Création de la base de données...")
    
    # Supprimer l'ancienne base si elle existe
    if os.path.exists("dance_app.db"):
        os.remove("dance_app.db")
        print("🗑️ Ancienne base supprimée")
    
    # Créer toutes les tables avec SQLAlchemy
    Base.metadata.create_all(bind=engine)
    
    print("✅ Base de données créée avec succès!")
    
    # Vérifier que les tables existent
    conn = sqlite3.connect("dance_app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"📋 Tables créées: {[table[0] for table in tables]}")
    conn.close()

if __name__ == "__main__":
    create_tables()