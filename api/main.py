# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from routes import inference, log
import random
=======
from fastapi.responses import JSONResponse
>>>>>>> 3c385f1dd1dfb5ebf6ae0201381c6982a3550f38
import uvicorn
import os
import sys

<<<<<<< HEAD



app = FastAPI(title="YOLO_Dance")
=======
# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
>>>>>>> 3c385f1dd1dfb5ebf6ae0201381c6982a3550f38

# Initialisation de la base de données au démarrage
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire d'événements de cycle de vie de l'application"""
    # Événements de démarrage
    print("🚀 Démarrage de l'API Dance Detection...")
    
    # Vérifier que la base de données existe
    try:
        from database import engine, Base
        from models import User, PredictionLog
        
        # Créer les tables si elles n'existent pas
        Base.metadata.create_all(bind=engine)
        print("✅ Base de données initialisée")
        
        # Vérifier la connexion
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            print(f"📊 Nombre d'utilisateurs: {user_count}")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de la DB: {e}")
        raise
    
    # Vérifier que le modèle YOLO est disponible
    try:
        from pose_detection_model import PoseDetector
        detector = PoseDetector(model_path="yolov8n-pose.pt")
        print("✅ Modèle YOLO initialisé")
    except Exception as e:
        print(f"⚠️ Attention: Modèle YOLO non disponible: {e}")
    
    yield  # L'application démarre ici
    
    # Événements d'arrêt
    print("🛑 Arrêt de l'API Dance Detection...")

app = FastAPI(
    title="Dance Detection API",
    description="API pour la reconnaissance de mouvements de danse avec authentification JWT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS - Configuration pour permettre les requêtes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Ajouter les routes
app.include_router(inference.router, prefix="/api")
app.include_router(log.router, prefix="/log")

# Endpoint racine
@app.get("/")
async def root():
    return {
        "message": "Dance Detection API is running!",
        "status": "ok",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """Endpoint de vérification de santé global"""
    try:
        # Tester la connexion à la base de données
        from database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "api_version": "1.0.0"
    }

# Inclure les routes
print("🔄 Chargement des routes...")

try:
    # Routes d'authentification
    from routes.log import router as auth_router
    app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
    print("✅ Routes d'authentification ajoutées (/auth/register, /auth/token, /auth/users, /auth/me)")
except ImportError as e:
    print(f"❌ Erreur import routes/log.py: {e}")
    raise

try:
    # Routes d'inférence
    from routes.inference import router as inference_router
    app.include_router(inference_router, prefix="/api", tags=["Inference"])
    print("✅ Routes d'inférence ajoutées (/api/detect-gesture, /api/gestures, /api/logs)")
except ImportError as e:
    print(f"❌ Erreur import routes/inference.py: {e}")
    raise

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global pour éviter les crashes"""
    print(f"❌ Erreur non gérée: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erreur interne du serveur",
            "type": type(exc).__name__,
            "message": str(exc) if app.debug else "Une erreur est survenue"
        }
    )

# Debug endpoint pour lister toutes les routes
@app.get("/debug/routes")
async def debug_routes():
    """Endpoint de debug pour lister toutes les routes disponibles"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unknown'),
                "tags": getattr(route, 'tags', [])
            })
    
    return {
        "total_routes": len(routes),
        "routes": sorted(routes, key=lambda x: x['path'])
    }

# Supprimer les anciens événements dépréciés
# @app.on_event("startup") et @app.on_event("shutdown") sont maintenant dans lifespan

if __name__ == "__main__":
    print("🎯 Lancement du serveur FastAPI...")
    print("📚 Documentation disponible sur: http://localhost:8000/docs")
    print("🔍 API Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "main:app",  # Utiliser le format string pour éviter le warning
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )