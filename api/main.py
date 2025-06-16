# api/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import inference, log
from model_loader import YoloDanceSystemWeighted
from utils import GestureFilterSystem
import uvicorn
import numpy as np
from typing import Dict, Optional
import cv2
import base64
import io
from PIL import Image
import uvicorn
import os
import sys

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialisation de la base de données au démarrage
from contextlib import asynccontextmanager

# Ajouter le répertoire courant au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Essayer d'importer les modules selon la structure disponible
try:
    # Si vous avez les modules dans api/
    from model_loader import YoloDanceSystemWeighted
    from utils import GestureFilterSystem
except ImportError:
    try:
        # Si les modules sont dans le dossier parent
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from scripts.yolo_pipeline.yolo_model_weighted import YoloDanceSystemWeighted
        from utils import GestureFilterSystem
    except ImportError:
        print("⚠️ Impossible d'importer les modèles YOLO. Fonctionnement en mode dégradé.")
        YoloDanceSystemWeighted = None
        GestureFilterSystem = None

# Variables globales pour les modèles (chargés une seule fois)
yolo_system_weighted = None
filter_system = None

# Configuration des modèles
MODEL_CONFIGS = {
    "weighted": {
        "path": "../models/final_weighted_model.pth", 
        "description": "Modèle YOLO pondéré (pose prioritaire)"
    }
}

# Seuils par défaut
DEFAULT_THRESHOLDS = {
    'hands_up': 0.7,
    'dab': 0.7,
    'twerk': 0.6,
    'jul': 0.6,
    'crossarm': 0.6,
    'neutral': 0.4
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire d'événements de cycle de vie de l'application"""
    # Événements de démarrage
    print("🚀 Démarrage de l'API YOLO Dance...")
    
    global yolo_system_weighted, filter_system
    
    # Initialiser les modèles
    if YoloDanceSystemWeighted:
        try:
            if os.path.exists(MODEL_CONFIGS["weighted"]["path"]):
                yolo_system_weighted = YoloDanceSystemWeighted(
                    custom_classifier_path=MODEL_CONFIGS["weighted"]["path"]
                )
                print("✅ Modèle pondéré chargé")
            else:
                print("⚠️ Modèle pondéré non trouvé, utilisation modèle non-entraîné")
                yolo_system_weighted = YoloDanceSystemWeighted()
        except Exception as e:
            print(f"❌ Erreur chargement modèle pondéré: {e}")
            yolo_system_weighted = None
    
    # Charger le système de filtres
    if GestureFilterSystem:
        try:
            filter_system = GestureFilterSystem()
            print("✅ Système de filtres chargé")
        except Exception as e:
            print(f"❌ Erreur chargement filtres: {e}")
            filter_system = None
    
    # Essayer d'initialiser la base de données si elle existe
    try:
        from database import engine, Base
        Base.metadata.create_all(bind=engine)
        print("✅ Base de données initialisée")
    except ImportError:
        print("ℹ️ Pas de base de données configurée")
    except Exception as e:
        print(f"⚠️ Erreur base de données: {e}")
    
    print("🎉 Initialisation terminée!")
    
    yield  # L'application démarre ici
    
    # Événements d'arrêt
    print("🛑 Arrêt de l'API YOLO Dance...")

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

# Variables globales pour les modèles (chargés une seule fois)
yolo_system_standard = None
yolo_system_weighted = None
filter_system = None

# Configuration des modèles
MODEL_CONFIGS = {
    "standard": {
        "path": "../models/best_model.pth",
        "description": "Modèle YOLO standard équilibré"
    },
    "weighted": {
        "path": "../models/final_weighted_model.pth", 
        "description": "Modèle YOLO pondéré (pose prioritaire)"
    }
}

# Seuils par défaut
DEFAULT_THRESHOLDS = {
    'hands_up': 0.7,
    'dab': 0.7,
    'twerk': 0.6,
    'jul': 0.6,
    'crossarm': 0.6,
    'neutral': 0.4
}

@app.on_event("startup")
async def startup_event():
    """Initialise les modèles au démarrage de l'API"""
    global yolo_system_standard, yolo_system_weighted, filter_system
    
    print("🚀 Initialisation des modèles YOLO Dance...")
    
    # Charger le modèle pondéré
    try:
        if os.path.exists(MODEL_CONFIGS["weighted"]["path"]):
            yolo_system_weighted = YoloDanceSystemWeighted(
                custom_classifier_path=MODEL_CONFIGS["weighted"]["path"]
            )
            print("✅ Modèle pondéré chargé")
        else:
            print("⚠️ Modèle pondéré non trouvé, utilisation modèle non-entraîné")
            yolo_system_weighted = YoloDanceSystemWeighted()
    except Exception as e:
        print(f"❌ Erreur chargement modèle pondéré: {e}")
        yolo_system_weighted = None
    
    # Charger le système de filtres
    try:
        filter_system = GestureFilterSystem()
        print("✅ Système de filtres chargé")
    except Exception as e:
        print(f"❌ Erreur chargement filtres: {e}")
        filter_system = None
    
    print("🎉 Initialisation terminée!")

def get_model(model_type: str = "weighted"):
    """Récupère le modèle approprié"""
    if model_type == "weighted" and yolo_system_weighted is not None:
        return yolo_system_weighted
    elif model_type == "standard" and yolo_system_standard is not None:
        return yolo_system_standard
    elif yolo_system_standard is not None:
        return yolo_system_standard
    else:
        raise HTTPException(status_code=503, detail="Aucun modèle disponible")

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convertit une image OpenCV en base64"""
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

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


@app.get("/models/")
async def list_models():
    """Liste les modèles disponibles"""
    models = {}
    
    if yolo_system_standard:
        models["standard"] = {
            **MODEL_CONFIGS["standard"],
            "available": True,
            "classes": yolo_system_standard.classes
        }
    
    if yolo_system_weighted:
        models["weighted"] = {
            **MODEL_CONFIGS["weighted"], 
            "available": True,
            "classes": yolo_system_weighted.classes,
            "weights": {
                "pose": 3.0,
                "hands": 0.5, 
                "context": 1.0
            }
        }
    print(f"models : {models}")
    return {"models": models, "default_thresholds": DEFAULT_THRESHOLDS}

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    """
    Détection de gestes simple (compatibilité avec l'ancien endpoint)
    """
    try:
        # Lire et décoder l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Utiliser le modèle par défaut
        model = get_model("weighted" if yolo_system_weighted else "standard")
        
        # Prédiction
        result = model.predict(img)
        
        predicted_gesture = result.get('predicted_class', 'neutral')
        confidence = result.get('confidence', 0.0)
        
        # Seuil simple pour compatibilité
        threshold = DEFAULT_THRESHOLDS.get(predicted_gesture, 0.7)
        detected = confidence >= threshold and predicted_gesture != 'neutral'
        
        return {
            "detected": detected,
            "gesture": predicted_gesture,
            "confidence": float(confidence),
            "threshold_used": threshold
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.post("/detect_advanced/")
async def detect_advanced(
    file: UploadFile = File(...),
    model_type: str = Form("weighted"),
    apply_filters: str = Form("false"),
    custom_thresholds: Optional[str] = Form(None),
    return_image: str = Form("false"),
    return_debug: str = Form("false")
):
    """
    Détection de gestes avancée avec options complètes
    """
    try:
        # Debug: afficher tous les paramètres reçus
        print(f"[API] Paramètres reçus:")
        print(f"  - model_type: {model_type}")
        print(f"  - apply_filters: {apply_filters}")
        print(f"  - custom_thresholds: {custom_thresholds}")
        print(f"  - return_image: {return_image}")
        print(f"  - return_debug: {return_debug}")
        # Convertir les paramètres string en boolean
        apply_filters_bool = apply_filters.lower() in ["true", "1", "yes"]
        return_image_bool = return_image.lower() in ["true", "1", "yes"]
        return_debug_bool = return_debug.lower() in ["true", "1", "yes"]
        
        # Lire et décoder l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Sélectionner le modèle
        model = get_model(model_type)
        
        # Prédiction
        result = model.predict(img)
        
        predicted_gesture = result.get('predicted_class', 'neutral')
        confidence = result.get('confidence', 0.0)
        all_probabilities = result.get('all_probabilities', {})
        debug_info = result.get('debug_info', {})
        
        # Utiliser seuils personnalisés ou par défaut
        thresholds = DEFAULT_THRESHOLDS.copy()
        if custom_thresholds:
            try:
                import json
                custom_dict = json.loads(custom_thresholds)
                thresholds.update(custom_dict)
                print(f"📊 Seuils personnalisés parsés: {custom_dict}")
                print(f"📊 Seuils finaux utilisés: {thresholds}")
            except Exception as e:
                print(f"⚠️ Erreur parsing seuils custom: {e}")
                print(f"⚠️ custom_thresholds reçu: {repr(custom_thresholds)}")
                thresholds = DEFAULT_THRESHOLDS
        else:
            print("⚠️ Aucun seuil personnalisé reçu, utilisation des défauts")
        threshold = thresholds.get(predicted_gesture, 0.7)
        
        # Détection
        gesture_detected = confidence >= threshold and predicted_gesture != 'neutral'

        
        print(f"🎯 Détection: {predicted_gesture} (conf: {confidence:.2f}, seuil: {threshold:.2f}, détecté: {gesture_detected})")
        
        
        # Réponse de base
        response = {
            "detected": gesture_detected,
            "predicted_class": predicted_gesture,
            "confidence": float(confidence),
            "threshold_used": threshold,
            "model_used": model_type,
            "all_probabilities": {k: float(v) for k, v in all_probabilities.items()}
        }
        
        # Ajouter debug si demandé
        if return_debug_bool:
            response["debug_info"] = debug_info
            response["image_shape"] = img.shape
            response["thresholds_config"] = thresholds
        
        # Traiter l'image si demandé
        if return_image_bool or apply_filters_bool:
            processed_img = img.copy()
            
            # Ajouter annotations de base
            color = (0, 255, 0) if gesture_detected else (0, 0, 255)
            cv2.putText(processed_img, f"{predicted_gesture}: {confidence:.2f}", 
                       (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Appliquer filtres si demandé et disponible
            if apply_filters_bool and filter_system and gesture_detected and predicted_gesture != 'neutral':
                # Extraire keypoints pour les filtres
                keypoints = None
                if hasattr(model, 'pose_detector'):
                    yolo_results = model.pose_detector(img)
                    if yolo_results and len(yolo_results) > 0:
                        result_pose = yolo_results[0]
                        if result_pose.keypoints is not None and len(result_pose.keypoints.data) > 0:
                            keypoints = result_pose.keypoints.data[0].cpu().numpy().flatten()
                
                processed_img = filter_system.apply_filter_for_gesture(
                    processed_img, predicted_gesture, keypoints
                )
            
            if return_image_bool:
                # Encoder l'image en base64
                img_base64 = encode_image_to_base64(processed_img)
                response["annotated_image"] = img_base64
        
        return response
    
    except Exception as e:
        print(f"❌ Erreur dans detect_advanced: {str(e)}")  # Log pour debug
        raise HTTPException(status_code=500, detail=f"Erreur de traitement: {str(e)}")

@app.post("/batch_detect/")
async def batch_detect(
    files: list[UploadFile] = File(...),
    model_type: str = "weighted",
    custom_thresholds: Optional[Dict[str, float]] = None
):
    """
    Détection de gestes en lot pour plusieurs images
    """
    if len(files) > 20:  # Limite pour éviter la surcharge
        raise HTTPException(status_code=400, detail="Maximum 20 images par batch")
    
    results = []
    model = get_model(model_type)
    thresholds = custom_thresholds if custom_thresholds else DEFAULT_THRESHOLDS
    
    for i, file in enumerate(files):
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "index": i,
                    "filename": file.filename,
                    "error": "Image invalide"
                })
                continue
            
            result = model.predict(img)
            predicted_gesture = result.get('predicted_class', 'neutral')
            confidence = result.get('confidence', 0.0)
            threshold = thresholds.get(predicted_gesture, 0.7)
            detected = confidence >= threshold and predicted_gesture != 'neutral'
            
            results.append({
                "index": i,
                "filename": file.filename,
                "detected": detected,
                "predicted_class": predicted_gesture,
                "confidence": float(confidence),
                "threshold_used": threshold
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "model_used": model_type,
        "total_processed": len(results),
        "total_detected": sum(1 for r in results if r.get('detected', False))
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