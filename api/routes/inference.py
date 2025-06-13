# api/routes/inference.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import numpy as np
import cv2
import io
from PIL import Image

from config import get_current_user
from pose_detection_model import PoseDetector
from database import get_db
import models
import schemas

router = APIRouter()

# Initialiser le détecteur de poses (une seule fois)
try:
    detector = PoseDetector(model_path="yolov8n-pose.pt")
    print("✅ PoseDetector initialisé avec succès")
except Exception as e:
    print(f"❌ Erreur lors de l'initialisation du PoseDetector: {e}")
    detector = None

@router.get("/health")
async def health_check():
    """Endpoint de vérification de santé"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "message": "Service de détection de poses actif"
    }

@router.post("/detect-gesture")
async def detect_gesture_api(
    file: UploadFile = File(...),
    current_user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint protégé qui détecte un geste spécifique à partir d'une image.
    """
    # Vérifier que le détecteur est initialisé
    if detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service de détection non disponible"
        )
    
    # Vérifier le format de l'image
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="Format d'image invalide. Formats acceptés: JPEG, PNG"
        )
    
    try:
        # Lire et convertir l'image
        contents = await file.read()
        
        # Convertir en image OpenCV
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Impossible de décoder l'image"
            )
        
        # Détection des poses
        results = detector.detect_poses(frame)
        
        if not results or len(results[0].keypoints.xy) == 0:
            # Enregistrer la tentative dans les logs
            log_prediction(db, current_user_id, "no_person", 0.0, "yolov8n-pose")
            
            return JSONResponse(
                content={
                    "user_id": current_user_id,
                    "gesture": "Neutral",
                    "detected": False,
                    "confidence": 0.0,
                    "message": "Aucune personne détectée"
                }, 
                status_code=200
            )
        
        # Extraire les keypoints
        keypoints = results[0].keypoints.xy.cpu().numpy()
        conf = results[0].keypoints.conf.cpu().numpy()
        
        # Ajouter à l'historique pour la détection de saut
        detector.add_position_to_history(keypoints, conf, frame=None)
        
        # Tester différents gestes
        detected_gestures = {}
        
        # Test hands_up
        hands_up_detected = detector.detect_gesture(keypoints, conf, gesture_type="hands_up")
        if hands_up_detected:
            detected_gestures["hands_up"] = 0.8
        
        # Test dab
        dab_detected = detector.detect_gesture(keypoints, conf, gesture_type="dab")
        if dab_detected:
            detected_gestures["dab"] = 0.85
        
        # Test twerk
        twerk_detected = detector.detect_gesture(keypoints, conf, gesture_type="twerk")
        if twerk_detected:
            detected_gestures["twerk"] = 0.75
        
        # Test jump (retourne un dictionnaire)
        jump_result = detector.detect_gesture(keypoints, conf, gesture_type="jump")
        if isinstance(jump_result, dict) and jump_result.get("detected", False):
            detected_gestures["jump"] = 0.9
        
        # Sélectionner le geste avec la plus haute confiance
        if detected_gestures:
            best_gesture = max(detected_gestures.items(), key=lambda x: x[1])
            gesture_name = best_gesture[0]
            confidence = best_gesture[1]
            detected = True
        else:
            gesture_name = "Neutral"
            confidence = 0.0
            detected = False
        
        # Enregistrer la prédiction dans les logs
        log_prediction(db, current_user_id, gesture_name, confidence, "yolov8n-pose")
        
        return {
            "user_id": current_user_id,
            "gesture": gesture_name,
            "detected": detected,
            "confidence": confidence,
            "all_detections": detected_gestures,
            "keypoints_count": len(keypoints[0]) if len(keypoints) > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la détection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )

@router.get("/gestures")
async def get_available_gestures(current_user_id: int = Depends(get_current_user)):
    """
    Retourne la liste des gestes disponibles
    """
    gestures = [
        {
            "name": "hands_up",
            "description": "Mains levées en l'air",
            "emoji": "🙌"
        },
        {
            "name": "dab",
            "description": "Mouvement dab",
            "emoji": "💪"
        },
        {
            "name": "twerk",
            "description": "Position twerk",
            "emoji": "🍑"
        },
        {
            "name": "jump",
            "description": "Saut en cours",
            "emoji": "🦘"
        },
        {
            "name": "Neutral",
            "description": "Position neutre",
            "emoji": "😐"
        }
    ]
    
    return {
        "user_id": current_user_id,
        "available_gestures": gestures,
        "total_count": len(gestures)
    }

@router.get("/logs")
async def get_user_logs(
    current_user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 10
):
    """
    Récupère l'historique des prédictions de l'utilisateur
    """
    logs = db.query(models.PredictionLog).filter(
        models.PredictionLog.user_id == current_user_id
    ).order_by(models.PredictionLog.created_at.desc()).limit(limit).all()
    
    return {
        "user_id": current_user_id,
        "logs": [
            {
                "id": log.id,
                "prediction": log.prediction,
                "confidence": log.confidence,
                "model_used": log.model_used,
                "created_at": log.created_at
            }
            for log in logs
        ],
        "total_count": len(logs)
    }

def log_prediction(db: Session, user_id: int, prediction: str, confidence: float, model_used: str):
    """
    Enregistre une prédiction dans la base de données
    """
    try:
        log_entry = models.PredictionLog(
            user_id=user_id,
            prediction=prediction,
            confidence=confidence,
            model_used=model_used
        )
        db.add(log_entry)
        db.commit()
        print(f"📝 Log enregistré: {prediction} ({confidence:.2f}) pour user {user_id}")
    except Exception as e:
        db.rollback()
        print(f"❌ Erreur lors de l'enregistrement du log: {e}")

print("✅ Router inference.py créé avec endpoints: /detect-gesture, /gestures, /logs, /health")
