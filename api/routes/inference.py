# app/routes/inference.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from config import get_current_user
from detector import PoseDetector
import numpy as np
import cv2

router = APIRouter()
detector = PoseDetector(model_path="yolov8n-pose.pt")

@router.post("/detect-gesture")
async def detect_gesture_api(
    file: UploadFile = File(...),
    current_user: int = Depends(get_current_user)
):
    """
    Endpoint protégé qui détecte un geste spécifique à partir d'une image.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Lire le fichier en image OpenCV
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Détection des poses
    results = detector.detect_poses(frame)
    if not results or len(results[0].keypoints.xy) == 0:
        return JSONResponse(content={"message": "No person detected"}, status_code=200)

    keypoints = results[0].keypoints.xy.cpu().numpy()
    conf = results[0].keypoints.conf.cpu().numpy()
    is_hands_up = detector.detect_gesture(keypoints, conf, gesture_type="hands_up")

    return {"user_id": current_user, "gesture": "hands_up", "detected": is_hands_up}
