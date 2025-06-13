from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from pose_detection_model import PoseDetector

app = FastAPI()
pose_detector = PoseDetector()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Détection
    results = pose_detector.detect_poses(frame)
    keypoints_data = pose_detector.get_keypoints_data(results)

    gesture = "None"
    detected = False

    if keypoints_data:
        kp_data = keypoints_data[0]
        keypoints = kp_data['keypoints']
        conf = kp_data['confidence']

        if pose_detector.detect_gesture(keypoints, conf, "jump"):
            gesture = "Jump"
            detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "dab"):
            gesture = "Dab"
            detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "hands_up"):
            gesture = "Hands_Up"
            detected = True
        elif pose_detector.detect_gesture(keypoints, conf, "twerk"):
            gesture = "Twerk"
            detected = True

    return JSONResponse(content={
        "detected": detected,
        "gesture": gesture
    })
