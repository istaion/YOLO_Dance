# Lance l'app Streamlit, récupère webcam et envoie à l'API
import streamlit as st
import cv2
import numpy as np
import requests
from datetime import datetime
import os

#Dossier pour sauvegarder les screenshots
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(SAVE_DIR, exist_ok=True)

#Adresse de l'API backend

API_URL = "http://localhost:8000/detect/"

#titre de l'app
st.set_page_config(page_title="YOLO_Danse - Détection en Live")
st.title("🕺 YOLO_Dance – Capture automatique de tes meilleurs moves")

#button pour démarrer la caméra
if st.button("🎥 Lancer la caméra"):
    stframe= st.empty()
    cap= cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Impossible d'accéder à la caméra")
    else:
        st.success("✅ Caméra active")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur lors de la lecture de la caméra.")
            break

        #Envoi à l'API
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(
            API_URL,
            files={"file":("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
        )

        if response.ok and response.json().get("detected"):
            # 📸 Mouvement détecté → capture d’écran
            ts= datetime.now().strftime("%Y%m%d_%H%M%S")
            filename= f"move_{ts}.jpg"
            filepath= os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            st.toast(f"📸 Mouvement détecté ! Photo enregistrée : {filename}")

        # 📺 Affichage live
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()