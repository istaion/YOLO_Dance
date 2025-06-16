#!/usr/bin/env python3
"""
Script de lancement pour l'application Teachable Machine
Généré automatiquement par setup_teachable_machine.py
"""

import streamlit as st
import sys
import os

# Ajouter le répertoire du projet au PYTHONPATH
project_root = "/home/utilisateur/YOLO_Dance"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Lancer l'application
if __name__ == "__main__":
    import subprocess
    app_path = os.path.join(project_root, "app", "streamlit_app_teachable_machine.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
