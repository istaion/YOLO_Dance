# 🕺 YOLO DANCE

**Système de détection de mouvements de danse en temps réel avec capture automatique**

YOLO_Dance est un projet étudiant développé pour une entreprise de karaoke souhaitant un système de photo automatique lorsque les clients exécutent certains mouvements particuliers (mains en l'air, dab, twerk, etc.). L'objectif est de réaliser cette mission via des modèles d'IA avancés.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3.148-yellow.svg)

## 🎯 Fonctionnalités

### 🤖 Double Système de Détection
- **Teachable Machine** : 15 gestes de danse (Twerk, Dab, Macaréna, Floss, Funk, CrossArm, etc.)
- **YOLOv8 API** : 5 gestes principaux (hands_up, dab, twerk, jump, neutral) avec pose detection avancée

### 📸 Capture Intelligente
- Détection en temps réel via webcam
- Capture automatique avec cooldown configurable
- Filtres visuels personnalisés par geste
- Sauvegarde des photos sans filtres

### 🔒 Système d'Authentification
- Authentification JWT sécurisée
- Gestion des utilisateurs avec base de données
- Logs personnalisés des prédictions

### 🎨 Interface 
- Interface Streamlit 
- Thème violet personnalisé
- Métriques en temps réel
- Configuration avancée des seuils

## 🏗️ Architecture

```
YOLO_Dance/
├── app/                          # 💻 Interface Streamlit
│   ├── main_app.py              # Point d'entrée principal
│   ├── auth_page.py             # Authentification
│   ├── streamlit_app_teachable_machine.py
│   ├── streamlit_app_unified.py
│   ├── components/              # Composants UI
│   ├── styles/                  # Thèmes CSS
│   └── config/                  # Configuration
├── api/                         # 🔗 Backend FastAPI
│   ├── main.py                  # API principale
│   ├── routes/                  # Routes REST
│   ├── models.py                # Modèles de données
│   ├── database.py              # Base de données
│   ├── pose_detection_model.py  # Détection YOLO
│   └── config.py                # Configuration JWT
├── models/                      # 🤖 Modèles IA
│   ├── teachable_machine/       # Modèle TM
│   └── best_model.pth           # Modèle YOLO
├── images/                      # 📸 Photos capturées
└── scripts/                     # 🛠️ Utilitaires
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- Webcam fonctionnelle
- 4GB RAM minimum
- GPU recommandé pour YOLO

### 1. Cloner le projet
```bash
git clone https://github.com/votre-username/YOLO_Dance.git
cd YOLO_Dance
```

### 2. Créer l'environnement virtuel
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configuration initiale
```bash
# Initialiser la base de données
python api/create_db_simple.py

# Corriger les fichiers manquants (si nécessaire)
python fix_misssing_files.py
```

### 5. Télécharger les modèles
```bash
# Le modèle YOLOv8 pose sera téléchargé automatiquement
# Placez votre modèle Teachable Machine dans models/teachable_machine/
```

## 🎮 Utilisation

### Lancement Automatique (Recommandé)
```bash
python run_app.py
```
Cette commande démarre automatiquement :
- ✅ L'API FastAPI (http://localhost:8000)
- ✅ L'interface Streamlit (http://localhost:8501)

### Lancement Manuel

#### Option 1 : API + Interface
```bash
# Terminal 1 - API
cd api
python main.py

# Terminal 2 - Interface
streamlit run app/main_app.py
```

#### Option 2 : Mode Teachable Machine uniquement
```bash
streamlit run app/streamlit_app_teachable_machine.py
```

### 🔐 Première Connexion
1. Ouvrez http://localhost:8501
2. Créez un compte dans l'onglet "Inscription"
3. Connectez-vous avec vos identifiants
4. Choisissez votre mode de détection

## 🎭 Modes de Détection

### 🤖 Mode Teachable Machine
- **Avantages** : Traitement local, 15 gestes spécialisés
- **Utilisation** : Idéal pour les gestes de danse complexes
- **Gestes** : Twerk, Dab, Macaréna, Floss, Funk, CrossArm, V_Signs, etc.

### 🔗 Mode API Unifiée
- **Avantages** : YOLOv8 + pose detection avancée
- **Utilisation** : Meilleure précision sur les poses corporelles
- **Gestes** : hands_up, dab, twerk, jump, neutral
- **Authentification** : Logs serveur personnalisés

## ⚙️ Configuration

### Seuils de Détection
Personnalisez la sensibilité par geste dans la sidebar :
- **Seuil élevé (0.8+)** : Détection stricte
- **Seuil moyen (0.6-0.8)** : Équilibré
- **Seuil bas (0.4-0.6)** : Permissif

### Filtres Visuels
- 🍑 **Twerk** : Pêche sur les hanches
- 👑 **Hands Up** : Couronne royale
- ⚡ **Dab** : Rayon lumineux
- ✝️ **CrossArm** : Croix lumineuse
- 🕶️ **Jul** : Lunettes de soleil

### Variables d'Environnement
```bash
# API Configuration
SECRET_KEY=your-super-secret-key
DATABASE_URL=sqlite:///./dance_app.db
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Camera Settings
CAMERA_FPS_LIMIT=30
CAMERA_RESOLUTION=640x480
```

## 📋 API Endpoints

### Authentification
- `POST /auth/register` - Inscription
- `POST /auth/token` - Connexion
- `GET /auth/me` - Profil utilisateur

### Détection
- `POST /api/detect-gesture` - Détecter un geste
- `GET /api/gestures` - Liste des gestes
- `GET /api/logs` - Historique utilisateur

### Monitoring
- `GET /health` - Santé de l'API
- `GET /docs` - Documentation interactive

## 🛠️ Résolution des Problèmes

### Erreurs Communes

#### 1. Modèle non trouvé
```bash
# Vérifiez la présence des modèles
ls models/
python fix_misssing_files.py
```

#### 2. API non accessible
```bash
# Vérifiez que l'API tourne
curl http://localhost:8000/health

# Redémarrez l'API
cd api && python main.py
```

#### 3. Caméra non détectée
```bash
# Testez la caméra
python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'ERREUR')"
```

#### 4. Erreurs de dépendances
```bash
# Réinstallez les dépendances
pip install -r requirements.txt --force-reinstall
```

### Scripts de Diagnostic
```bash
# Test complet de l'API
python api/test_api.py

# Réparation de la base de données
python api/fix_database.py

# Correction des fichiers manquants
python fix_misssing_files.py
```

## 📊 Performance

### Configuration Recommandée
- **CPU** : Intel i5+ ou AMD Ryzen 5+
- **RAM** : 8GB minimum
- **GPU** : NVIDIA GTX 1060+ (optionnel mais recommandé)
- **Webcam** : 720p minimum

### Optimisations
- Mode GPU automatique si CUDA disponible
- Limitation FPS configurable
- Traitement local pour Teachable Machine
- Cache des modèles pour des démarrages rapides

## 🧪 Tests

### Tests Unitaires
```bash
# Test de l'API
python api/test_api.py

# Test de la base de données
python api/fix_database.py
```

### Tests d'Intégration
```bash
# Test complet via l'interface
python run_app.py
# Puis accédez à http://localhost:8501
```

## 📈 Métriques

Le système suit automatiquement :
- ✅ Nombre de photos capturées
- ✅ Précision par geste
- ✅ Temps de réponse
- ✅ Utilisation des modèles
- ✅ Logs d'authentification

## 🔮 Roadmap

### Version 2.0
- [ ] Support multi-utilisateurs simultanés
- [ ] Intégration avec systèmes de karaoke
- [ ] Entraînement en ligne des modèles
- [ ] Application mobile companion

### Améliorations
- [ ] Support vidéo en temps réel
- [ ] Détection de groupes
- [ ] Analyses statistiques avancées
- [ ] Export des données

## 🤝 Contribution

### Comment Contribuer
1. Fork le projet
2. Créez votre branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

### Standards de Code
- Python 3.8+ compatible
- Documentation docstring
- Tests unitaires pour les nouvelles fonctionnalités
- Respect PEP 8

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Équipe

- **Victor Poutot**
  <a href="https://github.com/istaion" target="_blank">
      <img loading="lazy" src="images/github-mark.png" width="30" height="30" alt="GitHub Logo">
  </a>

- **Léo Gallus**
  <a href="https://github.com/Leozmee" target="_blank">
      <img loading="lazy" src="images/github-mark.png" width="30" height="30" alt="GitHub Logo">
  </a>

- **Raouf Addeche**
  <a href="https://github.com/RaoufAddeche" target="_blank">
      <img loading="lazy" src="images/github-mark.png" width="30" height="30" alt="GitHub Logo">
  </a>

- **Ludivine Raby**
  <a href="https://github.com/ludivineRB" target="_blank">
      <img loading="lazy" src="images/github-mark.png" width="30" height="30" alt="GitHub Logo">
  </a>

## 🙏 Remerciements

- [Ultralytics](https://github.com/ultralytics/ultralytics) pour YOLOv8
- [Google](https://teachablemachine.withgoogle.com/) pour Teachable Machine
- [Streamlit](https://streamlit.io/) pour l'interface
- [FastAPI](https://fastapi.tiangolo.com/) pour l'API
- [Simplon] (https://simplon.com/) pour tous la qualité d'apprentissage

## 📞 Support

- **Issues** : [GitHub Issues](https://github.com/votre-username/YOLO_Dance/issues)
- **Discussions** : [GitHub Discussions](https://github.com/votre-username/YOLO_Dance/discussions)


---

**Made with ❤️ for the Dance Community**

*"Dance like nobody's watching, but YOLO Dance is definitely capturing the moment!"* 🕺💃
yolo_dance_readme.md
10 Ko