# api/routes/log.py
from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from datetime import timedelta
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schemas
import models
import config
from database import get_db

# IMPORTANT: Créer le router
router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    """Hash un mot de passe"""
    return pwd_context.hash(password)

def authenticate_user(db: Session, username: str, password: str):
    """Authentifie un utilisateur"""
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

@router.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Enregistre un nouvel utilisateur"""
    print(f"🔍 Register endpoint called: {user.username}, {user.email}")
    
    # Vérifier si l'utilisateur existe déjà
    existing_user = db.query(models.User).filter(
        (models.User.username == user.username) | (models.User.email == user.email)
    ).first()
    
    if existing_user:
        if existing_user.username == user.username:
            raise HTTPException(
                status_code=400, 
                detail="Username already registered"
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail="Email already registered"
            )
    
    # Créer le nouvel utilisateur
    hashed_pw = get_password_hash(user.password)
    new_user = models.User(
        username=user.username, 
        email=user.email, 
        hashed_password=hashed_pw
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"✅ Utilisateur créé: {new_user.username} (ID: {new_user.id})")
        return {
            "message": "User created successfully", 
            "user_id": new_user.id,
            "username": new_user.username
        }
    except Exception as e:
        db.rollback()
        print(f"❌ Erreur DB: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Database error"
        )

@router.post("/token")
def login(user_login: schemas.UserLogin, db: Session = Depends(get_db)):
    """Connexion utilisateur et génération du token"""
    print(f"🔍 Login endpoint called: {user_login.username}")
    
    # Authentifier l'utilisateur
    user_db = authenticate_user(db, user_login.username, user_login.password)
    if not user_db:
        print(f"❌ Échec authentification pour: {user_login.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Créer le token d'accès
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = config.create_access_token(
        data={"sub": user_db.username}, 
        expires_delta=access_token_expires
    )
    
    print(f"✅ Login réussi pour: {user_login.username}")
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_id": user_db.id,
        "username": user_db.username
    }

@router.get("/users")
def list_users(
    current_user_id: int = Depends(config.get_current_user),
    db: Session = Depends(get_db)
):
    """Liste tous les utilisateurs (endpoint protégé)"""
    # Vérifier que l'utilisateur actuel existe toujours
    current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if not current_user:
        raise HTTPException(status_code=404, detail="Current user not found")
    
    users = db.query(models.User).all()
    return [
        {
            "id": u.id, 
            "username": u.username, 
            "email": u.email,
            "is_active": u.is_active,
            "created_at": u.created_at
        } 
        for u in users
    ]

@router.get("/me")
def get_current_user_info(
    current_user_id: int = Depends(config.get_current_user),
    db: Session = Depends(get_db)
):
    """Obtient les informations de l'utilisateur actuel"""
    user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_active": user.is_active,
        "created_at": user.created_at
    }

print("✅ Router log.py créé avec endpoints: /register, /token, /users, /me")
