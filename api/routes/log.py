from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import schemas
import models
import config
from database import SessionLocal
from authentication import get_db, get_password_hash, authenticate_user

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(user.password)
    new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"msg": "User created successfully"}

@router.post("/token")
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    user_db = authenticate_user(db, user.username, user.password)
    if not user_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = config.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
