from fastapi import APIRouter, HTTPException
import schemas
#from db.CRUD.user import get_user_by_username
from config import verify_token, create_access_token, verify_password
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import schemas, models, config
from database import SessionLocal
from passlib.context import CryptContext

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dépendance pour accéder à la DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return False
    if not pwd_context.verify(password, user.hashed_password):
        return False
    return user

# @router.post("/register")
# def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
#     existing_user = db.query(models.User).filter(models.User.username == user.username).first()
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Username already registered")
    
#     hashed_pw = get_password_hash(user.password)
#     new_user = models.User(username=user.username, email=user.email, hashed_password=hashed_pw)
#     db.add(new_user)
#     db.commit()
#     db.refresh(new_user)
#     return {"msg": "User created successfully"}

# @router.post("/token")
# def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
#     user_db = authenticate_user(db, user.username, user.password)
#     if not user_db:
#         raise HTTPException(status_code=401, detail="Invalid credentials")

#     access_token = config.create_access_token(data={"sub": user.username})
#     return {"access_token": access_token, "token_type": "bearer"}
