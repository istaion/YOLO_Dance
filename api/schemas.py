# (optionnel) Schémas de validation Pydantic

# app/schemas.py
from pydantic import BaseModel, EmailStr

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
