# (optionnel) Schémas de validation Pydantic

# app/schemas.py
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    label: str
    confidence: float
