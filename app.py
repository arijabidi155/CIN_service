from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from cin_validator import CINValidator
import os

app = FastAPI(
    title="Sahl Express CIN Validation Service",
    description="Microservice de validation et détection de carte d'identité (CIN)",
    version="1.0.0"
)

# API key for security, default provided for local development, should be overridden in HF space settings
API_KEY = os.getenv("CIN_API_KEY", "sahl_express_secret_key_123")

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# Instantiate validator globally to load YOLO model once at startup
validator = CINValidator()

class CINRequest(BaseModel):
    imageUrl: str
    side: str = "recto"  # "recto" or "verso"

@app.post("/validate-cin")
def validate_cin(data: CINRequest, api_key: str = Depends(verify_api_key)):
    """
    Télécharge l'image depuis Cloudinary, détecte la CIN et vérifie sa netteté et ses ancres visuelles.
    """
    result = validator.validate(data.imageUrl, data.side)
    return result

@app.get("/")
def read_root():
    return {
        "status": "active",
        "service": "Sahl Express CIN Validation Microservice",
        "info": "Consultez /docs pour l'interface Swagger."
    }
