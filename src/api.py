"""
API FastAPI pour l'inférence en temps réel
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from pathlib import Path
import logging
from datetime import datetime
import json
import sys
from pathlib import Path as PathlibPath

# ✅ AJOUTER CES IMPORTS
sys.path.insert(0, str(PathlibPath(__file__).parent))
from models import TransformerRCA
from inference import RCAInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Log RCA API",
    description="API de classification des causes racines dans les logs",
    version="1.0.0"
)

# Modèle global
MODEL = None
INFERENCER = None
CLASS_NAMES = ['network_failure', 'memory_leak', 'disk_full', 
               'authentication_error', 'resource_exhaustion']

class LogRequest(BaseModel):
    """Requête de prédiction"""
    log_text: str
    top_k: int = 1

class LogResponse(BaseModel):
    """Réponse de prédiction"""
    log_text: str
    predicted_class: str
    confidence: float
    probabilities: dict
    timestamp: str

class BatchLogRequest(BaseModel):
    """Requête batch"""
    logs: list[str]

@app.on_event("startup")
async def startup_event():
    """Chargement du modèle au démarrage"""
    global MODEL, INFERENCER
    
    model_path = Path("models/best_model.pth")
    
    if not model_path.exists():
        logger.warning(f"Modèle non trouvé: {model_path}")
        logger.warning("Initialisant un modèle par défaut...")
        MODEL = TransformerRCA(num_classes=len(CLASS_NAMES))
    else:
        MODEL = TransformerRCA(num_classes=len(CLASS_NAMES))
        MODEL.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"Modèle chargé: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL.to(device)
    INFERENCER = RCAInference(MODEL, device=device, class_names=CLASS_NAMES)
    logger.info(f"API démarrée (device: {device})")

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Log RCA API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Vérification de la santé de l'API"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=LogResponse)
async def predict(request: LogRequest):
    """Prédiction pour un seul log"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        result = INFERENCER.predict_single(request.log_text)
        
        return LogResponse(
            log_text=request.log_text,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Erreur de prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(request: BatchLogRequest):
    """Prédictions batch"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        results = INFERENCER.predict_batch(request.logs, batch_size=32)
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erreur batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
async def explain(request: LogRequest):
    """Explication détaillée"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        explanation = INFERENCER.explain_prediction(request.log_text)
        
        return {
            **explanation,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
