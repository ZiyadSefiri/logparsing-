"""
API améliorée avec alertes intégrées + Chargement du modèle entraîné
"""
from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.monitoring_pipeline import MonitoringPipeline

app = FastAPI(title="Log RCA Monitoring API")
PIPELINE = None

class LogRequest(BaseModel):
    log_text: str
    source: str | None = None
    event_timestamp: str | None = None

@app.on_event("startup")
async def startup():
    global PIPELINE
    PIPELINE = MonitoringPipeline()
    print(f"✓ API prête ({PIPELINE.backend}, device: {PIPELINE.device})")

@app.get("/")
async def root():
    return {
        "message": "Log RCA Monitoring API",
        "model_loaded": PIPELINE is not None,
        "classes": PIPELINE.class_names if PIPELINE else [],
        "backend": PIPELINE.backend if PIPELINE else "uninitialized",
        "endpoints": ["/predict", "/alerts", "/stats", "/health"],
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": PIPELINE is not None,
        "classes_count": len(PIPELINE.class_names) if PIPELINE else 0,
        "backend": PIPELINE.backend if PIPELINE else "uninitialized",
    }

@app.post("/predict")
async def predict(request: LogRequest):
    """Prédiction + Alerte automatique"""
    result = PIPELINE.process_log(
        request.log_text,
        source=request.source,
        event_timestamp=request.event_timestamp,
    )

    return {
        "log": request.log_text,
        "prediction": result["prediction"],
        "label": result["label"],
        "severity": result["severity"],
        "priority": result["priority"],
        "recommended_actions": result["recommended_actions"],
        "contact": result["contact"],
        "confidence": result["confidence_percent"],
        "confidence_raw": result["confidence"],
        "all_probabilities": result["probabilities"],
        "alert_triggered": result["alert_triggered"],
        "source": result["source"],
        "event_timestamp": result["event_timestamp"],
        "processed_at": result["processed_at"],
        "backend": result["model_backend"],
        "latency_ms": result["latency_ms"],
    }

@app.post("/predict_batch")
async def predict_batch(logs: list[LogRequest]):
    """Predictions batch pour integrations externes"""
    return {
        "count": len(logs),
        "predictions": [await predict(log_request) for log_request in logs],
    }

@app.get("/causes")
async def causes():
    """Catalogue des causes racines gerees par le systeme"""
    from src.rca_catalog import RCA_CATALOG

    return RCA_CATALOG

@app.get("/alerts")
async def get_alerts():
    """Récupère toutes les alertes"""
    return {
        "alerts": PIPELINE.alerter.alerts if PIPELINE else [],
        "count": len(PIPELINE.alerter.alerts) if PIPELINE else 0,
    }

@app.get("/stats")
async def get_stats():
    """Récupère les statistiques"""
    stats = PIPELINE.tracker.get_stats() if PIPELINE else {}
    return {
        **stats,
        "backend": PIPELINE.backend if PIPELINE else "uninitialized",
        "classes": PIPELINE.class_names if PIPELINE else [],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
