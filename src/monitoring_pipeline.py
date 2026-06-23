"""
Pipeline partagé pour le monitoring temps réel et le replay de logs.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

try:
    from src.inference import RCAInference
    from src.performance_tracker import PerformanceTracker
    from src.preprocessing import LogPreprocessor
    from src.rca_catalog import get_cause_details
    from src.runtime_loader import load_runtime_model
    from src.send_alert import SimpleAlerter
except ImportError:  # pragma: no cover - compatibilite execution directe
    from inference import RCAInference
    from performance_tracker import PerformanceTracker
    from preprocessing import LogPreprocessor
    from rca_catalog import get_cause_details
    from runtime_loader import load_runtime_model
    from send_alert import SimpleAlerter


class MonitoringPipeline:
    def __init__(
        self,
        alert_threshold: float = 0.7,
        predictions_file: str = "data/stream_predictions.jsonl",
    ):
        self.alert_threshold = alert_threshold
        self.predictions_file = Path(predictions_file)
        self.predictions_file.parent.mkdir(parents=True, exist_ok=True)

        self.model, self.class_names, self.device, self.backend = load_runtime_model()
        self.inferencer = RCAInference(
            self.model,
            device=self.device,
            class_names=self.class_names,
        )
        self.alerter = SimpleAlerter()
        self.tracker = PerformanceTracker()
        self.preprocessor = LogPreprocessor()

    def process_log(self, log_text: str, source: str | None = None, event_timestamp: str | None = None):
        started_at = time.perf_counter()
        cleaned_log = self.preprocessor.clean_log(log_text)
        prediction = self.inferencer.predict_single(log_text)
        predicted_class = prediction["predicted_class"]
        confidence = prediction["confidence"]
        details = get_cause_details(predicted_class)

        alert_triggered = (
            confidence >= self.alert_threshold and predicted_class != "normal_operation"
        )
        self.tracker.record_prediction(
            predicted_class,
            confidence,
            alert_triggered=alert_triggered,
        )

        alert = None
        if alert_triggered:
            alert = self.alerter.log_alert(predicted_class, confidence, log_text, details=details)

        latency_ms = (time.perf_counter() - started_at) * 1000

        record = {
            "source": source or "unknown",
            "event_timestamp": event_timestamp,
            "processed_at": datetime.now().isoformat(),
            "log": log_text,
            "cleaned_log": cleaned_log,
            "prediction": predicted_class,
            "label": details["label"],
            "severity": details["severity"],
            "priority": details["priority"],
            "contact": details["contact"],
            "recommended_actions": details["actions"],
            "confidence": confidence,
            "probabilities": prediction["probabilities"],
            "alert_triggered": alert_triggered,
            "model_backend": self.backend,
            "latency_ms": latency_ms,
        }

        with open(self.predictions_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        return {
            **record,
            "confidence_percent": f"{confidence:.2%}",
            "alert": alert,
        }

    def reset_outputs(self):
        for path in [
            self.predictions_file,
            Path("data/alerts.jsonl"),
            Path("data/performance.json"),
        ]:
            if path.exists():
                path.unlink()

        self.alerter = SimpleAlerter()
        self.tracker = PerformanceTracker()
