"""
Script de test du système de monitoring complet
"""
import requests
import time
import json
from pathlib import Path

# Logs réalistes à tester
test_logs = [
    {
        "text": "CRITICAL Disk space low on /var: 99% used",
        "expected": "disk_full",
        "severity": "🔴 CRITICAL"
    },
    {
        "text": "ERROR Connection timeout to 192.168.1.100:8080 after 30000ms",
        "expected": "network_failure",
        "severity": "🟠 HIGH"
    },
    {
        "text": "WARNING Memory usage at 95% on node-001, possible memory leak detected",
        "expected": "memory_leak",
        "severity": "🟠 HIGH"
    },
    {
        "text": "ERROR Authentication failed for user admin from IP 10.0.0.5",
        "expected": "authentication_error",
        "severity": "🟡 MEDIUM"
    },
    {
        "text": "ERROR Thread pool exhausted, 500 pending tasks, cannot process requests",
        "expected": "resource_exhaustion",
        "severity": "🟡 MEDIUM"
    }
]

print("=" * 70)
print("🧪 TEST COMPLET DU SYSTÈME DE MONITORING")
print("=" * 70)


def build_client():
    """Utilise HTTP si possible, sinon le pipeline local."""
    session = requests.Session()
    try:
        response = session.get("http://localhost:8000/health", timeout=1)
        response.raise_for_status()
        print("Mode test: HTTP localhost")
        return session, True
    except Exception:
        from src.monitoring_pipeline import MonitoringPipeline

        class LocalPipelineClient:
            def __init__(self):
                self.pipeline = MonitoringPipeline()

            def predict(self, payload):
                result = self.pipeline.process_log(payload["log_text"], source="local_test")
                return {
                    "prediction": result["prediction"],
                    "confidence": result["confidence_percent"],
                    "alert_triggered": result["alert_triggered"],
                    "severity": result["severity"],
                    "recommended_actions": result["recommended_actions"],
                    "latency_ms": result["latency_ms"],
                }

            def alerts(self):
                return {
                    "alerts": self.pipeline.alerter.alerts,
                    "count": len(self.pipeline.alerter.alerts),
                }

        print("Mode test: pipeline local (fallback sandbox)")
        return LocalPipelineClient(), False


client, use_http = build_client()

correct = 0
total = len(test_logs)

for i, log_data in enumerate(test_logs, 1):
    print(f"\n[Test {i}/{total}]")
    print(f"Log: {log_data['text'][:60]}...")
    
    try:
        # Envoyer à l'API
        if use_http:
            response = client.post(
                "http://localhost:8000/predict",
                json={"log_text": log_data['text']}
            )
            result = response.json()
        else:
            result = client.predict({"log_text": log_data['text']})
        prediction = result['prediction']
        confidence = result['confidence']
        alert = result['alert_triggered']
        severity = result.get('severity', 'N/A')
        actions = result.get('recommended_actions', [])
        latency_ms = result.get('latency_ms', 0)
        
        # Vérifier
        is_correct = prediction == log_data['expected']
        if is_correct:
            correct += 1
            print(f"✅ Prédiction correcte: {prediction}")
        else:
            print(f"❌ Attendu: {log_data['expected']}, Got: {prediction}")
        
        print(f"   Confiance: {confidence}")
        print(f"   Sévérité: {severity}")
        print(f"   Latence: {latency_ms:.0f} ms")
        if actions:
            print(f"   Action: {actions[0]}")
        print(f"   Alerte: {'🚨 OUI' if alert else '❌ NON'}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    time.sleep(1)  # Pause entre les requêtes

# Résultats finaux
print("\n" + "=" * 70)
print("📊 RÉSULTATS")
print("=" * 70)
print(f"✅ Correctes: {correct}/{total} ({correct/total*100:.0f}%)")

# Afficher les alertes enregistrées
print("\n📋 ALERTES ENREGISTRÉES:")
try:
    if use_http:
        response = client.get("http://localhost:8000/alerts")
        alerts = response.json()['alerts']
    else:
        alerts = client.alerts()['alerts']
    print(f"Total: {len(alerts)} alertes")
    for alert in alerts[-3:]:  # Afficher les 3 dernières
        print(f"  • {alert['cause']}: {alert['confidence']}")
except:
    print("Impossible de récupérer les alertes")

# Stats
print("\n📈 STATISTIQUES:")
stats_file = Path('data/performance.json')
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
    print(f"  Prédictions totales: {stats['total_predictions']}")
    print(f"  Alertes déclenchées: {stats['alerts_triggered']}")
    print(f"  Distribution:")
    for cause, count in stats['predictions_by_cause'].items():
        print(f"    - {cause}: {count}")

print("\n" + "=" * 70)
print("🎉 TEST TERMINÉ!")
print("=" * 70)
