"""
Service d'alerte simplifié et fonctionnel
"""
import json
from datetime import datetime

try:
    from src.rca_catalog import get_cause_details
except ImportError:  # pragma: no cover - compatibilite execution directe
    from rca_catalog import get_cause_details

class SimpleAlerter:
    def __init__(self):
        self.alerts = []
    
    def log_alert(self, cause, confidence, log_text, details=None):
        """Enregistre une alerte"""
        details = details or get_cause_details(cause)
        alert = {
            'timestamp': datetime.now().isoformat(),
            'cause': cause,
            'label': details['label'],
            'confidence': f"{confidence:.2%}",
            'confidence_raw': float(confidence),
            'log': log_text,
            'severity': details['severity'],
            'priority': details['priority'],
            'contact': details['contact'],
            'recommended_actions': details['actions'],
        }
        self.alerts.append(alert)
        
        # Afficher
        print(f"\n{'='*60}")
        print(f"🚨 ALERTE {alert['severity']}")
        print(f"{'='*60}")
        print(f"Cause: {cause} ({alert['label']})")
        print(f"Confiance: {alert['confidence']}")
        print(f"Priorite: {alert['priority']}")
        print(f"Contact: {alert['contact']}")
        print(f"Log: {log_text}")
        print("Actions:")
        for action in alert['recommended_actions']:
            print(f"- {action}")
        print(f"Heure: {alert['timestamp']}")
        print(f"{'='*60}\n")
        
        # Sauvegarder
        with open('data/alerts.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(alert, ensure_ascii=True) + '\n')
        
        return alert
    
    def get_severity(self, cause):
        return get_cause_details(cause)['severity']

# Test
if __name__ == "__main__":
    alerter = SimpleAlerter()
    alerter.log_alert('disk_full', 0.95, 'CRITICAL Disk space low on /var: 99% used')
    alerter.log_alert('memory_leak', 0.87, 'WARNING Memory usage at 95%')
