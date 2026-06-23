"""
Suivi des performances du modèle
"""
import json
from datetime import datetime
from pathlib import Path

class PerformanceTracker:
    def __init__(self, file='data/performance.json'):
        self.file = Path(file)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.stats = self.load_stats()
    
    def load_stats(self):
        if self.file.exists():
            with open(self.file) as f:
                return json.load(f)
        return {
            'total_predictions': 0,
            'alerts_triggered': 0,
            'accuracy_estimate': 0,
            'predictions_by_cause': {}
        }
    
    def record_prediction(self, cause, confidence, alert_triggered=None):
        """Enregistre une prédiction"""
        self.stats['total_predictions'] += 1
        
        if cause not in self.stats['predictions_by_cause']:
            self.stats['predictions_by_cause'][cause] = 0
        self.stats['predictions_by_cause'][cause] += 1
        
        if alert_triggered is None:
            alert_triggered = confidence > 0.7

        if alert_triggered:
            self.stats['alerts_triggered'] += 1
        
        self.save()
    
    def save(self):
        with open(self.file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def get_stats(self):
        return self.stats

# Test
if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.record_prediction('disk_full', 0.95)
    tracker.record_prediction('memory_leak', 0.87)
    print(tracker.get_stats())
