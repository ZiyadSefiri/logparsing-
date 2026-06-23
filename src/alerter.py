"""
Système d'alerte automatique
"""
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import requests
import json

class AlertManager:
    def __init__(self, config_file='config/alerts.json'):
        self.config = self.load_config(config_file)
        self.alert_history = []
    
    def load_config(self, config_file):
        """Charge la configuration des alertes"""
        try:
            with open(config_file) as f:
                return json.load(f)
        except:
            return {
                'email': {'enabled': False, 'smtp': '', 'to': ''},
                'slack': {'enabled': False, 'webhook': ''},
                'thresholds': {
                    'network_failure': 0.8,
                    'memory_leak': 0.7,
                    'disk_full': 0.7,
                    'authentication_error': 0.9,
                    'resource_exhaustion': 0.8
                }
            }
    
    def send_alert(self, prediction, confidence, log_text):
        """Envoie une alerte si le seuil est dépassé"""
        cause = prediction
        threshold = self.config['thresholds'].get(cause, 0.8)
        
        if confidence < threshold:
            return False  # Pas assez confiant
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'cause': cause,
            'confidence': confidence,
            'log': log_text,
            'severity': self.get_severity(cause)
        }
        
        self.alert_history.append(alert)
        
        # Email
        if self.config['email']['enabled']:
            self.send_email(alert)
        
        # Slack
        if self.config['slack']['enabled']:
            self.send_slack(alert)
        
        # Console
        print(f"\n🚨 ALERTE {alert['severity']}: {cause}")
        print(f"   Confiance: {confidence:.2%}")
        print(f"   Log: {log_text[:100]}...")
        
        return True
    
    def get_severity(self, cause):
        """Détermine la sévérité"""
        severity_map = {
            'disk_full': 'CRITICAL',
            'memory_leak': 'HIGH',
            'network_failure': 'HIGH',
            'authentication_error': 'MEDIUM',
            'resource_exhaustion': 'MEDIUM'
        }
        return severity_map.get(cause, 'LOW')
    
    def send_slack(self, alert):
        """Envoie alerte Slack"""
        webhook = self.config['slack']['webhook']
        message = {
            'text': f"🚨 {alert['severity']} - {alert['cause']}",
            'attachments': [{
                'color': 'danger' if alert['severity'] == 'CRITICAL' else 'warning',
                'fields': [
                    {'title': 'Confiance', 'value': f"{alert['confidence']:.2%}", 'short': True},
                    {'title': 'Log', 'value': alert['log'][:100], 'short': False}
                ]
            }]
        }
        requests.post(webhook, json=message)
    
    def send_email(self, alert):
        """Envoie alerte Email"""
        # À implémenter avec SMTP
        pass

if __name__ == "__main__":
    # Test
    alerter = AlertManager()
    alerter.send_alert('memory_leak', 0.95, 'WARNING Memory usage at 98%')
