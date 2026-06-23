"""
Générateur de données de test pour le projet
"""
import pandas as pd
import numpy as np
import random
from typing import Tuple

class LogDataGenerator:
    """Générateur de logs synthétiques pour tests"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.log_templates = [
            "ERROR Connection timeout to {ip}:{port}",
            "WARNING Memory usage at {percent}% on {node}",
            "INFO Process {pid} started successfully",
            "CRITICAL Disk space low on {mount}: {percent}% used",
            "ERROR Authentication failed for user {user}",
            "WARNING High CPU usage detected: {cpu}%",
            "INFO Database backup completed in {time}ms",
            "ERROR Thread pool exhausted, {pending} tasks pending",
            "WARNING Cache miss rate increased to {percent}%",
            "ERROR Network interface {iface} is down",
        ]
        
        self.root_causes = [
            "network_failure",
            "memory_leak",
            "disk_full",
            "authentication_error",
            "resource_exhaustion"
        ]
    
    def generate_log(self, root_cause: str) -> str:
        """Génère un log basé sur une cause racine"""
        if root_cause == "network_failure":
            template = random.choice([
                "ERROR Connection timeout to {ip}:{port}",
                "ERROR Network interface {iface} is down",
                "WARNING Packet loss detected: {percent}%"
            ])
            return template.format(
                ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                port=random.choice([80, 443, 8080, 5432]),
                iface=f"eth{random.randint(0,3)}",
                percent=random.randint(10, 100)
            )
        
        elif root_cause == "memory_leak":
            template = random.choice([
                "WARNING Memory usage at {percent}% on {node}",
                "CRITICAL Out of memory error on {node}",
                "WARNING Heap size exceeds {size}MB"
            ])
            return template.format(
                percent=random.randint(80, 99),
                node=f"node-{random.randint(1,10):03d}",
                size=random.randint(4000, 8000)
            )
        
        elif root_cause == "disk_full":
            return f"CRITICAL Disk space low on /var: {random.randint(90, 99)}% used"
        
        elif root_cause == "authentication_error":
            return f"ERROR Authentication failed for user {random.choice(['admin', 'webapp', 'db_user'])}"
        
        elif root_cause == "resource_exhaustion":
            return f"ERROR Thread pool exhausted, {random.randint(100, 1000)} tasks pending"
        
        return random.choice(self.log_templates)
    
    def generate_dataset(self, n_samples: int = 1000, 
                         distribution: dict = None) -> pd.DataFrame:
        """
        Génère un dataset complet
        
        Args:
            n_samples: nombre total de logs
            distribution: dict avec proportion par cause racine
                Ex: {'network_failure': 0.3, 'memory_leak': 0.3, ...}
        """
        if distribution is None:
            distribution = {cause: 1/len(self.root_causes) 
                          for cause in self.root_causes}
        
        logs = []
        labels = []
        
        for cause, prop in distribution.items():
            n = int(n_samples * prop)
            for _ in range(n):
                logs.append(self.generate_log(cause))
                labels.append(cause)
        
        # Compléter si nécessaire
        while len(logs) < n_samples:
            cause = random.choice(self.root_causes)
            logs.append(self.generate_log(cause))
            labels.append(cause)
        
        df = pd.DataFrame({
            'log': logs,
            'label': labels
        })
        
        # Mélanger
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

if __name__ == "__main__":
    gen = LogDataGenerator()
    df = gen.generate_dataset(n_samples=100)
    print(df.head(10))
    print(f"\nDistribution des causes:")
    print(df['label'].value_counts())
