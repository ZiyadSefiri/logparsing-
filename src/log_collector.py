"""
Collecte les vrais logs du système en temps réel
"""
import subprocess
import time
from datetime import datetime
from pathlib import Path

class RealLogCollector:
    def __init__(self, output_file='data/raw/real_logs.csv'):
        self.output_file = Path(output_file)
        self.sources = [
            # Logs système Linux
            '/var/log/syslog',
            '/var/log/auth.log',
            '/var/log/kern.log',
            # Logs applicatifs (à adapter)
            '/var/log/nginx/error.log',
            '/var/log/apache2/error.log',
        ]
    
    def collect_tail(self, log_file, duration=60):
        """Collecte les logs en temps réel avec tail -f"""
        try:
            process = subprocess.Popen(
                ['tail', '-f', log_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            start_time = time.time()
            logs = []
            
            while time.time() - start_time < duration:
                line = process.stdout.readline()
                if line:
                    logs.append({
                        'timestamp': datetime.now().isoformat(),
                        'source': log_file,
                        'log': line.strip()
                    })
            
            process.terminate()
            return logs
            
        except Exception as e:
            print(f"Erreur sur {log_file}: {e}")
            return []
    
    def collect_all(self, duration=30):
        """Collecte depuis toutes les sources"""
        all_logs = []
        
        for source in self.sources:
            if Path(source).exists():
                print(f"📡 Collecte: {source}")
                logs = self.collect_tail(source, duration=duration)
                all_logs.extend(logs)
        
        # Sauvegarder
        import pandas as pd
        df = pd.DataFrame(all_logs)
        df.to_csv(self.output_file, index=False)
        print(f"✓ {len(all_logs)} logs sauvegardés dans {self.output_file}")
        
        return df

if __name__ == "__main__":
    collector = RealLogCollector()
    collector.collect_all(duration=30)
