"""
Configuration du logging global
"""
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Configure le logging pour tout le projet"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Utiliser partout:
# from logger_config import setup_logging
# logger = setup_logging()
# logger.info("Message important")
