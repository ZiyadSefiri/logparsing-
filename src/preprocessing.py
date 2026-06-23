"""
Module de prétraitement des logs
"""
import re
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LogPreprocessor:
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.vocab = {}
        self.vocab_size = 0
        
    def clean_log(self, log_line: str) -> str:
        """Nettoyage basique des logs"""
        # Suppression des timestamps
        log_line = re.sub(r'\d{4}-\d{2}-\d{2}.*?Z', '', log_line)
        # Suppression des adresses IP
        log_line = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP_ADDR', log_line)
        # Suppression des numéros de fichier/ID
        log_line = re.sub(r'\b[0-9]{5,}\b', 'NUM_ID', log_line)
        # Suppression des espaces excédentaires
        log_line = re.sub(r'\s+', ' ', log_line).strip()
        return log_line.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenisation simple"""
        # Séparation par mots et caractères spéciaux
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def build_vocab(self, logs: List[str], min_freq=2):
        """Construction du vocabulaire"""
        word_freq = {}
        
        for log in logs:
            tokens = self.tokenize(log)
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
        
        # Filtrer par fréquence minimale
        vocab = {word: idx + 2 for idx, (word, freq) in enumerate(word_freq.items()) 
                if freq >= min_freq}
        
        # Ajouter tokens spéciaux
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        logger.info(f"Vocabulaire créé: {self.vocab_size} tokens")
        return vocab
    
    def encode(self, text: str) -> List[int]:
        """Conversion texte -> indices"""
        tokens = self.tokenize(text)
        encoded = []
        
        for token in tokens:
            idx = self.vocab.get(token, self.vocab.get('<UNK>', 1))
            encoded.append(idx)
        
        # Padding/Truncation
        if len(encoded) < self.max_length:
            encoded += [0] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]
        
        return encoded
    
    def process_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Traitement complet d'un dataset"""
        logger.info(f"Traitement de {len(df)} logs...")
        
        # Nettoyage
        df['cleaned'] = df['log'].apply(self.clean_log)
        
        # Encodage
        X = np.array([self.encode(log) for log in df['cleaned']])
        
        # Labels si disponibles
        if 'label' in df.columns:
            y = df['label'].values
        else:
            y = None
        
        logger.info(f"Dataset traité: X.shape={X.shape}")
        return X, y

# Fonction de test
if __name__ == "__main__":
    # Test simple
    preprocessor = LogPreprocessor()
    
    test_logs = [
        "2024-01-15T10:30:45Z ERROR Connection failed to 192.168.1.100:8080",
        "2024-01-15T10:31:20Z WARNING Memory usage at 95% on node-001",
        "ERROR Database connection timeout after 5000ms",
        "INFO Backup completed successfully at 2024-01-15 22:00:00"
    ]
    
    # Construction du vocabulaire
    preprocessor.build_vocab(test_logs)
    
    # Test de nettoyage
    print("=== NETTOYAGE ===")
    for log in test_logs[:2]:
        cleaned = preprocessor.clean_log(log)
        print(f"Original: {log}")
        print(f"Cleaned:  {cleaned}\n")
    
    # Test d'encodage
    print("=== ENCODAGE ===")
    encoded = preprocessor.encode(test_logs[0])
    print(f"Tokens: {encoded[:20]}...")
