"""
Module d'inférence et prédiction
"""
from contextlib import nullcontext

try:
    import torch
except ImportError:  # pragma: no cover - fallback pour execution sans torch
    class _TorchShim:
        @staticmethod
        def no_grad():
            return nullcontext()

    torch = _TorchShim()

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class RCAInference:
    """Classe pour l'inférence de Root Cause Analysis"""
    
    def __init__(self, model, device='cpu', class_names=None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.class_names = class_names or [f"Class_{i}" for i in range(5)]
    
    def predict_single(self, text: str, threshold: float = 0.5) -> Dict:
        """Prédiction pour un seul log"""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                pred_class, confidence, probs = self.model.predict(text, self.device)

                result = {
                    'text': text,
                    'predicted_class': self.class_names[pred_class],
                    'confidence': float(confidence),
                    'probabilities': {
                        self.class_names[i]: float(probs[i])
                        for i in range(len(self.class_names))
                    },
                    'is_confident': confidence >= threshold
                }
            else:
                result = {
                    'text': text,
                    'error': 'Model does not expose a predict() method'
                }
        
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Prédictions batch"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                result = self.predict_single(text)
                results.append(result)
        
        return results
    
    def get_top_k_predictions(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retourne les top-k prédictions"""
        self.model.eval()
        
        with torch.no_grad():
            pred_class, confidence, probs = self.model.predict(text, self.device)
        
        top_k_indices = np.argsort(probs)[::-1][:k]
        top_k = [
            (self.class_names[idx], float(probs[idx]))
            for idx in top_k_indices
        ]
        
        return top_k
    
    def explain_prediction(self, text: str) -> Dict:
        """Explication détaillée de la prédiction"""
        self.model.eval()
        
        with torch.no_grad():
            pred_class, confidence, probs = self.model.predict(text, self.device)
        
        # Extraction de l'attention si disponible
        attention_scores = None
        if hasattr(self.model, 'transformer') and hasattr(self.model, 'tokenizer'):
            inputs = self.model.tokenizer(
                text, max_length=256, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            outputs = self.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            attention_scores = outputs.attentions
        
        explanation = {
            'text': text,
            'predicted_class': self.class_names[pred_class],
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            'has_attention_weights': attention_scores is not None
        }
        
        return explanation

if __name__ == "__main__":
    print("Module d'inférence importable")
