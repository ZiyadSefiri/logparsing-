"""
Script d'entraînement des modèles
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Trainer:
    """Classe d'entraînement"""
    
    def __init__(self, model, device='cpu', output_dir='./models'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def train_epoch(self, train_loader, optimizer, scheduler, criterion):
        """Entraîne une époque"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device) if len(batch) > 2 else None
            labels = batch[-1].to(self.device)
            
            optimizer.zero_grad()
            
            if attention_mask is not None:
                logits = self.model(input_ids, attention_mask)
            else:
                logits = self.model(input_ids)
            
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def evaluate(self, val_loader, criterion):
        """Évaluation sur le set de validation"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device) if len(batch) > 2 else None
                labels = batch[-1].to(self.device)
                
                if attention_mask is not None:
                    logits = self.model(input_ids, attention_mask)
                else:
                    logits = self.model(input_ids)
                
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=2e-5, 
              warmup_steps=500):
        """Entraînement complet"""
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        for epoch in range(epochs):
            logger.info(f"\n=== ÉPOQUE {epoch+1}/{epochs} ===")
            
            train_loss = self.train_epoch(train_loader, optimizer, scheduler, criterion)
            self.history['train_loss'].append(train_loss)
            
            val_metrics, preds, labels = self.evaluate(val_loader, criterion)
            
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"Val F1: {val_metrics['f1']:.4f}")
            
            # Sauvegarder le meilleur modèle
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model('best_model')
                logger.info(f"✓ Modèle sauvegardé (F1: {best_f1:.4f})")
        
        return self.history
    
    def save_model(self, name='model'):
        """Sauvegarde le modèle"""
        path = self.output_dir / f"{name}.pth"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Modèle sauvegardé: {path}")
    
    def load_model(self, name='best_model'):
        """Charge un modèle"""
        path = self.output_dir / f"{name}.pth"
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Modèle chargé: {path}")

if __name__ == "__main__":
    print("Module de training importable")
