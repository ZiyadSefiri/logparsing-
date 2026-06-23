"""
Modèles de classification des causes racines
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class TransformerRCA(nn.Module):
    """Modèle Transformer pour Root Cause Analysis"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_classes: int = 5, dropout: float = 0.2):
        super().__init__()

        self.transformer = self._load_transformer(model_name)
        self.tokenizer = self._load_tokenizer(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
        self.num_classes = num_classes
        logger.info(f"TransformerRCA initialisé avec {model_name}")

    @staticmethod
    def _load_transformer(model_name: str):
        try:
            return AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            return AutoModel.from_pretrained(model_name)

    @staticmethod
    def _load_tokenizer(model_name: str):
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            return AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Utiliser le [CLS] token (premier token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits
    
    def predict(self, text: str, device: str = 'cpu'):
        """Prédiction pour un texte"""
        self.eval()
        
        inputs = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        
        probabilities = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probabilities, dim=1)
        confidence = probabilities[0, pred_class].item()
        
        return pred_class.item(), confidence, probabilities[0].cpu().numpy()


class SimpleLSTMRCA(nn.Module):
    """Modèle LSTM simple (baseline)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_classes: int = 5, num_layers: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2, bidirectional=True
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
        logger.info(f"SimpleLSTMRCA initialisé: vocab={vocab_size}, classes={num_classes}")
    
    def forward(self, input_ids, lengths=None):
        """Forward pass"""
        x = self.embedding(input_ids)
        
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Utiliser le dernier hidden state
        last_hidden = h_n[-1]
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        
        return logits


if __name__ == "__main__":
    # Test du modèle Transformer
    print("=== Test TransformerRCA ===")
    model = TransformerRCA(num_classes=5)
    test_text = "ERROR Connection timeout to 192.168.1.100:8080"
    pred, conf, probs = model.predict(test_text)
    print(f"Prédiction: {pred}, Confiance: {conf:.4f}")
    
    # Test du modèle LSTM
    print("\n=== Test SimpleLSTMRCA ===")
    lstm_model = SimpleLSTMRCA(vocab_size=5000, num_classes=5)
    dummy_input = torch.randint(0, 5000, (1, 256))
    output = lstm_model(dummy_input)
    print(f"Output shape: {output.shape}")
