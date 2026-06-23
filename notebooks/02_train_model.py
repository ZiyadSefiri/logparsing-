"""
Transformer model training script (enhanced version)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

from src.models import TransformerRCA
from src.training import Trainer
from src.preprocessing import LogPreprocessor

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

BANNER = "=" * 76
print(BANNER)
print("TRANSFORMER TRAINING — IMPROVED CONFIGURATION")
print(BANNER)

# hyper-parameters
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BS = 8
N_EPOCHS = 15
LR = 1e-5
WARMUP = 1000

print(f"\n[Config]")
print(f"  Device:      {DEV}")
print(f"  Batch size:  {BS}")
print(f"  Epochs:      {N_EPOCHS}")
print(f"  LR:          {LR}")

# ---- 1. Load ----
print(f"\n[1] LOADING DATA...")
data = pd.read_csv('data/raw/synthetic_logs.csv')
print(f"  Loaded {len(data)} records")
print(f"  Distribution:\n{data['label'].value_counts()}")

# ---- 2. Encode labels ----
encoder = LabelEncoder()
data['y'] = encoder.fit_transform(data['label'])
classes = encoder.classes_.tolist()

print(f"  Classes ({len(classes)}):")
for i, c in enumerate(classes):
    print(f"    [{i}] {c}  ({len(data[data['label'] == c])}x)")

# ---- 3. Preprocess ----
print(f"\n[2] PREPROCESSING...")
pp = LogPreprocessor(max_length=256)
pp.build_vocab(data['log'].tolist(), min_freq=1)

X = np.array([pp.encode(txt) for txt in data['log']])
y = data['y'].values

# ---- 4. Train / val / test split ----
idx = np.arange(len(data))
tr_idx, tmp_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
vl_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, random_state=42, stratify=y[tmp_idx])

X_tr, X_vl, X_te = X[tr_idx], X[vl_idx], X[te_idx]
y_tr, y_vl, y_te = y[tr_idx], y[vl_idx], y[te_idx]

print(f"  Train: {len(X_tr)}  Val: {len(X_vl)}  Test: {len(X_te)}")

# ---- 5. Dataloaders ----
print(f"\n[3] BUILDING DATALOADERS...")
train_ds = TensorDataset(
    torch.tensor(X_tr, dtype=torch.long),
    torch.zeros_like(torch.tensor(X_tr, dtype=torch.long)),
    torch.tensor(y_tr, dtype=torch.long)
)
val_ds = TensorDataset(
    torch.tensor(X_vl, dtype=torch.long),
    torch.zeros_like(torch.tensor(X_vl, dtype=torch.long)),
    torch.tensor(y_vl, dtype=torch.long)
)

train_ld = DataLoader(train_ds, batch_size=BS, shuffle=True)
val_ld = DataLoader(val_ds, batch_size=BS, shuffle=False)

print(f"  Train batches: {len(train_ld)}")
print(f"  Val batches:   {len(val_ld)}")

# ---- 6. Model ----
print(f"\n[4] INITIALIZING MODEL...")
model = TransformerRCA(model_name="distilbert-base-uncased", num_classes=len(classes), dropout=0.3)
model.to(DEV)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params:,}")

# ---- 7. Train ----
print(f"\n[5] TRAINING...")
trainer = Trainer(model, device=DEV, output_dir='models')
history = trainer.train(train_ld, val_ld, epochs=N_EPOCHS, learning_rate=LR, warmup_steps=WARMUP)

# ---- 8. Plot results ----
print(f"\n[6] SAVING CHARTS...")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(history['train_loss'], 'o-', color='coral', label='Train')
ax1.plot(history['val_loss'], 'o-', color='teal', label='Validation')
ax1.set_title("Loss per epoch", weight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid(alpha=0.25)

ax2.plot(history['val_accuracy'], 'o-', color='coral', label='Accuracy')
ax2.plot(history['val_f1'], 'o-', color='teal', label='F1')
ax2.set_title("Validation metrics", weight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
ax2.legend(); ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('results/03_training_history.png', dpi=150, bbox_inches='tight')
print("  Saved → results/03_training_history.png")

# ---- 9. Metadata ----
print(f"\n[7] PERSISTING METADATA...")
import json

meta = {
    'model_type': 'TransformerRCA',
    'backbone': 'distilbert-base-uncased',
    'n_classes': len(classes),
    'class_labels': classes,
    'vocab_size': pp.vocab_size,
    'max_seq_len': pp.max_length,
    'epochs': N_EPOCHS,
    'final': {
        'train_loss': float(history['train_loss'][-1]),
        'val_loss': float(history['val_loss'][-1]),
        'val_acc': float(history['val_accuracy'][-1]),
        'val_f1': float(history['val_f1'][-1])
    }
}

with open('models/metadata.json', 'w') as fp:
    json.dump(meta, fp, indent=2)
print("  Saved → models/metadata.json")

print(f"\n{BANNER}")
print("DONE — TRAINING COMPLETE")
print(BANNER)
