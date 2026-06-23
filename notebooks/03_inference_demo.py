"""
Inference demo — classify real-time log messages
Usage:  python notebooks/03_inference_demo.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from src.models import TransformerRCA
from src.inference import RCAInference

HR = "=" * 74
print(HR)
print("INFERENCE DEMO — ROOT CAUSE CLASSIFICATION")
print(HR)

# settings
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT = 'models/best_model.pth'
META_FILE = 'models/metadata.json'

# ---- 1. Load metadata ----
print(f"\n[Stage 1] Loading metadata...")
if Path(META_FILE).exists():
    with open(META_FILE) as f:
        meta = json.load(f)
    labels = meta['class_labels']
    print(f"  OK — {len(labels)} classes found")
else:
    labels = ['network_failure', 'memory_leak', 'disk_full',
              'authentication_error', 'resource_exhaustion']
    print(f"  WARNING — using default classes")

# ---- 2. Load model ----
print(f"\n[Stage 2] Loading model weights...")
net = TransformerRCA(num_classes=len(labels))
net.to(DEV)

if Path(CKPT).exists():
    net.load_state_dict(torch.load(CKPT, map_location=DEV))
    print(f"  OK — weights loaded from {CKPT}")
else:
    print(f"  WARNING — {CKPT} not found, using untrained model")

# ---- 3. Build inference engine ----
print(f"\n[Stage 3] Initialising inference engine...")
engine = RCAInference(net, device=DEV, class_names=labels)
print(f"  Ready  (device={DEV})")

# ---- 4. Run predictions ----
print(f"\n[Stage 4] Evaluating sample logs...")
samples = [
    "ERROR Connection timeout to 192.168.1.100:8080",
    "WARNING Memory usage at 95% on node-001",
    "CRITICAL Disk space low on /var: 99% used",
    "ERROR Authentication failed for user admin",
    "ERROR Thread pool exhausted, 500 tasks pending"
]

for idx, msg in enumerate(samples, 1):
    print(f"\n  ── Log #{idx} ──")
    print(f"  Text:  {msg}")

    out = engine.predict_single(msg, threshold=0.5)
    print(f"  Top:   {out['predicted_class']}")
    print(f"  Conf:  {out['confidence']:.2%}")

    top3 = engine.get_top_k_predictions(msg, k=3)
    print(f"  Top-3:")
    for rank, (cl, prob) in enumerate(top3, 1):
        print(f"    {rank}. {cl:<28} {prob:.2%}")

# ---- 5. Detailed explanation ----
print(f"\n[Stage 5] Detailed explanation for first sample...")
first = samples[0]
detail = engine.explain_prediction(first)

print(f"  Log:  {first}")
print(f"  Pred: {detail['predicted_class']}  (conf={detail['confidence']:.2%})")
print(f"  Probabilities:")
for cl, prob in detail['probabilities'].items():
    bar = "▓" * int(prob * 50)
    print(f"    {cl:<28} {prob:5.1%}  {bar}")

print(f"\n{HR}")
print("DONE")
print(HR)
