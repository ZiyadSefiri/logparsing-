"""
Modele hybride: heuristique forte d'abord, transformer ensuite.
"""
from __future__ import annotations

import numpy as np

try:
    from src.rule_based_model import HeuristicRCA
except ImportError:  # pragma: no cover - compatibilite execution directe
    from rule_based_model import HeuristicRCA


class HybridRCA:
    def __init__(self, primary_model, primary_class_names, class_names):
        self.primary_model = primary_model
        self.primary_class_names = list(primary_class_names)
        self.class_names = list(class_names)
        self.heuristic_model = HeuristicRCA(class_names=self.class_names)

    def eval(self):
        if hasattr(self.primary_model, "eval"):
            self.primary_model.eval()
        self.heuristic_model.eval()
        return self

    def to(self, device: str):
        if hasattr(self.primary_model, "to"):
            self.primary_model.to(device)
        return self

    def predict(self, text: str, device: str = "cpu"):
        heuristic_result = self.heuristic_model.predict_explicit(text, device=device)
        if heuristic_result is not None:
            return heuristic_result

        pred_idx, confidence, probs = self.primary_model.predict(text, device)
        probs = np.asarray(probs, dtype=float)

        if len(self.class_names) == len(self.primary_class_names):
            return pred_idx, confidence, probs

        expanded = np.zeros(len(self.class_names), dtype=float)
        all_indices = {label: index for index, label in enumerate(self.class_names)}

        for index, label in enumerate(self.primary_class_names):
            expanded[all_indices[label]] = float(probs[index])

        predicted_label = self.primary_class_names[pred_idx]
        expanded = expanded / expanded.sum()
        expanded_idx = all_indices[predicted_label]
        return expanded_idx, float(expanded[expanded_idx]), expanded
