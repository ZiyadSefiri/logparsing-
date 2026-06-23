"""
Classifieur heuristique pour un mode monitoring robuste hors ligne.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np


class HeuristicRCA:
    """Fallback déterministe pour classer les signatures de logs connues."""

    DEFAULT_CLASS_NAMES = [
        "authentication_error",
        "disk_full",
        "memory_leak",
        "network_failure",
        "resource_exhaustion",
        "normal_operation",
    ]

    def __init__(self, class_names: Iterable[str] | None = None):
        self.class_names = list(class_names or self.DEFAULT_CLASS_NAMES)
        self.patterns = {
            "authentication_error": [
                (r"authentication failed", 0.95),
                (r"\bpam_unix\b", 0.75),
                (r"\bsudo\b", 0.72),
                (r"\bpkexec\b", 0.72),
                (r"permission denied", 0.78),
                (r"session opened", 0.68),
                (r"session closed", 0.64),
                (r"\bauth\b", 0.74),
            ],
            "disk_full": [
                (r"disk space low", 0.98),
                (r"no space left", 0.98),
                (r"\bdisk\b", 0.8),
                (r"\bmount\b", 0.68),
                (r"\binode\b", 0.75),
                (r"\bfilesystem\b", 0.72),
            ],
            "memory_leak": [
                (r"memory leak", 0.98),
                (r"out of memory", 0.97),
                (r"\boom\b", 0.92),
                (r"heap", 0.82),
                (r"memory usage", 0.85),
                (r"\bmemory\b", 0.72),
            ],
            "network_failure": [
                (r"connection timeout", 0.98),
                (r"packet loss", 0.95),
                (r"network interface", 0.94),
                (r"\btimeout\b", 0.84),
                (r"wpa_supplicant", 0.78),
                (r"connection", 0.75),
                (r"\bnetwork\b", 0.76),
                (r"\bdns\b", 0.72),
                (r"\bsocket\b", 0.7),
            ],
            "resource_exhaustion": [
                (r"thread pool exhausted", 0.98),
                (r"tasks pending", 0.92),
                (r"cpu usage", 0.82),
                (r"resource", 0.72),
                (r"too many open files", 0.95),
                (r"max_map_count", 0.8),
                (r"\bperf:\b", 0.7),
                (r"callbacks suppressed", 0.7),
                (r"apparmor", 0.62),
                (r"audit:", 0.58),
            ],
            "normal_operation": [
                (r"completed successfully", 0.86),
                (r"deactivated successfully", 0.84),
                (r"\bstarted\b", 0.76),
                (r"\bfinished\b", 0.78),
                (r"backup completed", 0.86),
                (r"group rekeying completed", 0.8),
                (r"service", 0.55),
            ],
        }

    def eval(self):
        return self

    def to(self, device: str):
        del device
        return self

    def score_text(self, text: str):
        lowered = text.lower()
        scores = {class_name: 0.01 for class_name in self.class_names}

        for class_name, class_patterns in self.patterns.items():
            if class_name not in scores:
                continue
            for pattern, weight in class_patterns:
                if re.search(pattern, lowered):
                    scores[class_name] = max(scores[class_name], weight)

        return scores

    def predict_explicit(self, text: str, device: str = "cpu", min_score: float = 0.55):
        del device

        scores = self.score_text(text)
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]

        if best_score < min_score:
            return None

        raw_scores = np.array([scores[class_name] for class_name in self.class_names], dtype=float)
        probabilities = raw_scores / raw_scores.sum()
        pred_idx = self.class_names.index(best_class)
        confidence = float(probabilities[pred_idx])
        return pred_idx, confidence, probabilities

    def predict(self, text: str, device: str = "cpu"):
        """Retourne (index, confiance, probabilités)."""
        del device

        scores = self.score_text(text)
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]

        if best_score <= 0.05:
            fallback_class = (
                "normal_operation"
                if "normal_operation" in self.class_names
                else self.class_names[-1]
            )
            scores[fallback_class] = 0.6
            best_class = fallback_class
            best_score = scores[fallback_class]

        raw_scores = np.array([scores[class_name] for class_name in self.class_names], dtype=float)
        probabilities = raw_scores / raw_scores.sum()
        pred_idx = self.class_names.index(best_class)
        confidence = float(probabilities[pred_idx])

        return pred_idx, confidence, probabilities
