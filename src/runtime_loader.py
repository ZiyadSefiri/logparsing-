"""
Chargement robuste du moteur de prédiction.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from src.hybrid_model import HybridRCA
    from src.rule_based_model import HeuristicRCA
except ImportError:  # pragma: no cover - compatibilite execution directe
    from hybrid_model import HybridRCA
    from rule_based_model import HeuristicRCA

logger = logging.getLogger(__name__)


def _load_metadata_class_names():
    metadata_path = Path("models/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return metadata.get("class_names", [])
    return []


def load_runtime_model():
    """
    Charge le modèle entraîné si possible, sinon un fallback heuristique.
    """
    class_names = _load_metadata_class_names()

    try:
        import torch

        try:
            from src.models import TransformerRCA
        except ImportError:  # pragma: no cover - compatibilite execution directe
            from models import TransformerRCA

        if not class_names:
            class_names = HeuristicRCA.DEFAULT_CLASS_NAMES[:-1]

        model = TransformerRCA(num_classes=len(class_names))
        model_path = Path("models/best_model.pth")
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info("Modele entraine charge depuis %s", model_path)
        else:
            logger.warning("Modele %s introuvable, poids aleatoires utilises", model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        runtime_class_names = list(class_names)
        if "normal_operation" not in runtime_class_names:
            runtime_class_names.append("normal_operation")
        hybrid_model = HybridRCA(model, class_names, runtime_class_names)
        hybrid_model.to(device)
        hybrid_model.eval()
        return hybrid_model, runtime_class_names, device, "hybrid"

    except Exception as exc:
        logger.warning("Bascule vers le classifieur heuristique: %s", exc)

    if "normal_operation" not in class_names:
        class_names = list(class_names) + ["normal_operation"] if class_names else HeuristicRCA.DEFAULT_CLASS_NAMES

    model = HeuristicRCA(class_names=class_names)
    return model, class_names, "cpu", "heuristic"
