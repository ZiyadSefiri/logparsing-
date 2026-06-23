"""
Rejoue les logs reels existants comme un flux de streaming.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

try:
    from src.monitoring_pipeline import MonitoringPipeline
except ImportError:  # pragma: no cover - compatibilite execution directe
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.monitoring_pipeline import MonitoringPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Replay streaming de logs reels")
    parser.add_argument(
        "--input",
        default="data/raw/real_logs.csv",
        help="Fichier CSV source contenant des logs reels",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delai entre deux logs pour simuler le streaming",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Nombre maximal de logs a rejouer (0 = tous)",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Reinitialise les fichiers de sorties avant execution",
    )
    return parser.parse_args()


def load_rows(csv_path: Path, limit: int = 0):
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return rows[:limit] if limit else rows


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {input_path}")

    pipeline = MonitoringPipeline()
    if args.reset_output:
        pipeline.reset_outputs()

    rows = load_rows(input_path, args.limit)
    print("=" * 80)
    print("REPLAY STREAMING DES LOGS REELS")
    print("=" * 80)
    print(f"Source: {input_path}")
    print(f"Nombre de logs a rejouer: {len(rows)}")
    print(f"Moteur utilise: {pipeline.backend}")

    predictions = Counter()
    alerts = 0

    for index, row in enumerate(rows, start=1):
        result = pipeline.process_log(
            row.get("log", ""),
            source=row.get("source"),
            event_timestamp=row.get("timestamp"),
        )
        predictions[result["prediction"]] += 1
        alerts += int(result["alert_triggered"])

        print(
            f"[{index:03d}/{len(rows):03d}] "
            f"{result['prediction']:<22} "
            f"{result['confidence_percent']:>8} "
            f"alert={'oui' if result['alert_triggered'] else 'non'}"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    summary = {
        "input_file": str(input_path),
        "processed_logs": len(rows),
        "alerts_triggered": alerts,
        "backend": pipeline.backend,
        "prediction_distribution": dict(predictions),
        "performance_stats": pipeline.tracker.get_stats(),
    }

    summary_path = Path("results/realtime_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n" + "=" * 80)
    print("RESUME")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"\n✓ Resume sauvegarde dans {summary_path}")


if __name__ == "__main__":
    main()
