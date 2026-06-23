"""
Collecte et analyse des logs Docker en temps reel.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from src.monitoring_pipeline import MonitoringPipeline
except ImportError:  # pragma: no cover - compatibilite execution directe
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.monitoring_pipeline import MonitoringPipeline


@dataclass
class ContainerInfo:
    name: str
    image: str = "unknown"
    status: str = "unknown"


class DockerLogFollower:
    def __init__(self, container: ContainerInfo, output_queue: queue.Queue, tail: int):
        self.container = container
        self.output_queue = output_queue
        self.tail = tail
        self.process = None
        self.thread = None

    def start(self):
        cmd = [
            "docker",
            "logs",
            "-f",
            "--timestamps",
            "--tail",
            str(self.tail),
            self.container.name,
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        assert self.process is not None
        assert self.process.stdout is not None

        try:
            for line in self.process.stdout:
                clean_line = line.rstrip("\n")
                if clean_line:
                    self.output_queue.put(
                        {
                            "container": self.container.name,
                            "image": self.container.image,
                            "status": self.container.status,
                            "line": clean_line,
                        }
                    )
        finally:
            self.output_queue.put(
                {
                    "container": self.container.name,
                    "image": self.container.image,
                    "status": self.container.status,
                    "event": "stream_closed",
                    "returncode": self.process.poll(),
                }
            )

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming d'analyse sur docker logs")
    parser.add_argument(
        "--container",
        action="append",
        default=[],
        help="Nom d'un conteneur a suivre. Option repetable.",
    )
    parser.add_argument(
        "--all-running",
        action="store_true",
        help="Suit tous les conteneurs Docker en cours d'execution.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Nombre de lignes historiques a recuperer par conteneur avant le suivi.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duree maximale du suivi en secondes (0 = jusqu'a interruption).",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Reinitialise les fichiers de sorties avant execution.",
    )
    parser.add_argument(
        "--raw-output",
        default="data/raw/docker_logs.csv",
        help="Fichier CSV ou sauvegarder les logs Docker bruts.",
    )
    return parser.parse_args()


def list_running_containers() -> dict[str, ContainerInfo]:
    cmd = ["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)

    containers: dict[str, ContainerInfo] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name = parts[0]
        image = parts[1] if len(parts) > 1 else "unknown"
        status = parts[2] if len(parts) > 2 else "unknown"
        containers[name] = ContainerInfo(name=name, image=image, status=status)
    return containers


def resolve_containers(args) -> list[ContainerInfo]:
    running = list_running_containers()

    if args.container:
        resolved = []
        for name in args.container:
            resolved.append(running.get(name, ContainerInfo(name=name)))
        return resolved

    if args.all_running:
        return list(running.values())

    return list(running.values())


def ensure_raw_writer(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(csv_path, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=["timestamp", "source", "container", "image", "status", "log"],
    )
    writer.writeheader()
    return handle, writer


def split_docker_timestamp(raw_line: str):
    timestamp, separator, message = raw_line.partition(" ")
    if separator:
        return timestamp, message
    return datetime.now(timezone.utc).isoformat(), raw_line


def write_summary(summary: dict):
    summary_path = Path("results/realtime_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path


def main():
    args = parse_args()
    containers = resolve_containers(args)
    if not containers:
        raise RuntimeError("Aucun conteneur disponible pour docker logs.")

    pipeline = MonitoringPipeline()
    if args.reset_output:
        pipeline.reset_outputs()

    raw_output_path = Path(args.raw_output)
    raw_handle, raw_writer = ensure_raw_writer(raw_output_path)

    event_queue: queue.Queue = queue.Queue()
    followers = [DockerLogFollower(container, event_queue, tail=args.tail) for container in containers]

    for follower in followers:
        follower.start()

    print("=" * 80)
    print("STREAMING DOCKER LOGS")
    print("=" * 80)
    print(f"Conteneurs suivis: {', '.join(container.name for container in containers)}")
    print(f"Historique initial par conteneur: {args.tail} lignes")
    print(f"Moteur utilise: {pipeline.backend}")
    if args.duration:
        print(f"Duree maximale: {args.duration}s")

    predictions = Counter()
    alerts = 0
    processed_logs = 0
    deadline = time.time() + args.duration if args.duration else None
    active_streams = {container.name for container in containers}

    try:
        while active_streams:
            if deadline and time.time() >= deadline:
                print("\n⏹️  Duree atteinte, arret du suivi Docker.")
                break

            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event.get("event") == "stream_closed":
                active_streams.discard(event["container"])
                continue

            event_timestamp, log_message = split_docker_timestamp(event["line"])
            raw_writer.writerow(
                {
                    "timestamp": event_timestamp,
                    "source": f"docker:{event['container']}",
                    "container": event["container"],
                    "image": event["image"],
                    "status": event["status"],
                    "log": log_message,
                }
            )
            raw_handle.flush()

            result = pipeline.process_log(
                log_message,
                source=f"docker:{event['container']}",
                event_timestamp=event_timestamp,
            )
            processed_logs += 1
            predictions[result["prediction"]] += 1
            alerts += int(result["alert_triggered"])

            print(
                f"[{event['container']}] "
                f"{result['prediction']:<22} "
                f"{result['confidence_percent']:>8} "
                f"alert={'oui' if result['alert_triggered'] else 'non'}"
            )

    except KeyboardInterrupt:
        print("\n⏹️  Interruption utilisateur, arret du suivi Docker.")
    finally:
        for follower in followers:
            follower.stop()
        raw_handle.close()

    summary = {
        "source_type": "docker_logs",
        "containers": [container.name for container in containers],
        "processed_logs": processed_logs,
        "alerts_triggered": alerts,
        "backend": pipeline.backend,
        "prediction_distribution": dict(predictions),
        "performance_stats": pipeline.tracker.get_stats(),
        "raw_output_file": str(raw_output_path),
    }
    summary_path = write_summary(summary)

    print("\n" + "=" * 80)
    print("RESUME")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print(f"\n✓ Resume sauvegarde dans {summary_path}")


if __name__ == "__main__":
    main()
