"""
Script principal - Point d'entrée du projet.
Utilisation: python3 main.py [option]
"""
import argparse
import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

# Ajouter src au path
sys.path.insert(0, str(SRC_DIR))


def run_script(script_path: Path) -> None:
    """Exécute un script Python par son chemin."""
    runpy.run_path(str(script_path), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(
        description="Log Monitoring & Root Cause Classification"
    )

    parser.add_argument(
        "action",
        choices=["eda", "train", "infer", "api", "stream", "docker-stream", "full"],
        help="Action à effectuer",
    )

    args = parser.parse_args()

    if args.action == "eda":
        print("Exécution de l'analyse exploratoire...")
        run_script(ROOT_DIR / "notebooks/01_exploratory_analysis.py")

    elif args.action == "train":
        print("Exécution du training...")
        run_script(ROOT_DIR / "notebooks/02_train_model.py")

    elif args.action == "infer":
        print("Exécution de l'inférence...")
        run_script(ROOT_DIR / "notebooks/03_inference_demo.py")

    elif args.action == "api":
        print("Démarrage de l'API de monitoring...")
        import uvicorn

        uvicorn.run("src.api_monitoring:app", host="0.0.0.0", port=8000, reload=False)

    elif args.action == "stream":
        print("Exécution du replay streaming sur logs réels...")
        run_script(ROOT_DIR / "src/replay_real_stream.py")

    elif args.action == "docker-stream":
        print("Exécution du streaming sur docker logs...")
        run_script(ROOT_DIR / "src/docker_log_stream.py")

    elif args.action == "full":
        print("Exécution complète du pipeline...")
        run_script(ROOT_DIR / "notebooks/01_exploratory_analysis.py")
        run_script(ROOT_DIR / "notebooks/02_train_model.py")
        run_script(ROOT_DIR / "notebooks/03_inference_demo.py")
        run_script(ROOT_DIR / "src/replay_real_stream.py")


if __name__ == "__main__":
    main()
