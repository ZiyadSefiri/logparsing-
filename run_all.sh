#!/bin/bash

# Script d'exécution complète du projet
# Utilisation: bash run_all.sh

set -e  # Arrêter en cas d'erreur

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Log Monitoring & Root Cause Classification - Exécution      ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Activation de l'environnement virtuel
echo -e "\n🔧 Activation de l'environnement virtuel..."
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "✓ Environnement activé"
else
    echo "⚠️  Dossier venv non trouvé. Installation..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Environnement créé et dépendances installées"
fi

# Vérification des dossiers
echo -e "\n📁 Vérification des dossiers..."
mkdir -p data/raw data/processed models notebooks results logs
echo "✓ Dossiers vérifiés"

# Phase 1: EDA
echo -e "\n" 
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"

python3 notebooks/01_exploratory_analysis.py

if [ $? -eq 0 ]; then
    echo -e "\n✅ Phase 1 terminée avec succès"
else
    echo -e "\n❌ Erreur lors de la Phase 1"
    exit 1
fi

# Phase 2: Entraînement
echo -e "\n" 
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 2: ENTRAÎNEMENT DU MODÈLE                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"

python3 notebooks/02_train_model.py

if [ $? -eq 0 ]; then
    echo -e "\n✅ Phase 2 terminée avec succès"
else
    echo -e "\n❌ Erreur lors de la Phase 2"
    exit 1
fi

# Phase 3: Inférence
echo -e "\n" 
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ PHASE 3: DÉMONSTRATION D'INFÉRENCE                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

python3 notebooks/03_inference_demo.py

if [ $? -eq 0 ]; then
    echo -e "\n✅ Phase 3 terminée avec succès"
else
    echo -e "\n❌ Erreur lors de la Phase 3"
    exit 1
fi

echo -e "\n"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║ 🎉 EXÉCUTION COMPLÈTE TERMINÉE AVEC SUCCÈS                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"

echo -e "\n📊 Résultats:"
echo "  ✓ Graphiques EDA: results/"
echo "  ✓ Modèle entraîné: models/best_model.pth"
echo "  ✓ Métadonnées: models/metadata.json"

echo -e "\n🚀 Prochaines étapes:"
echo "  1. Lancer tout le monitoring Docker: ./start_monitoring.sh docker"
echo "  2. Lancer l'API seule: python3 -m uvicorn src.api_monitoring:app --host 0.0.0.0 --port 8000"
echo "  3. Lancer le dashboard seul: python3 -m streamlit run src/dashboard.py --server.port 8501"
echo "  4. Consulter les résultats dans data/ et results/"

echo -e "\n"
