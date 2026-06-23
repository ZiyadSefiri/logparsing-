#!/bin/bash

echo "╔══════════════════════════════════════════════════════════╗"
echo "║        DÉMARRAGE DU SYSTÈME DE MONITORING               ║"
echo "╚══════════════════════════════════════════════════════════╝"

MODE="${1:-docker}"

# Terminal 1: API
echo "🚀 Démarrage de l'API..."
python3 -m uvicorn src.api_monitoring:app --host 0.0.0.0 --port 8000 &
API_PID=$!

COLLECTOR_CMD=()
if [ "$MODE" = "docker" ]; then
    echo "📡 Démarrage du streaming Docker..."
    DOCKER_ARGS=(--reset-output --tail "${DOCKER_TAIL:-100}")
    if [ -n "${DOCKER_CONTAINERS:-}" ]; then
        IFS=',' read -r -a CONTAINERS <<< "$DOCKER_CONTAINERS"
        for name in "${CONTAINERS[@]}"; do
            DOCKER_ARGS+=(--container "$name")
        done
    else
        DOCKER_ARGS+=(--all-running)
    fi
    COLLECTOR_CMD=(python3 -m src.docker_log_stream "${DOCKER_ARGS[@]}")
else
    echo "📡 Démarrage du replay streaming..."
    COLLECTOR_CMD=(python3 -m src.replay_real_stream --reset-output --delay 0.2)
fi

"${COLLECTOR_CMD[@]}" &
COLLECTOR_PID=$!

# Terminal 3: Dashboard
echo "📊 Démarrage du dashboard..."
python3 -m streamlit run src/dashboard.py --server.port 8501 &
DASHBOARD_PID=$!

echo ""
echo "✅ Système démarré!"
echo ""
echo "📍 API: http://localhost:8000"
echo "📍 Dashboard: http://localhost:8501"
echo "📍 Mode logs: $MODE"
echo ""
echo "Pour arrêter: kill $API_PID $COLLECTOR_PID $DASHBOARD_PID"

wait
