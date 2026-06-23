"""
Dashboard Streamlit pour le monitoring
"""
import streamlit as st
import pandas as pd
import time
import json
from pathlib import Path

st.set_page_config(page_title="Log Monitoring", layout="wide")

st.title("🔍 Dashboard de Monitoring - Root Cause Analysis")

# Sidebar
st.sidebar.header("Configuration")
refresh_rate = st.sidebar.slider("Rafraîchissement (secondes)", 1, 60, 1)
show_details = st.sidebar.checkbox("Afficher les détails", True)
show_raw_logs = st.sidebar.checkbox("Afficher les logs bruts", True)
rows_to_show = st.sidebar.slider("Nombre de lignes", 10, 100, 25)

def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


predictions_file = Path("data/stream_predictions.jsonl")
alerts_file = Path("data/alerts.jsonl")
raw_logs_file = Path("data/raw/real_logs.csv")
docker_raw_logs_file = Path("data/raw/docker_logs.csv")
performance_file = Path("data/performance.json")
summary_file = Path("results/realtime_summary.json")

pred_df = load_jsonl(predictions_file)
alerts_df = load_jsonl(alerts_file)

if not pred_df.empty and "processed_at" in pred_df.columns:
    pred_df["processed_at"] = pd.to_datetime(pred_df["processed_at"])
if not pred_df.empty and "event_timestamp" in pred_df.columns:
    pred_df["event_timestamp"] = pd.to_datetime(pred_df["event_timestamp"], errors="coerce")

if not alerts_df.empty and "timestamp" in alerts_df.columns:
    alerts_df["timestamp"] = pd.to_datetime(alerts_df["timestamp"], errors="coerce")

if not alerts_df.empty and "recommended_actions" in alerts_df.columns:
    alerts_df["recommended_actions"] = alerts_df["recommended_actions"].apply(
        lambda actions: " | ".join(actions) if isinstance(actions, list) else actions
    )

raw_frames = []
for candidate in [docker_raw_logs_file, raw_logs_file]:
    if candidate.exists():
        frame = pd.read_csv(candidate)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["raw_source_file"] = str(candidate)
        raw_frames.append(frame)

if raw_frames:
    raw_df = pd.concat(raw_frames, ignore_index=True)
else:
    raw_df = pd.DataFrame(columns=["timestamp", "source", "log"])

if performance_file.exists():
    with open(performance_file, "r", encoding="utf-8") as handle:
        performance_stats = json.load(handle)
else:
    performance_stats = {
        "total_predictions": 0,
        "alerts_triggered": 0,
        "predictions_by_cause": {},
    }

if summary_file.exists():
    with open(summary_file, "r", encoding="utf-8") as handle:
        runtime_summary = json.load(handle)
else:
    runtime_summary = {}

source_type = runtime_summary.get("source_type", "unknown")
container_list = runtime_summary.get("containers", [])

st.caption(
    f"Source active: `{source_type}`"
    + (f" | Conteneurs: `{', '.join(container_list)}`" if container_list else "")
)

# Métriques principales
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Logs analysés", performance_stats.get("total_predictions", len(pred_df)))
with col2:
    critical_alerts = 0 if alerts_df.empty else len(
        alerts_df[alerts_df["severity"].str.contains("CRITICAL", na=False)]
    )
    st.metric("Alertes critiques", critical_alerts)
with col3:
    total_alerts = int(performance_stats.get("alerts_triggered", len(alerts_df)))
    alert_ratio = total_alerts / max(performance_stats.get("total_predictions", len(pred_df)), 1) * 100
    st.metric("Taux d'alertes", f"{alert_ratio:.1f}%")
with col4:
    confident_count = 0 if pred_df.empty else int((pred_df["confidence"] >= 0.7).sum())
    ratio = confident_count / max(len(pred_df), 1) * 100
    st.metric("Prédictions confiantes", f"{ratio:.1f}%")
with col5:
    if not pred_df.empty:
        avg_latency = pred_df["latency_ms"].mean() if "latency_ms" in pred_df.columns else 0
        st.metric("Latence moyenne", f"{avg_latency:.0f} ms")
    else:
        st.metric("Latence moyenne", "N/A")

if not pred_df.empty:
    st.caption(f"Dernière analyse: `{pred_df['processed_at'].max().strftime('%H:%M:%S')}`")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Répartition des prédictions")
    if performance_stats.get("predictions_by_cause"):
        prediction_counts = pd.Series(performance_stats["predictions_by_cause"]).sort_values(ascending=False)
        st.bar_chart(prediction_counts)
    else:
        st.info("Aucune prédiction disponible pour le moment")

with chart_col2:
    st.subheader("⏱️ Activité par seconde")
    if not pred_df.empty:
        per_second = (
            pred_df.sort_values("processed_at")
            .assign(second=lambda df: df["processed_at"].dt.floor("s"))
            .groupby("second")
            .size()
            .tail(30)
        )
        st.line_chart(per_second)
    else:
        st.info("Aucune activité disponible")

st.subheader("📉 Confiance des dernières analyses")
if not pred_df.empty:
    chart_df = pred_df.sort_values("processed_at").tail(50).set_index("processed_at")[["confidence"]]
    st.line_chart(chart_df)
else:
    st.info("Courbe indisponible")

table_col1, table_col2 = st.columns(2)

with table_col1:
    st.subheader("📋 Logs traités en temps réel")
    if not pred_df.empty:
        display_df = pred_df.sort_values("processed_at", ascending=False).head(rows_to_show)
        columns = [
            "processed_at",
            "source",
            "prediction",
            "severity",
            "priority",
            "confidence",
            "latency_ms",
            "alert_triggered",
            "log",
        ]
        visible_columns = [column for column in columns if column in display_df.columns]
        st.dataframe(
            display_df[visible_columns],
            use_container_width=True,
        )
    else:
        st.info("Aucune prédiction enregistrée pour le moment")

with table_col2:
    st.subheader("🚨 Alertes récentes")
    if not alerts_df.empty:
        alert_columns = [
            "timestamp",
            "severity",
            "priority",
            "cause",
            "confidence",
            "contact",
            "recommended_actions",
            "log",
        ]
        visible_alert_columns = [column for column in alert_columns if column in alerts_df.columns]
        st.dataframe(
            alerts_df.sort_values("timestamp", ascending=False).head(rows_to_show)[visible_alert_columns],
            use_container_width=True,
        )
    else:
        st.info("Aucune alerte enregistrée")

if show_details and not pred_df.empty:
    st.subheader("🧠 Nettoyage et explication")
    explain_df = pred_df.sort_values("processed_at", ascending=False).head(rows_to_show)
    explain_columns = [
        "processed_at",
        "prediction",
        "cleaned_log",
        "recommended_actions",
        "contact",
    ]
    visible_explain_columns = [column for column in explain_columns if column in explain_df.columns]
    if "recommended_actions" in explain_df.columns:
        explain_df = explain_df.copy()
        explain_df["recommended_actions"] = explain_df["recommended_actions"].apply(
            lambda actions: " | ".join(actions) if isinstance(actions, list) else actions
        )
    st.dataframe(explain_df[visible_explain_columns], use_container_width=True)

if show_details and show_raw_logs:
    st.subheader("🧾 Logs bruts extraits")
    if not raw_df.empty:
        raw_display = raw_df.sort_values("timestamp", ascending=False).head(rows_to_show)
        visible_columns = [col for col in ["timestamp", "source", "container", "image", "log"] if col in raw_display.columns]
        st.dataframe(raw_display[visible_columns], use_container_width=True)
    else:
        st.info("Aucun log brut disponible")

# Auto-refresh
st.write(f"⏱️ Rafraîchissement automatique toutes les {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()
