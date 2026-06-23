# 🔍 Système Intelligent de Monitoring des Logs et Root Cause Analysis

Système de détection automatique d'anomalies système et d'analyse de cause racine (Root Cause Analysis) basé sur du Deep Learning (DistilBERT), intégré dans une architecture de monitoring temps réel.

---

## 📋 Description

Les infrastructures modernes génèrent un volume massif de logs, rendant impossible toute analyse manuelle efficace. Ce projet propose une solution intelligente capable de :

- Détecter automatiquement les anomalies système critiques
- Classifier les incidents par type
- Identifier la cause racine d'un problème
- Déclencher des alertes en temps réel

---

## 🏗️ Architecture

```
Logs → Préprocessing → DistilBERT → Classification → Alertes
```

Le système est composé de trois modules principaux :

1. **Pipeline de données** : collecte et preprocessing des logs
2. **Modèle IA** : DistilBERT fine-tuné pour la classification de logs
3. **Système de production** : API (FastAPI) + Dashboard (Streamlit) + Alertes

---

## 📊 Données

Dataset synthétique de **1000 logs**, équilibré sur **5 classes** (200 logs/classe) :

| Classe | Description |
|---|---|
| `disk_full` | Saturation disque |
| `memory_leak` | Fuite mémoire |
| `network_failure` | Panne réseau |
| `authentication_error` | Erreur d'authentification |
| `resource_exhaustion` | Épuisement des ressources |

---

## 🤖 Modèle IA

Le modèle utilise **DistilBERT** pour :

- Compréhension sémantique des logs
- Inférence rapide
- Adaptation via fine-tuning

### Résultats obtenus

| Métrique | Valeur |
|---|---|
| Accuracy | 94% |
| F1-score | 94% |
| Latence | 150 ms / log |

---

## 🚨 Système de Monitoring

- **API FastAPI** : prédiction en temps réel
- **Dashboard Streamlit** : visualisation des résultats
- **Système d'alertes** : déclenchement automatique si confiance > 70%

---

## 🎯 Capacités de Détection

### Erreurs système critiques
- Disk Full
- Memory Leak
- Network Failure
- Authentication Errors
- Resource Exhaustion

### Anomalies applicatives
- Crash d'applications
- Timeout des services backend
- Erreurs HTTP 5xx
- Dégradation des performances

### Infrastructure
- Panne serveur
- Saturation CPU / RAM
- Problèmes disque / I/O
- Latence réseau élevée

### Root Cause Analysis (RCA)
- Identification automatique de la cause racine
- Classification des incidents
- Priorisation des alertes critiques

---

## 📁 Structure du Projet

```
log_monitoring_project/
├── config/             # Fichiers de configuration
├── data/               # Datasets
├── logs/               # Logs générés/collectés
├── models/             # Modèles entraînés
├── notebooks/          # Notebooks d'expérimentation
├── results/            # Résultats et métriques
├── src/                # Code source
├── config.yaml         # Configuration principale
├── main.py             # Point d'entrée principal
├── requirements.txt    # Dépendances Python
├── run_all.sh          # Script d'exécution complète
├── start_monitoring.sh # Script de lancement du monitoring
└── test_monitoring.py  # Tests du système
```

---


---

## 🚀 Utilisation

```bash
# Lancer le système complet
bash run_all.sh

# Ou lancer uniquement le monitoring
bash start_monitoring.sh

# Lancer les tests
python test_monitoring.py
```

---

## 🔮 Perspectives

- Détection de data drift
- Détection d'anomalies inconnues (Autoencoder / Isolation Forest)
- Déploiement cloud (Docker + Kubernetes)
- Monitoring continu du modèle IA (MLOps)

---

## 📈 État du Projet

| Composant | Avancement |
|---|---|
| Données | 100% |
| Modèle IA | 95% |
| Infrastructure | 70% |

Le système est fonctionnel avec un modèle performant (**94% F1-score**) et prêt pour la phase de production.

---


# logparsing-
# logparsing-
# logparsing-
# logparsing-
# logparsing-
# logparsing-
