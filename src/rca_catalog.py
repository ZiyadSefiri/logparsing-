"""
Catalogue metier des causes racines detectees.
"""
from __future__ import annotations


RCA_CATALOG = {
    "disk_full": {
        "label": "Disk full",
        "severity": "CRITICAL",
        "priority": "IMMEDIATE",
        "contact": "DevOps Team",
        "symptoms": ["Disk space low", "No space left", "filesystem full"],
        "actions": [
            "Verifier immediatement l'espace disque",
            "Nettoyer les fichiers temporaires et logs obsoletes",
            "Agrandir le volume ou ajouter de la capacite",
        ],
    },
    "memory_leak": {
        "label": "Memory leak",
        "severity": "HIGH",
        "priority": "HIGH",
        "contact": "SRE Team",
        "symptoms": ["Memory usage high", "Out of memory", "heap exceeded"],
        "actions": [
            "Verifier la consommation memoire du service",
            "Redemarrer le service si la pression memoire est critique",
            "Analyser les dumps ou traces applicatives",
        ],
    },
    "network_failure": {
        "label": "Network failure",
        "severity": "HIGH",
        "priority": "HIGH",
        "contact": "Network Team",
        "symptoms": ["Connection timeout", "No route to host", "interface down"],
        "actions": [
            "Verifier la connectivite reseau",
            "Tester DNS, ports et routes",
            "Controler les regles firewall et la configuration du service",
        ],
    },
    "authentication_error": {
        "label": "Authentication error",
        "severity": "MEDIUM",
        "priority": "NORMAL",
        "contact": "Security Team",
        "symptoms": ["Authentication failed", "Permission denied", "session refused"],
        "actions": [
            "Verifier les identifiants et permissions",
            "Controler les logs d'acces suspects",
            "Valider la configuration IAM ou comptes applicatifs",
        ],
    },
    "resource_exhaustion": {
        "label": "Resource exhaustion",
        "severity": "MEDIUM",
        "priority": "NORMAL",
        "contact": "Platform Team",
        "symptoms": ["Thread pool exhausted", "too many open files", "limits reached"],
        "actions": [
            "Verifier les limites systeme et quotas applicatifs",
            "Reduire la charge ou augmenter les workers",
            "Analyser la saturation CPU, files et threads",
        ],
    },
    "normal_operation": {
        "label": "Normal operation",
        "severity": "LOW",
        "priority": "LOW",
        "contact": "Operations Team",
        "symptoms": ["Service started", "operation completed", "health check ok"],
        "actions": [
            "Aucune action immediate",
            "Continuer la surveillance",
        ],
    },
}


def get_cause_details(cause: str) -> dict:
    default = {
        "label": cause.replace("_", " ").title(),
        "severity": "LOW",
        "priority": "LOW",
        "contact": "Operations Team",
        "symptoms": [],
        "actions": ["Analyser le log et confirmer la cause racine"],
    }
    return {**default, **RCA_CATALOG.get(cause, {})}


def format_actions(actions: list[str]) -> str:
    return " | ".join(actions)
