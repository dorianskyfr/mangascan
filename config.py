"""Chargement et sauvegarde de la configuration utilisateur (config.json).

Ce module centralise la gestion des paramètres de l'application. Si le fichier
``config.json`` n'existe pas, il est créé automatiquement avec des valeurs par
défaut. Les valeurs manquantes dans un fichier existant sont complétées.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

# Chemin du fichier de configuration, situé à côté de ce module.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


@dataclass
class Config:
    """Représente les paramètres de l'application.

    Les valeurs par défaut correspondent à celles attendues dans le cahier
    des charges (voir ``config.json``).
    """

    # Langue cible de la traduction (code ISO : "fr", "en", "es", ...).
    target_language: str = "fr"
    # Nombre de captures d'écran analysées par seconde.
    capture_fps: int = 2
    # Moteur de traduction : "google" (gratuit) ou "deepl" (clé requise).
    translator: str = "google"
    # Clé API DeepL. Si vide, on tente de lire la variable d'environnement.
    deepl_api_key: str = ""
    # Raccourci global pour activer/désactiver l'overlay.
    hotkey_toggle: str = "ctrl+shift+t"
    # Monitor à capturer si aucune zone n'est sélectionnée (1 = écran principal).
    monitor_index: int = 1
    # Zone de capture personnalisée {"top","left","width","height"} ou None.
    capture_region: Dict[str, int] | None = field(default=None)
    # Affichage des logs de debug dans la fenêtre de contrôle.
    debug_logs: bool = True

    def resolve_deepl_key(self) -> str:
        """Retourne la clé DeepL : config en priorité, sinon variable d'env."""
        return self.deepl_api_key or os.environ.get("DEEPL_API_KEY", "")


def load_config(path: str = CONFIG_PATH) -> Config:
    """Charge la configuration depuis le disque.

    Crée un fichier par défaut si nécessaire et complète les clés manquantes.
    """
    if not os.path.exists(path):
        cfg = Config()
        save_config(cfg, path)
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data: Dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError):
        # Fichier corrompu : on repart sur des valeurs par défaut.
        cfg = Config()
        save_config(cfg, path)
        return cfg

    # On ne garde que les clés connues du dataclass pour éviter les erreurs.
    valid_keys = {f for f in Config.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return Config(**filtered)


def save_config(cfg: Config, path: str = CONFIG_PATH) -> None:
    """Sauvegarde la configuration sur le disque au format JSON lisible."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)
