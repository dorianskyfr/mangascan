"""Wrapper de traduction basé sur ``deep-translator`` (Google / DeepL).

Gère un cache mémoire pour éviter les appels API répétés sur des bulles
identiques, la détection automatique de la langue source via ``langdetect``, et
le court-circuit lorsque la langue source == langue cible.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("mangascan.translator")

try:
    from deep_translator import GoogleTranslator, DeeplTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore
    DeeplTranslator = None  # type: ignore

try:
    from langdetect import detect as _lang_detect
    from langdetect import DetectorFactory

    # Rend la détection déterministe.
    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover
    _lang_detect = None  # type: ignore


class TranslationResult:
    """Résultat d'une traduction, incluant la provenance (Google/DeepL)."""

    def __init__(self, text: str, source_lang: str, provider: str,
                 from_cache: bool = False) -> None:
        self.text = text
        self.source_lang = source_lang
        self.provider = provider  # "google", "deepl", "none" ou "cache"
        self.from_cache = from_cache


class Translator:
    """Traducteur avec cache, détection de langue et choix du moteur."""

    def __init__(self, target_language: str = "fr", engine: str = "google",
                 deepl_api_key: str = "") -> None:
        self.target_language = target_language
        self.engine = engine  # "google" ou "deepl"
        self.deepl_api_key = deepl_api_key
        # Cache : clé = (texte_source, langue_cible, moteur) -> texte traduit.
        self._cache: Dict[Tuple[str, str, str], str] = {}

    # ------------------------------------------------------------------
    # Configuration dynamique
    # ------------------------------------------------------------------
    def set_target_language(self, lang: str) -> None:
        self.target_language = lang

    def set_engine(self, engine: str, deepl_api_key: str = "") -> None:
        self.engine = engine
        if deepl_api_key:
            self.deepl_api_key = deepl_api_key

    # ------------------------------------------------------------------
    # Détection de langue
    # ------------------------------------------------------------------
    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """Détecte la langue source ; renvoie None en cas d'échec."""
        if not text.strip() or _lang_detect is None:
            return None
        try:
            return _lang_detect(text)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Traduction
    # ------------------------------------------------------------------
    def translate(self, text: str) -> TranslationResult:
        """Traduit ``text`` vers la langue cible configurée.

        - Court-circuite si la langue source == langue cible.
        - Utilise le cache si la traduction a déjà été faite.
        - Gère proprement les erreurs (timeout, moteur indisponible).
        """
        text = (text or "").strip()
        if not text:
            return TranslationResult("", "unknown", "none")

        source_lang = self.detect_language(text) or "unknown"

        # Si la source correspond déjà à la cible, on n'affiche rien.
        if source_lang != "unknown" and source_lang.startswith(self.target_language):
            return TranslationResult("", source_lang, "none")

        cache_key = (text, self.target_language, self.engine)
        if cache_key in self._cache:
            return TranslationResult(self._cache[cache_key], source_lang,
                                     self.engine, from_cache=True)

        translated = self._call_engine(text)
        if translated:
            self._cache[cache_key] = translated
            return TranslationResult(translated, source_lang, self.engine)

        # Échec de traduction : on renvoie un résultat vide explicite.
        return TranslationResult("", source_lang, "none")

    def _call_engine(self, text: str) -> str:
        """Appelle le moteur de traduction choisi avec gestion d'erreurs."""
        if GoogleTranslator is None:
            logger.error("deep-translator n'est pas installé.")
            return ""

        try:
            if self.engine == "deepl":
                if not self.deepl_api_key:
                    logger.error("Clé DeepL manquante ; traduction impossible.")
                    return ""
                translator = DeeplTranslator(
                    api_key=self.deepl_api_key,
                    source="auto",
                    target=self.target_language,
                )
            else:  # google par défaut
                translator = GoogleTranslator(
                    source="auto",
                    target=self.target_language,
                )
            return translator.translate(text) or ""
        except Exception as exc:
            # Timeout, quota dépassé, langue non supportée, etc.
            logger.error("Erreur de traduction (%s) : %s", self.engine, exc)
            return ""

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
