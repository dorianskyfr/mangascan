"""Wrapper OCR : ``manga-ocr`` en priorité, ``EasyOCR`` en repli.

``manga-ocr`` est spécialisé pour le japonais manuscrit/imprimé des mangas et
donne d'excellents résultats sur le japonais. Pour le coréen et le chinois, on
bascule sur ``EasyOCR``. Les deux moteurs sont importés de façon paresseuse :
l'application reste utilisable même si l'un des deux n'est pas installé.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

try:  # PIL est requis par manga-ocr (entrée attendue : image PIL).
    from PIL import Image
except ImportError:  # pragma: no cover - PIL fait partie des dépendances
    Image = None  # type: ignore

logger = logging.getLogger("mangascan.ocr")


class OCREngine:
    """Reconnaissance de texte combinant manga-ocr (JP) et EasyOCR (KO/ZH)."""

    def __init__(self, easyocr_langs: Optional[List[str]] = None,
                 use_gpu: bool = False) -> None:
        """:param easyocr_langs: langues à charger pour EasyOCR (fallback).
        :param use_gpu: active le GPU pour EasyOCR si disponible.
        """
        # Langues EasyOCR : coréen, chinois simplifié + anglais par défaut.
        self.easyocr_langs = easyocr_langs or ["ko", "ch_sim", "en"]
        self.use_gpu = use_gpu
        self._manga_ocr = None  # instance manga_ocr.MangaOcr (lazy)
        self._easyocr_reader = None  # instance easyocr.Reader (lazy)
        self._manga_ocr_available: Optional[bool] = None
        self._easyocr_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Initialisation paresseuse des moteurs
    # ------------------------------------------------------------------
    def _ensure_manga_ocr(self) -> bool:
        """Charge manga-ocr à la demande. Renvoie True si disponible."""
        if self._manga_ocr_available is not None:
            return self._manga_ocr_available
        try:
            from manga_ocr import MangaOcr

            logger.info("Chargement du modèle manga-ocr…")
            self._manga_ocr = MangaOcr()
            self._manga_ocr_available = True
        except Exception as exc:  # ImportError ou échec de téléchargement modèle
            logger.warning("manga-ocr indisponible : %s", exc)
            self._manga_ocr_available = False
        return self._manga_ocr_available

    def _ensure_easyocr(self) -> bool:
        """Charge EasyOCR à la demande. Renvoie True si disponible."""
        if self._easyocr_available is not None:
            return self._easyocr_available
        try:
            import easyocr

            logger.info("Chargement du modèle EasyOCR (%s)…", self.easyocr_langs)
            self._easyocr_reader = easyocr.Reader(self.easyocr_langs,
                                                  gpu=self.use_gpu)
            self._easyocr_available = True
        except Exception as exc:
            logger.warning("EasyOCR indisponible : %s", exc)
            self._easyocr_available = False
        return self._easyocr_available

    # ------------------------------------------------------------------
    # Reconnaissance
    # ------------------------------------------------------------------
    def recognize(self, image_bgr: np.ndarray,
                  prefer_japanese: bool = True) -> str:
        """Reconnaît le texte d'une image (BGR NumPy).

        :param prefer_japanese: si True, tente manga-ocr en premier.
        :returns: le texte reconnu (chaîne vide si rien / aucun moteur).
        """
        if image_bgr is None or image_bgr.size == 0:
            return ""

        text = ""
        if prefer_japanese and self._ensure_manga_ocr():
            text = self._recognize_manga_ocr(image_bgr)

        # Repli EasyOCR si manga-ocr a échoué/est vide ou non préféré.
        if not text.strip() and self._ensure_easyocr():
            text = self._recognize_easyocr(image_bgr)

        return text.strip()

    def _recognize_manga_ocr(self, image_bgr: np.ndarray) -> str:
        """Exécute manga-ocr (attend une image PIL en RVB)."""
        if Image is None or self._manga_ocr is None:
            return ""
        try:
            rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
            pil_image = Image.fromarray(rgb)
            return self._manga_ocr(pil_image) or ""
        except Exception as exc:
            logger.error("Erreur manga-ocr : %s", exc)
            return ""

    def _recognize_easyocr(self, image_bgr: np.ndarray) -> str:
        """Exécute EasyOCR et concatène les fragments détectés."""
        if self._easyocr_reader is None:
            return ""
        try:
            # detail=0 -> renvoie directement la liste des chaînes.
            fragments = self._easyocr_reader.readtext(image_bgr, detail=0,
                                                      paragraph=True)
            return " ".join(fragments) if fragments else ""
        except Exception as exc:
            logger.error("Erreur EasyOCR : %s", exc)
            return ""

    @property
    def manga_ocr_ready(self) -> bool:
        """Indique si manga-ocr a pu être chargé (None tant que non testé)."""
        return bool(self._manga_ocr_available)
