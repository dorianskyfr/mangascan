"""Détection des bulles de dialogue dans une planche de manga (OpenCV).

L'approche est volontairement simple et robuste : les bulles de manga sont
majoritairement des zones blanches entourées d'un contour noir. On isole donc
les régions claires, on filtre par taille/forme, puis on renvoie les boîtes
englobantes candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class TextRegion:
    """Boîte englobante d'une bulle candidate (coordonnées locales à l'image)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Découpe la région correspondante dans l'image source."""
        return image[self.y:self.y + self.h, self.x:self.x + self.w]


class BubbleDetector:
    """Détecte les bulles de dialogue via un seuillage sur les zones blanches."""

    def __init__(self, min_area_ratio: float = 0.002,
                 max_area_ratio: float = 0.5,
                 white_threshold: int = 200) -> None:
        """:param min_area_ratio: aire minimale d'une bulle (fraction de l'image).
        :param max_area_ratio: aire maximale d'une bulle (fraction de l'image).
        :param white_threshold: seuil de luminosité pour considérer un pixel
            comme « blanc » (fond de bulle).
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.white_threshold = white_threshold

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Renvoie la liste des bulles candidates trouvées dans ``image`` (BGR)."""
        if image is None or image.size == 0:
            return []

        h_img, w_img = image.shape[:2]
        total_area = float(h_img * w_img)
        min_area = self.min_area_ratio * total_area
        max_area = self.max_area_ratio * total_area

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # On isole les zones claires (fond des bulles).
        _, mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        # Fermeture morphologique pour relier les caractères à l'intérieur.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        regions: List[TextRegion] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Filtre sur le ratio d'aspect pour éliminer les bandes très fines.
            aspect = w / float(h) if h else 0
            if aspect < 0.1 or aspect > 10:
                continue
            regions.append(TextRegion(x, y, w, h))

        return regions

    def detect_or_full(self, image: np.ndarray) -> List[TextRegion]:
        """Comme :meth:`detect`, mais renvoie l'image entière en dernier recours.

        Utile quand aucune bulle n'est détectée : on laisse alors l'OCR
        s'exécuter sur toute la zone capturée.
        """
        regions = self.detect(image)
        if regions:
            return regions
        h_img, w_img = image.shape[:2]
        return [TextRegion(0, 0, w_img, h_img)]
