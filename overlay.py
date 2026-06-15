"""Fenêtres PyQt5 : overlay de traduction transparent + sélecteur de zone.

Deux composants :

- :class:`RegionSelector` : superposition plein écran permettant à l'utilisateur
  de dessiner un rectangle pour délimiter la zone à surveiller (façon ShareX).
- :class:`TranslationOverlay` : fenêtre sans décoration, toujours au premier
  plan, transparente aux clics, qui affiche les traductions par-dessus les
  bulles détectées.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass
class OverlayItem:
    """Un texte traduit à afficher, avec sa position/taille à l'écran (px)."""

    x: int
    y: int
    w: int
    h: int
    text: str
    provider: str  # "google", "deepl", ... (affiché en petit)


class RegionSelector(QtWidgets.QWidget):
    """Widget plein écran pour sélectionner une zone rectangulaire.

    Émet :pyattr:`region_selected` avec un dict
    {"top","left","width","height"} compatible mss, ou ``None`` si annulé.
    """

    region_selected = QtCore.pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        # Couvre tout le bureau virtuel (multi-écrans).
        geo = QtWidgets.QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setCursor(QtCore.Qt.CrossCursor)
        # Fond semi-transparent pour voir le bureau pendant la sélection.
        self.setWindowOpacity(0.3)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self._origin: Optional[QtCore.QPoint] = None
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 120))
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.setFont(QtGui.QFont("Arial", 16))
        painter.drawText(
            self.rect(),
            QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
            "\nDessinez la zone à surveiller — Échap pour annuler",
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._origin = event.pos()
        self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is not None:
            self._rubber.setGeometry(
                QtCore.QRect(self._origin, event.pos()).normalized()
            )

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        rect = QtCore.QRect(self._origin, event.pos()).normalized()
        # Coordonnées globales (gère le décalage multi-écrans).
        top_left = self.mapToGlobal(rect.topLeft())
        region = {
            "left": top_left.x(),
            "top": top_left.y(),
            "width": rect.width(),
            "height": rect.height(),
        }
        # On ignore les sélections trop petites (clic accidentel).
        if region["width"] < 10 or region["height"] < 10:
            self.region_selected.emit(None)
        else:
            self.region_selected.emit(region)
        self.close()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key_Escape:
            self.region_selected.emit(None)
            self.close()


class TranslationOverlay(QtWidgets.QWidget):
    """Overlay transparent affichant les traductions par-dessus les bulles."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )
        # Fond global transparent ; on dessine nous-mêmes les rectangles.
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # Ne bloque pas les interactions avec les fenêtres en dessous.
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        self._items: List[OverlayItem] = []
        self._enabled = True

        # Par défaut, l'overlay couvre tout le bureau virtuel.
        geo = QtWidgets.QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)

    # ------------------------------------------------------------------
    def set_items(self, items: List[OverlayItem]) -> None:
        """Met à jour les traductions à dessiner et redessine."""
        self._items = items
        self.update()

    def clear(self) -> None:
        """Efface l'overlay."""
        self._items = []
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        """Affiche ou masque l'overlay (toggle hotkey)."""
        self._enabled = enabled
        if enabled:
            self.show()
        else:
            self.hide()

    def toggle(self) -> bool:
        """Inverse l'état d'affichage. Renvoie le nouvel état."""
        self.set_enabled(not self._enabled)
        return self._enabled

    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if not self._enabled or not self._items:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for item in self._items:
            self._draw_item(painter, item)

    def _draw_item(self, painter: QtGui.QPainter, item: OverlayItem) -> None:
        """Dessine une bulle traduite : fond noir semi-transparent + texte."""
        rect = QtCore.QRect(item.x, item.y, item.w, item.h)

        # Fond semi-transparent noir.
        painter.setBrushOrigin(rect.topLeft())
        painter.fillRect(rect, QtGui.QColor(0, 0, 0, 200))

        # Taille de police adaptative selon la hauteur de la bulle.
        font = self._fit_font(painter, item)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255))

        # Marge interne pour ne pas coller le texte au bord.
        text_rect = rect.adjusted(6, 6, -6, -6)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
            item.text,
        )

        # Petit badge de provenance (Google / DeepL) en bas à droite.
        if item.provider and item.provider not in ("none",):
            painter.setFont(QtGui.QFont("Arial", 7))
            painter.setPen(QtGui.QColor(180, 220, 255))
            painter.drawText(
                rect.adjusted(0, 0, -3, -1),
                QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight,
                item.provider.upper(),
            )

    @staticmethod
    def _fit_font(painter: QtGui.QPainter, item: OverlayItem) -> QtGui.QFont:
        """Calcule une taille de police qui tient dans la bulle détectée."""
        # Heuristique : taille proportionnelle à la hauteur, bornée.
        base_size = max(8, min(28, item.h // 4))
        font = QtGui.QFont("Arial", base_size)
        font.setBold(True)

        # Réduit la police tant que le texte déborde de la bulle.
        text_rect = QtCore.QRect(item.x, item.y, item.w, item.h).adjusted(6, 6, -6, -6)
        for size in range(base_size, 6, -1):
            font.setPointSize(size)
            metrics = QtGui.QFontMetrics(font)
            bounding = metrics.boundingRect(
                text_rect, QtCore.Qt.TextWordWrap | QtCore.Qt.AlignCenter, item.text
            )
            if bounding.height() <= text_rect.height() and bounding.width() <= text_rect.width():
                break
        return font
