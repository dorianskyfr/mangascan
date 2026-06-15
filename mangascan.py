#!/usr/bin/env python3
"""Mangascan v0.6 — application tout-en-un (un seul fichier à exécuter).

Capture l'écran en temps réel, détecte le texte des planches de manga
(japonais en priorité, coréen/chinois en repli), le traduit dans la langue
choisie et l'affiche via une overlay transparente posée par-dessus l'écran.

Lancement :

    python mangascan.py

Ce fichier regroupe l'intégralité de l'application (config, capture, détection
des bulles, OCR, traduction, overlay et interface) pour n'avoir qu'un unique
fichier à exécuter. Les dépendances restent celles du requirements.txt.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "0.6"

logger = logging.getLogger("mangascan")

# ---------------------------------------------------------------------------
# Imports « lourds » : on les importe ici mais on tolère leur absence partielle.
# PyQt5, mss, opencv et numpy sont indispensables ; manga-ocr/easyocr/keyboard
# sont optionnels et chargés paresseusement plus bas.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import mss  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    from deep_translator import GoogleTranslator, DeeplTranslator
except ImportError:  # pragma: no cover
    GoogleTranslator = None  # type: ignore
    DeeplTranslator = None  # type: ignore

try:
    from langdetect import detect as _lang_detect
    from langdetect import DetectorFactory

    DetectorFactory.seed = 0  # détection déterministe
except ImportError:  # pragma: no cover
    _lang_detect = None  # type: ignore

try:
    import keyboard as _keyboard
except ImportError:  # pragma: no cover
    _keyboard = None


# ===========================================================================
# 1. Configuration (config.json)
# ===========================================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config.json")


@dataclass
class Config:
    """Paramètres de l'application (persistés dans config.json)."""

    target_language: str = "fr"        # langue cible ("fr", "en", "es", ...)
    capture_fps: int = 2               # captures analysées par seconde
    translator: str = "google"         # "google" (gratuit) ou "deepl"
    deepl_api_key: str = ""            # clé DeepL (sinon variable d'env)
    hotkey_toggle: str = "ctrl+shift+t"  # raccourci global overlay
    monitor_index: int = 1             # monitor capturé si pas de zone
    capture_region: Optional[Dict[str, int]] = field(default=None)
    debug_logs: bool = True            # logs de debug dans l'UI

    def resolve_deepl_key(self) -> str:
        """Clé DeepL : config en priorité, sinon variable d'environnement."""
        return self.deepl_api_key or os.environ.get("DEEPL_API_KEY", "")


def load_config(path: str = CONFIG_PATH) -> Config:
    """Charge la configuration, en créant un fichier par défaut si besoin."""
    if not os.path.exists(path):
        cfg = Config()
        save_config(cfg, path)
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        cfg = Config()
        save_config(cfg, path)
        return cfg
    valid = {f for f in Config.__dataclass_fields__}
    return Config(**{k: v for k, v in data.items() if k in valid})


def save_config(cfg: Config, path: str = CONFIG_PATH) -> None:
    """Sauvegarde la configuration au format JSON lisible."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, ensure_ascii=False, indent=2)


# ===========================================================================
# 2. Capture d'écran (mss)
# ===========================================================================
class ScreenCapturer:
    """Capture une zone de l'écran sous forme d'image NumPy (BGR).

    ``mss`` n'étant pas thread-safe, l'objet est créé paresseusement afin
    d'être instancié dans le thread de capture.
    """

    def __init__(self, monitor_index: int = 1,
                 region: Optional[Dict[str, int]] = None) -> None:
        self.monitor_index = monitor_index
        self.region = region
        self._sct = None

    def _ensure_sct(self):
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def _current_bbox(self, sct) -> Dict[str, int]:
        if self.region is not None:
            return self.region
        monitors = sct.monitors
        idx = self.monitor_index if 0 <= self.monitor_index < len(monitors) else 1
        return monitors[idx]

    def grab(self) -> np.ndarray:
        """Capture la zone courante (renvoie une image BGR contiguë)."""
        sct = self._ensure_sct()
        shot = sct.grab(self._current_bbox(sct))
        frame = np.asarray(shot)[:, :, :3]  # BGRA -> BGR
        return np.ascontiguousarray(frame)

    def set_region(self, region: Optional[Dict[str, int]]) -> None:
        self.region = region

    def set_monitor(self, monitor_index: int) -> None:
        self.monitor_index = monitor_index

    def list_monitors(self) -> List[Dict[str, int]]:
        return list(self._ensure_sct().monitors)

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


# ===========================================================================
# 3. Détection des bulles (OpenCV)
# ===========================================================================
@dataclass
class TextRegion:
    """Boîte englobante d'une bulle candidate (coordonnées locales)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y:self.y + self.h, self.x:self.x + self.w]


class BubbleDetector:
    """Détecte les bulles via un seuillage sur les zones blanches."""

    def __init__(self, min_area_ratio: float = 0.002,
                 max_area_ratio: float = 0.5,
                 white_threshold: int = 200) -> None:
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.white_threshold = white_threshold

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        if image is None or image.size == 0:
            return []
        h_img, w_img = image.shape[:2]
        total = float(h_img * w_img)
        min_area = self.min_area_ratio * total
        max_area = self.max_area_ratio * total

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.white_threshold, 255,
                                cv2.THRESH_BINARY)
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
            aspect = w / float(h) if h else 0
            if aspect < 0.1 or aspect > 10:
                continue
            regions.append(TextRegion(x, y, w, h))
        return regions

    def detect_or_full(self, image: np.ndarray) -> List[TextRegion]:
        """Détecte les bulles ou renvoie l'image entière en dernier recours."""
        regions = self.detect(image)
        if regions:
            return regions
        h_img, w_img = image.shape[:2]
        return [TextRegion(0, 0, w_img, h_img)]


# ===========================================================================
# 4. OCR (manga-ocr + EasyOCR)
# ===========================================================================
class OCREngine:
    """OCR combinant manga-ocr (JP prioritaire) et EasyOCR (KO/ZH fallback)."""

    def __init__(self, easyocr_langs: Optional[List[str]] = None,
                 use_gpu: bool = False) -> None:
        self.easyocr_langs = easyocr_langs or ["ko", "ch_sim", "en"]
        self.use_gpu = use_gpu
        self._manga_ocr = None
        self._easyocr_reader = None
        self._manga_ocr_available: Optional[bool] = None
        self._easyocr_available: Optional[bool] = None

    def _ensure_manga_ocr(self) -> bool:
        if self._manga_ocr_available is not None:
            return self._manga_ocr_available
        try:
            from manga_ocr import MangaOcr

            logger.info("Chargement du modèle manga-ocr…")
            self._manga_ocr = MangaOcr()
            self._manga_ocr_available = True
        except Exception as exc:
            logger.warning("manga-ocr indisponible : %s", exc)
            self._manga_ocr_available = False
        return self._manga_ocr_available

    def _ensure_easyocr(self) -> bool:
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

    def recognize(self, image_bgr: np.ndarray,
                  prefer_japanese: bool = True) -> str:
        if image_bgr is None or image_bgr.size == 0:
            return ""
        text = ""
        if prefer_japanese and self._ensure_manga_ocr():
            text = self._recognize_manga_ocr(image_bgr)
        if not text.strip() and self._ensure_easyocr():
            text = self._recognize_easyocr(image_bgr)
        return text.strip()

    def _recognize_manga_ocr(self, image_bgr: np.ndarray) -> str:
        if Image is None or self._manga_ocr is None:
            return ""
        try:
            pil = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
            return self._manga_ocr(pil) or ""
        except Exception as exc:
            logger.error("Erreur manga-ocr : %s", exc)
            return ""

    def _recognize_easyocr(self, image_bgr: np.ndarray) -> str:
        if self._easyocr_reader is None:
            return ""
        try:
            frags = self._easyocr_reader.readtext(image_bgr, detail=0,
                                                  paragraph=True)
            return " ".join(frags) if frags else ""
        except Exception as exc:
            logger.error("Erreur EasyOCR : %s", exc)
            return ""

    @property
    def manga_ocr_ready(self) -> bool:
        return bool(self._manga_ocr_available)


# ===========================================================================
# 5. Traduction (deep-translator + cache + langdetect)
# ===========================================================================
class TranslationResult:
    """Résultat d'une traduction (texte + provenance Google/DeepL)."""

    def __init__(self, text: str, source_lang: str, provider: str,
                 from_cache: bool = False) -> None:
        self.text = text
        self.source_lang = source_lang
        self.provider = provider  # "google", "deepl", "none"
        self.from_cache = from_cache


class Translator:
    """Traducteur avec cache mémoire, détection de langue et choix du moteur."""

    def __init__(self, target_language: str = "fr", engine: str = "google",
                 deepl_api_key: str = "") -> None:
        self.target_language = target_language
        self.engine = engine
        self.deepl_api_key = deepl_api_key
        self._cache: Dict[Tuple[str, str, str], str] = {}

    def set_target_language(self, lang: str) -> None:
        self.target_language = lang

    def set_engine(self, engine: str, deepl_api_key: str = "") -> None:
        self.engine = engine
        if deepl_api_key:
            self.deepl_api_key = deepl_api_key

    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        if not text.strip() or _lang_detect is None:
            return None
        try:
            return _lang_detect(text)
        except Exception:
            return None

    def translate(self, text: str) -> TranslationResult:
        text = (text or "").strip()
        if not text:
            return TranslationResult("", "unknown", "none")

        source_lang = self.detect_language(text) or "unknown"
        # Source == cible : on n'affiche rien.
        if source_lang != "unknown" and source_lang.startswith(self.target_language):
            return TranslationResult("", source_lang, "none")

        key = (text, self.target_language, self.engine)
        if key in self._cache:
            return TranslationResult(self._cache[key], source_lang,
                                     self.engine, from_cache=True)

        translated = self._call_engine(text)
        if translated:
            self._cache[key] = translated
            return TranslationResult(translated, source_lang, self.engine)
        return TranslationResult("", source_lang, "none")

    def _call_engine(self, text: str) -> str:
        if GoogleTranslator is None:
            logger.error("deep-translator n'est pas installé.")
            return ""
        try:
            if self.engine == "deepl":
                if not self.deepl_api_key:
                    logger.error("Clé DeepL manquante ; traduction impossible.")
                    return ""
                tr = DeeplTranslator(api_key=self.deepl_api_key,
                                     source="auto", target=self.target_language)
            else:
                tr = GoogleTranslator(source="auto", target=self.target_language)
            return tr.translate(text) or ""
        except Exception as exc:
            logger.error("Erreur de traduction (%s) : %s", self.engine, exc)
            return ""

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()


# ===========================================================================
# 6. Overlay PyQt5 + sélecteur de zone
# ===========================================================================
@dataclass
class OverlayItem:
    """Texte traduit à afficher, avec sa position/taille écran (px)."""

    x: int
    y: int
    w: int
    h: int
    text: str
    provider: str


class RegionSelector(QtWidgets.QWidget):
    """Superposition plein écran pour dessiner la zone à surveiller."""

    region_selected = QtCore.pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        geo = QtWidgets.QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint
                            | QtCore.Qt.WindowStaysOnTopHint
                            | QtCore.Qt.Tool)
        self.setWindowState(QtCore.Qt.WindowFullScreen)
        self.setCursor(QtCore.Qt.CrossCursor)
        self.setWindowOpacity(0.3)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._origin: Optional[QtCore.QPoint] = None
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

    def paintEvent(self, event):  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 120))
        p.setPen(QtGui.QColor(255, 255, 255))
        p.setFont(QtGui.QFont("Arial", 16))
        p.drawText(self.rect(), QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter,
                   "\nDessinez la zone à surveiller — Échap pour annuler")

    def mousePressEvent(self, event):  # noqa: N802
        self._origin = event.pos()
        self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._rubber.show()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._origin is not None:
            self._rubber.setGeometry(
                QtCore.QRect(self._origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._origin is None:
            return
        rect = QtCore.QRect(self._origin, event.pos()).normalized()
        tl = self.mapToGlobal(rect.topLeft())
        region = {"left": tl.x(), "top": tl.y(),
                  "width": rect.width(), "height": rect.height()}
        if region["width"] < 10 or region["height"] < 10:
            self.region_selected.emit(None)
        else:
            self.region_selected.emit(region)
        self.close()

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == QtCore.Qt.Key_Escape:
            self.region_selected.emit(None)
            self.close()


class TranslationOverlay(QtWidgets.QWidget):
    """Overlay transparent affichant les traductions par-dessus les bulles."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint
                            | QtCore.Qt.WindowStaysOnTopHint
                            | QtCore.Qt.Tool
                            | QtCore.Qt.WindowTransparentForInput)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._items: List[OverlayItem] = []
        self._enabled = True
        geo = QtWidgets.QApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)

    def set_items(self, items: List[OverlayItem]) -> None:
        self._items = items
        self.update()

    def clear(self) -> None:
        self._items = []
        self.update()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.show() if enabled else self.hide()

    def toggle(self) -> bool:
        self.set_enabled(not self._enabled)
        return self._enabled

    def paintEvent(self, event):  # noqa: N802
        if not self._enabled or not self._items:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        for item in self._items:
            self._draw_item(p, item)

    def _draw_item(self, p: QtGui.QPainter, item: OverlayItem) -> None:
        rect = QtCore.QRect(item.x, item.y, item.w, item.h)
        p.fillRect(rect, QtGui.QColor(0, 0, 0, 200))  # fond noir semi-transparent
        font = self._fit_font(item)
        p.setFont(font)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.drawText(rect.adjusted(6, 6, -6, -6),
                   QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, item.text)
        if item.provider and item.provider != "none":
            p.setFont(QtGui.QFont("Arial", 7))
            p.setPen(QtGui.QColor(180, 220, 255))
            p.drawText(rect.adjusted(0, 0, -3, -1),
                       QtCore.Qt.AlignBottom | QtCore.Qt.AlignRight,
                       item.provider.upper())

    @staticmethod
    def _fit_font(item: OverlayItem) -> QtGui.QFont:
        """Police adaptative : réduit la taille jusqu'à tenir dans la bulle."""
        base = max(8, min(28, item.h // 4))
        font = QtGui.QFont("Arial", base)
        font.setBold(True)
        text_rect = QtCore.QRect(item.x, item.y, item.w,
                                 item.h).adjusted(6, 6, -6, -6)
        for size in range(base, 6, -1):
            font.setPointSize(size)
            metrics = QtGui.QFontMetrics(font)
            b = metrics.boundingRect(
                text_rect, QtCore.Qt.TextWordWrap | QtCore.Qt.AlignCenter,
                item.text)
            if b.height() <= text_rect.height() and b.width() <= text_rect.width():
                break
        return font


# ===========================================================================
# 7. Thread de capture/OCR/traduction (QThread)
# ===========================================================================
class CaptureWorker(QtCore.QThread):
    """Boucle capture -> détection -> OCR -> traduction, hors thread UI."""

    items_ready = QtCore.pyqtSignal(list)
    stats_updated = QtCore.pyqtSignal(float, int, str)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._running = False
        self.capturer = ScreenCapturer(config.monitor_index, config.capture_region)
        self.detector = BubbleDetector()
        self.ocr = OCREngine()
        self.translator = Translator(config.target_language, config.translator,
                                     config.resolve_deepl_key())

    def stop(self) -> None:
        self._running = False

    def update_config(self, config: Config) -> None:
        self.config = config
        self.capturer.set_region(config.capture_region)
        self.capturer.set_monitor(config.monitor_index)
        self.translator.set_target_language(config.target_language)
        self.translator.set_engine(config.translator, config.resolve_deepl_key())

    def run(self) -> None:
        self._running = True
        fps = max(1, int(self.config.capture_fps))
        frame_interval = 1.0 / fps
        self._emit_log("Démarrage du thread de capture.")

        while self._running:
            cycle_start = time.time()
            error_msg = ""
            items: List[OverlayItem] = []
            n_bubbles = 0
            try:
                frame = self.capturer.grab()
                regions = self.detector.detect_or_full(frame)
                n_bubbles = len(regions)
                for region in regions:
                    crop = region.crop(frame)
                    text = self.ocr.recognize(crop, prefer_japanese=True)
                    if not text:
                        continue
                    result = self.translator.translate(text)
                    if not result.text:
                        continue
                    offset = self.capturer.region or {}
                    items.append(OverlayItem(
                        x=region.x + offset.get("left", 0),
                        y=region.y + offset.get("top", 0),
                        w=region.w, h=region.h,
                        text=result.text, provider=result.provider))
            except Exception as exc:
                error_msg = str(exc)
                logger.exception("Erreur dans la boucle de capture")
                self._emit_log(f"Erreur : {exc}")

            self.items_ready.emit(items)
            elapsed = time.time() - cycle_start
            self.stats_updated.emit(1.0 / elapsed if elapsed > 0 else 0.0,
                                    n_bubbles, error_msg)
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.capturer.close()
        self._emit_log("Thread de capture arrêté.")

    def _emit_log(self, message: str) -> None:
        if self.config.debug_logs:
            self.log_message.emit(message)


# ===========================================================================
# 8. Interface principale (fenêtre de contrôle)
# ===========================================================================
TARGET_LANGUAGES = {
    "fr": "Français", "en": "Anglais", "es": "Espagnol", "de": "Allemand",
    "it": "Italien", "pt": "Portugais", "ja": "Japonais",
}


class ControlWindow(QtWidgets.QMainWindow):
    """Fenêtre de contrôle (Start/Stop, statut, langue, moteur, logs)."""

    _toggle_requested = QtCore.pyqtSignal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.worker: Optional[CaptureWorker] = None
        self.setWindowTitle(f"Mangascan v{__version__} — Traduction manga")
        self.resize(460, 540)
        self.overlay = TranslationOverlay()
        self._build_ui()
        self._register_hotkey()
        self._toggle_requested.connect(self._toggle_overlay)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel(f"🈂️ Mangascan v{__version__}")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ Démarrer")
        self.stop_btn = QtWidgets.QPushButton("■ Arrêter")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        region_row = QtWidgets.QHBoxLayout()
        self.select_region_btn = QtWidgets.QPushButton("✎ Sélectionner une zone")
        self.select_region_btn.clicked.connect(self.select_region)
        self.full_monitor_btn = QtWidgets.QPushButton("🖵 Monitor entier")
        self.full_monitor_btn.clicked.connect(self.use_full_monitor)
        region_row.addWidget(self.select_region_btn)
        region_row.addWidget(self.full_monitor_btn)
        layout.addLayout(region_row)

        self.region_label = QtWidgets.QLabel(self._region_text())
        layout.addWidget(self.region_label)

        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Langue cible :"))
        self.lang_combo = QtWidgets.QComboBox()
        for code, label in TARGET_LANGUAGES.items():
            self.lang_combo.addItem(f"{label} ({code})", code)
        idx = self.lang_combo.findData(self.config.target_language)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self.lang_combo)
        layout.addLayout(lang_row)

        engine_row = QtWidgets.QHBoxLayout()
        engine_row.addWidget(QtWidgets.QLabel("Traducteur :"))
        self.engine_combo = QtWidgets.QComboBox()
        self.engine_combo.addItem("Google (gratuit)", "google")
        self.engine_combo.addItem("DeepL (clé requise)", "deepl")
        eidx = self.engine_combo.findData(self.config.translator)
        if eidx >= 0:
            self.engine_combo.setCurrentIndex(eidx)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        engine_row.addWidget(self.engine_combo)
        layout.addLayout(engine_row)

        self.status_label = QtWidgets.QLabel("Statut : à l'arrêt")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        self.stats_label = QtWidgets.QLabel("FPS : 0.0 | Bulles : 0 | Erreurs : —")
        layout.addWidget(self.stats_label)

        hint = QtWidgets.QLabel(f"Hotkey overlay : {self.config.hotkey_toggle}")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        self.toggle_overlay_btn = QtWidgets.QPushButton("👁 Afficher/Masquer overlay")
        self.toggle_overlay_btn.clicked.connect(self._toggle_overlay)
        layout.addWidget(self.toggle_overlay_btn)

        self.debug_checkbox = QtWidgets.QCheckBox("Afficher les logs de debug")
        self.debug_checkbox.setChecked(self.config.debug_logs)
        self.debug_checkbox.stateChanged.connect(self._on_debug_toggled)
        layout.addWidget(self.debug_checkbox)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        layout.addWidget(self.log_view, stretch=1)

        self.setCentralWidget(central)

    def _region_text(self) -> str:
        if self.config.capture_region:
            r = self.config.capture_region
            return f"Zone : {r['width']}×{r['height']} @ ({r['left']}, {r['top']})"
        return f"Zone : monitor entier (#{self.config.monitor_index})"

    def _register_hotkey(self) -> None:
        if _keyboard is None:
            self._log("Module 'keyboard' absent : hotkey globale désactivée.")
            return
        try:
            _keyboard.add_hotkey(self.config.hotkey_toggle,
                                 lambda: self._toggle_requested.emit())
            self._log(f"Hotkey '{self.config.hotkey_toggle}' enregistrée.")
        except Exception as exc:
            self._log(f"Impossible d'enregistrer la hotkey : {exc}")

    def start_capture(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.worker = CaptureWorker(self.config)
        self.worker.items_ready.connect(self._on_items_ready)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.log_message.connect(self._log)
        self.worker.start()
        self.overlay.set_enabled(True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Statut : en cours…")
        self._log("Capture démarrée.")

    def stop_capture(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.overlay.clear()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Statut : à l'arrêt")
        self._log("Capture arrêtée.")

    def select_region(self) -> None:
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self._on_region_selected)
        self.selector.show()

    def use_full_monitor(self) -> None:
        self.config.capture_region = None
        save_config(self.config)
        self.region_label.setText(self._region_text())
        if self.worker is not None:
            self.worker.update_config(self.config)
        self._log("Capture configurée sur le monitor entier.")

    def _on_region_selected(self, region: Optional[dict]) -> None:
        if region is None:
            self._log("Sélection de zone annulée.")
            return
        self.config.capture_region = region
        save_config(self.config)
        self.region_label.setText(self._region_text())
        if self.worker is not None:
            self.worker.update_config(self.config)
        self._log(f"Nouvelle zone : {region}")

    def _on_items_ready(self, items: List[OverlayItem]) -> None:
        self.overlay.set_items(items)

    def _on_stats_updated(self, fps: float, n_bubbles: int, error: str) -> None:
        self.stats_label.setText(
            f"FPS : {fps:.1f} | Bulles : {n_bubbles} | "
            f"Erreurs : {error if error else '—'}")

    def _on_language_changed(self) -> None:
        code = self.lang_combo.currentData()
        self.config.target_language = code
        save_config(self.config)
        if self.worker is not None:
            self.worker.update_config(self.config)
        self._log(f"Langue cible : {code}")

    def _on_engine_changed(self) -> None:
        engine = self.engine_combo.currentData()
        self.config.translator = engine
        save_config(self.config)
        if self.worker is not None:
            self.worker.update_config(self.config)
        if engine == "deepl" and not self.config.resolve_deepl_key():
            self._log("⚠ DeepL sélectionné mais aucune clé API trouvée.")
        self._log(f"Moteur de traduction : {engine}")

    def _on_debug_toggled(self, state: int) -> None:
        self.config.debug_logs = bool(state)
        save_config(self.config)
        if self.worker is not None:
            self.worker.config.debug_logs = self.config.debug_logs

    def _toggle_overlay(self) -> None:
        enabled = self.overlay.toggle()
        self._log(f"Overlay {'affiché' if enabled else 'masqué'}.")

    def _log(self, message: str) -> None:
        if self.config.debug_logs:
            self.log_view.appendPlainText(message)
        logger.info(message)

    def closeEvent(self, event):  # noqa: N802
        self.stop_capture()
        self.overlay.close()
        if _keyboard is not None:
            try:
                _keyboard.clear_all_hotkeys()
            except Exception:
                pass
        super().closeEvent(event)


# ===========================================================================
# 9. Point d'entrée
# ===========================================================================
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    config = load_config()
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow(config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
