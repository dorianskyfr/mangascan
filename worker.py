"""Thread de capture/OCR/traduction découplé de l'UI (QThread).

Ce thread exécute en boucle : capture d'écran -> détection des bulles -> OCR ->
traduction. Les résultats sont émis vers le thread UI via des signaux Qt, ce qui
garantit que PyQt n'est jamais bloqué par les opérations lourdes.
"""

from __future__ import annotations

import logging
import time
from typing import List

from PyQt5 import QtCore

from capture import ScreenCapturer
from config import Config
from detector import BubbleDetector
from ocr import OCREngine
from overlay import OverlayItem
from translator import Translator

logger = logging.getLogger("mangascan.worker")


class CaptureWorker(QtCore.QThread):
    """Boucle de traitement exécutée dans un thread séparé."""

    # Émis à chaque cycle avec la liste d'items prêts pour l'overlay.
    items_ready = QtCore.pyqtSignal(list)
    # Statistiques temps réel : (fps_mesuré, nb_bulles, message_erreur).
    stats_updated = QtCore.pyqtSignal(float, int, str)
    # Message de log (debug).
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self._running = False

        # Composants instanciés ici, mais mss/manga-ocr s'initialisent dans run().
        self.capturer = ScreenCapturer(
            monitor_index=config.monitor_index,
            region=config.capture_region,
        )
        self.detector = BubbleDetector()
        self.ocr = OCREngine()
        self.translator = Translator(
            target_language=config.target_language,
            engine=config.translator,
            deepl_api_key=config.resolve_deepl_key(),
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Demande l'arrêt propre de la boucle."""
        self._running = False

    def update_config(self, config: Config) -> None:
        """Applique à chaud les changements de configuration."""
        self.config = config
        self.capturer.set_region(config.capture_region)
        self.capturer.set_monitor(config.monitor_index)
        self.translator.set_target_language(config.target_language)
        self.translator.set_engine(config.translator, config.resolve_deepl_key())

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401 - appelé par QThread.start()
        """Boucle principale de capture/traitement."""
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
                        continue  # langue == cible, ou échec de traduction

                    # Conversion des coordonnées locales -> globales écran.
                    offset = self.capturer.region or {}
                    gx = region.x + offset.get("left", 0)
                    gy = region.y + offset.get("top", 0)
                    items.append(
                        OverlayItem(
                            x=gx, y=gy, w=region.w, h=region.h,
                            text=result.text, provider=result.provider,
                        )
                    )
            except Exception as exc:  # robustesse : la boucle ne doit pas mourir
                error_msg = str(exc)
                logger.exception("Erreur dans la boucle de capture")
                self._emit_log(f"Erreur : {exc}")

            self.items_ready.emit(items)

            # Calcul du FPS réel (1 / durée du cycle).
            elapsed = time.time() - cycle_start
            measured_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.stats_updated.emit(measured_fps, n_bubbles, error_msg)

            # Respecte la cadence cible.
            sleep_for = frame_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.capturer.close()
        self._emit_log("Thread de capture arrêté.")

    def _emit_log(self, message: str) -> None:
        if self.config.debug_logs:
            self.log_message.emit(message)
