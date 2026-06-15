"""Point d'entrée de Mangascan : fenêtre de contrôle + orchestration.

Lance l'interface PyQt5 principale (Start/Stop, statut temps réel, sélection de
la langue cible, logs), gère l'overlay de traduction, le sélecteur de zone, et
la hotkey globale d'activation/désactivation de l'overlay.
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional

from PyQt5 import QtCore, QtWidgets

from config import Config, load_config, save_config
from overlay import OverlayItem, RegionSelector, TranslationOverlay
from worker import CaptureWorker

logger = logging.getLogger("mangascan")

# Langues cibles proposées dans l'UI (code ISO -> libellé).
TARGET_LANGUAGES = {
    "fr": "Français",
    "en": "Anglais",
    "es": "Espagnol",
    "de": "Allemand",
    "it": "Italien",
    "pt": "Portugais",
    "ja": "Japonais",
}

# Bibliothèque optionnelle pour la hotkey globale.
try:
    import keyboard as _keyboard
except ImportError:  # pragma: no cover
    _keyboard = None


class ControlWindow(QtWidgets.QMainWindow):
    """Fenêtre principale de contrôle de l'application."""

    # Signal interne pour basculer l'overlay depuis le thread de la hotkey.
    _toggle_requested = QtCore.pyqtSignal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.worker: Optional[CaptureWorker] = None

        self.setWindowTitle("Mangascan — Traduction manga en temps réel")
        self.resize(460, 520)

        # Overlay de traduction (créé une fois, réutilisé).
        self.overlay = TranslationOverlay()

        self._build_ui()
        self._register_hotkey()

        self._toggle_requested.connect(self._toggle_overlay)

    # ------------------------------------------------------------------
    # Construction de l'UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        # --- Boutons Start / Stop ---
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("▶ Démarrer")
        self.stop_btn = QtWidgets.QPushButton("■ Arrêter")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_capture)
        self.stop_btn.clicked.connect(self.stop_capture)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        # --- Sélection de la zone / monitor ---
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

        # --- Sélecteur de langue cible ---
        lang_row = QtWidgets.QHBoxLayout()
        lang_row.addWidget(QtWidgets.QLabel("Langue cible :"))
        self.lang_combo = QtWidgets.QComboBox()
        for code, label in TARGET_LANGUAGES.items():
            self.lang_combo.addItem(f"{label} ({code})", code)
        # Présélectionne la langue de la config.
        idx = self.lang_combo.findData(self.config.target_language)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(self.lang_combo)
        layout.addLayout(lang_row)

        # --- Sélecteur de moteur de traduction ---
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

        # --- Statut temps réel ---
        self.status_label = QtWidgets.QLabel("Statut : à l'arrêt")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)

        self.stats_label = QtWidgets.QLabel("FPS : 0.0 | Bulles : 0 | Erreurs : —")
        layout.addWidget(self.stats_label)

        hotkey_hint = QtWidgets.QLabel(
            f"Hotkey overlay : {self.config.hotkey_toggle}"
        )
        hotkey_hint.setStyleSheet("color: gray;")
        layout.addWidget(hotkey_hint)

        # --- Bouton de bascule overlay ---
        self.toggle_overlay_btn = QtWidgets.QPushButton("👁 Afficher/Masquer overlay")
        self.toggle_overlay_btn.clicked.connect(self._toggle_overlay)
        layout.addWidget(self.toggle_overlay_btn)

        # --- Logs de debug ---
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
            return (f"Zone : {r['width']}×{r['height']} "
                    f"@ ({r['left']}, {r['top']})")
        return f"Zone : monitor entier (#{self.config.monitor_index})"

    # ------------------------------------------------------------------
    # Hotkey globale
    # ------------------------------------------------------------------
    def _register_hotkey(self) -> None:
        if _keyboard is None:
            self._log("Module 'keyboard' absent : hotkey globale désactivée.")
            return
        try:
            _keyboard.add_hotkey(
                self.config.hotkey_toggle,
                lambda: self._toggle_requested.emit(),
            )
            self._log(f"Hotkey '{self.config.hotkey_toggle}' enregistrée.")
        except Exception as exc:
            # Sous Linux, 'keyboard' requiert souvent les droits root.
            self._log(f"Impossible d'enregistrer la hotkey : {exc}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
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
        """Ouvre le sélecteur plein écran pour dessiner la zone de capture."""
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self._on_region_selected)
        self.selector.show()

    def use_full_monitor(self) -> None:
        """Réinitialise la zone pour capturer le monitor entier."""
        self.config.capture_region = None
        save_config(self.config)
        self.region_label.setText(self._region_text())
        if self.worker is not None:
            self.worker.update_config(self.config)
        self._log("Capture configurée sur le monitor entier.")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
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
        err_text = error if error else "—"
        self.stats_label.setText(
            f"FPS : {fps:.1f} | Bulles : {n_bubbles} | Erreurs : {err_text}"
        )

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
            self._log("⚠ DeepL sélectionné mais aucune clé API trouvée "
                      "(config ou DEEPL_API_KEY).")
        self._log(f"Moteur de traduction : {engine}")

    def _on_debug_toggled(self, state: int) -> None:
        self.config.debug_logs = bool(state)
        save_config(self.config)
        if self.worker is not None:
            self.worker.config.debug_logs = self.config.debug_logs

    def _toggle_overlay(self) -> None:
        enabled = self.overlay.toggle()
        self._log(f"Overlay {'affiché' if enabled else 'masqué'}.")

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self.config.debug_logs:
            self.log_view.appendPlainText(message)
        logger.info(message)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Arrêt propre du thread et de l'overlay à la fermeture."""
        self.stop_capture()
        self.overlay.close()
        if _keyboard is not None:
            try:
                _keyboard.clear_all_hotkeys()
            except Exception:
                pass
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = load_config()

    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow(config)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
