"""Capture d'écran en temps réel via ``mss``.

Le module fournit une classe :class:`ScreenCapturer` capable de capturer soit
un monitor entier, soit une zone rectangulaire arbitraire. La sortie est une
image NumPy au format BGR, directement exploitable par OpenCV.
"""

from __future__ import annotations

from typing import Dict, Optional

import mss
import numpy as np


class ScreenCapturer:
    """Capture une portion de l'écran sous forme de tableau NumPy (BGR).

    ``mss`` n'est pas thread-safe : une instance ``mss.mss()`` doit être créée
    et utilisée dans le même thread. On crée donc l'objet de façon paresseuse
    pour qu'il soit instancié dans le thread de capture.
    """

    def __init__(self, monitor_index: int = 1,
                 region: Optional[Dict[str, int]] = None) -> None:
        """:param monitor_index: index du monitor mss (1 = écran principal).
        :param region: zone {"top","left","width","height"} ou None pour le
            monitor entier.
        """
        self.monitor_index = monitor_index
        self.region = region
        self._sct: Optional[mss.base.MSSBase] = None

    def _ensure_sct(self) -> mss.base.MSSBase:
        """Instancie l'objet mss à la première utilisation (lazy init)."""
        if self._sct is None:
            self._sct = mss.mss()
        return self._sct

    def _current_bbox(self, sct: mss.base.MSSBase) -> Dict[str, int]:
        """Détermine la zone à capturer (région perso ou monitor complet)."""
        if self.region is not None:
            return self.region
        # mss expose la liste des monitors ; l'index 0 = tous les écrans.
        monitors = sct.monitors
        idx = self.monitor_index if 0 <= self.monitor_index < len(monitors) else 1
        return monitors[idx]

    def grab(self) -> np.ndarray:
        """Capture la zone courante et renvoie une image NumPy au format BGR."""
        sct = self._ensure_sct()
        bbox = self._current_bbox(sct)
        shot = sct.grab(bbox)
        # mss renvoie du BGRA ; on convertit en BGR pour OpenCV.
        frame = np.asarray(shot)[:, :, :3]
        return np.ascontiguousarray(frame)

    def set_region(self, region: Optional[Dict[str, int]]) -> None:
        """Définit (ou réinitialise) la zone de capture personnalisée."""
        self.region = region

    def set_monitor(self, monitor_index: int) -> None:
        """Change le monitor capturé (utilisé en l'absence de région)."""
        self.monitor_index = monitor_index

    def list_monitors(self) -> list[Dict[str, int]]:
        """Retourne la liste des monitors détectés par mss."""
        sct = self._ensure_sct()
        return list(sct.monitors)

    def close(self) -> None:
        """Libère les ressources mss."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None
