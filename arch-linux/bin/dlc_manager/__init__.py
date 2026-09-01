"""Native Deep-Live-Cam manager: shell, view model, pages, and adapters.

The package sits beside ``tester.py`` in both the development checkout
(``arch-linux/bin``) and the installed layout
(``/usr/local/lib/deep-live-cam-arch``).  The existing helper modules
(``common``, ``camera_adapters``, ``camera_profiles``, ``pipeline_topology``,
``source_history``) are flat modules in that same directory and keep their
current import names, so the directory is put on ``sys.path`` here rather than
rewriting those contracts.
"""

from __future__ import annotations

from pathlib import Path
import sys

_HELPER_DIRECTORY = str(Path(__file__).resolve().parent.parent)
if _HELPER_DIRECTORY not in sys.path:
    sys.path.append(_HELPER_DIRECTORY)

__all__ = [
    "contracts",
    "decoders",
    "health",
    "pages",
    "quality",
    "services",
    "shell",
    "theme",
    "viewmodel",
    "wheel",
    "widgets",
]
