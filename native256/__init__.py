"""Training and export scaffold for the native 256px face-swap student.

The package is intentionally isolated from the application runtime. Importing
``native256`` does not import PyTorch; training tools import it explicitly.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
