"""One widget per primary workspace, each with an explicit ``render`` step.

Pages never talk to the network, systemd, or a helper process directly.  They
receive a fully normalized :class:`~dlc_manager.viewmodel.ManagerView` and emit
Qt signals for the few actions a user can take.
"""

from __future__ import annotations

__all__ = ["analysis", "input", "live", "output", "processing", "routing", "system"]
