"""Per-user ownership and activation for the native manager.

The manager is an active controller, not a passive status window: constructing
two copies starts two desired-state reconciliation loops.  An advisory lock is
therefore acquired before :class:`ManagerWindow` is constructed.  ``flock``
is intentionally used instead of a PID-file-only scheme because the kernel
releases it when a process crashes and it cannot leave a stale owner behind.

A private local socket gives subsequent launches a best-effort way to raise
the existing window.  Failure to deliver that request never permits a second
controller to start; the lock remains authoritative.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat
import tempfile

from PySide6.QtCore import QObject, QIODevice, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


LOCK_NAME = "manager.lock"
SOCKET_NAME = "manager.sock"


class InstanceGuardError(RuntimeError):
    """The manager cannot establish safe single-instance ownership."""


def default_instance_directory(
    *,
    environment: dict[str, str] | None = None,
    user_id: int | None = None,
) -> Path:
    """Return one private runtime directory shared by this user's launches."""
    values = os.environ if environment is None else environment
    uid = os.getuid() if user_id is None else user_id
    configured = values.get("XDG_RUNTIME_DIR", "").strip()
    if configured and Path(configured).is_absolute():
        return Path(configured) / "deep-live-cam-manager"

    runtime = Path("/run/user") / str(uid)
    if runtime.is_dir():
        return runtime / "deep-live-cam-manager"
    return Path(tempfile.gettempdir()) / f"deep-live-cam-manager-{uid}"


class ManagerInstanceGuard(QObject):
    """Own the per-user manager lock and its best-effort activation socket."""

    activationRequested = Signal()

    def __init__(self, directory: Path | None = None) -> None:
        super().__init__()
        self.directory = Path(directory or default_instance_directory())
        self.lock_path = self.directory / LOCK_NAME
        self.socket_path = self.directory / SOCKET_NAME
        self._lock_descriptor: int | None = None
        self._owner_pid: int | None = None
        self._server: QLocalServer | None = None
        self._listening = False

    @property
    def owner_pid(self) -> int | None:
        """Return the PID recorded by the current owner, when it is readable."""
        return self._owner_pid

    @property
    def owns_lock(self) -> bool:
        return self._lock_descriptor is not None

    def _prepare_directory(self) -> None:
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            status = self.directory.lstat()
        except OSError as exc:
            raise InstanceGuardError(
                f"cannot create manager runtime directory {self.directory}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(status.st_mode)
            or status.st_uid != os.getuid()
        ):
            raise InstanceGuardError(
                f"unsafe manager runtime directory ownership: {self.directory}"
            )
        try:
            self.directory.chmod(0o700)
        except OSError as exc:
            raise InstanceGuardError(
                f"cannot secure manager runtime directory {self.directory}: {exc}"
            ) from exc

    def try_acquire(self) -> bool:
        """Acquire controller ownership without waiting.

        ``False`` means another live process owns the lock.  Setup failures are
        errors rather than a reason to run unguarded.
        """
        if self.owns_lock:
            return True
        self._prepare_directory()
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.lock_path, flags, 0o600)
        except OSError as exc:
            raise InstanceGuardError(
                f"cannot open manager lock {self.lock_path}: {exc}"
            ) from exc
        try:
            status = os.fstat(descriptor)
            if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
                raise InstanceGuardError(f"unsafe manager lock file: {self.lock_path}")
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._owner_pid = self._read_owner(descriptor)
                return False
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            self._lock_descriptor = descriptor
            self._owner_pid = os.getpid()
            descriptor = -1
            return True
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _read_owner(descriptor: int) -> int | None:
        try:
            value = os.pread(descriptor, 64, 0).decode("ascii").strip()
            owner = int(value)
        except (OSError, UnicodeError, ValueError):
            return None
        return owner if owner > 0 else None

    def listen(self) -> bool:
        """Start the private activation listener while retaining the lock."""
        if not self.owns_lock:
            raise InstanceGuardError("activation listener requires manager ownership")
        if self._listening:
            return True

        # Only the lock owner removes a stale endpoint, so it cannot unlink a
        # socket belonging to another guarded manager.
        QLocalServer.removeServer(str(self.socket_path))
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        server.newConnection.connect(self._accept_activation_requests)
        if not server.listen(str(self.socket_path)):
            self._server = server
            return False
        self._server = server
        self._listening = True
        return True

    def request_activation(self, timeout_ms: int = 750) -> bool:
        """Ask the lock owner to raise its window, waiting for at most timeout."""
        timeout = max(0, int(timeout_ms))
        socket = QLocalSocket()
        socket.connectToServer(
            str(self.socket_path), QIODevice.OpenModeFlag.WriteOnly
        )
        if not socket.waitForConnected(timeout):
            socket.abort()
            return False
        socket.write(b"activate\n")
        socket.waitForBytesWritten(min(timeout, 250))
        socket.disconnectFromServer()
        return True

    def _accept_activation_requests(self) -> None:
        if self._server is None:
            return
        while self._server.hasPendingConnections():
            connection = self._server.nextPendingConnection()
            if connection is not None:
                connection.close()
                connection.deleteLater()
            self.activationRequested.emit()

    def close(self) -> None:
        """Release the socket and lock in race-safe order."""
        if self._server is not None:
            self._server.close()
            self._server.deleteLater()
            self._server = None
        if self._listening:
            QLocalServer.removeServer(str(self.socket_path))
            self._listening = False
        if self._lock_descriptor is not None:
            descriptor = self._lock_descriptor
            self._lock_descriptor = None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
