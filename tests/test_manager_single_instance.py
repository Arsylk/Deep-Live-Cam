"""A second native manager must never start another reconciliation loop."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "arch-linux" / "bin"
MODULE_PATH = BIN / "tester.py"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from dlc_manager.single_instance import (  # noqa: E402
    InstanceGuardError,
    ManagerInstanceGuard,
    default_instance_directory,
)


def _application() -> QApplication:
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def test_runtime_directory_is_stable_and_per_user(tmp_path):
    assert default_instance_directory(
        environment={"XDG_RUNTIME_DIR": str(tmp_path)}, user_id=123
    ) == tmp_path / "deep-live-cam-manager"
    assert default_instance_directory(
        environment={"XDG_RUNTIME_DIR": "relative"}, user_id=987654
    ).name == "deep-live-cam-manager-987654"


def test_kernel_lock_allows_only_one_manager_and_recovers_after_exit(tmp_path):
    _application()
    primary = ManagerInstanceGuard(tmp_path / "instance")
    duplicate = ManagerInstanceGuard(tmp_path / "instance")
    successor = ManagerInstanceGuard(tmp_path / "instance")
    try:
        assert primary.try_acquire() is True
        assert duplicate.try_acquire() is False
        assert duplicate.owner_pid == os.getpid()

        primary.close()
        assert successor.try_acquire() is True
    finally:
        duplicate.close()
        successor.close()


def test_guard_refuses_a_symlinked_runtime_directory(tmp_path):
    _application()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "instance"
    link.symlink_to(target, target_is_directory=True)
    guard = ManagerInstanceGuard(link)

    with pytest.raises(InstanceGuardError, match="unsafe manager runtime"):
        guard.try_acquire()

    assert guard.owns_lock is False


def test_duplicate_requests_activation_without_becoming_an_owner(tmp_path):
    app = _application()
    primary = ManagerInstanceGuard(tmp_path / "instance")
    duplicate = ManagerInstanceGuard(tmp_path / "instance")
    activations: list[bool] = []
    try:
        assert primary.try_acquire() is True
        assert primary.listen() is True
        primary.activationRequested.connect(lambda: activations.append(True))

        assert duplicate.try_acquire() is False
        assert duplicate.request_activation(timeout_ms=250) is True
        app.processEvents()

        assert activations == [True]
        assert duplicate.owns_lock is False
    finally:
        duplicate.close()
        primary.close()


def test_main_exits_before_constructing_a_duplicate_controller(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "arch_tester_single_instance", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    tester = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tester)
    calls: list[str] = []

    class FakeApplication:
        def __init__(self, _arguments):
            calls.append("application")

        def installEventFilter(self, _guard):
            pass

        def setApplicationName(self, _name):
            pass

        def setApplicationDisplayName(self, _name):
            pass

        def setDesktopFileName(self, _name):
            pass

    class DuplicateGuard:
        owner_pid = 4242

        def try_acquire(self):
            calls.append("lock")
            return False

        def request_activation(self):
            calls.append("activate")
            return True

        def close(self):
            calls.append("close")

    monkeypatch.setattr(tester, "QApplication", FakeApplication)
    monkeypatch.setattr(tester, "WheelValueGuard", lambda _application: object())
    monkeypatch.setattr(tester, "ManagerInstanceGuard", DuplicateGuard)
    monkeypatch.setattr(
        tester,
        "ManagerWindow",
        lambda _config: calls.append("manager") or object(),
    )
    monkeypatch.setattr(tester, "load_env_file", lambda _path: {})
    monkeypatch.setattr(sys, "argv", ["tester.py"])

    assert tester.main() == 0
    assert calls == ["application", "lock", "activate", "close"]
