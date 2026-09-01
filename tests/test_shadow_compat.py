from __future__ import annotations

from pathlib import Path
import sys


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import shadow  # noqa: E402
from common import resolve_capture_device  # noqa: E402


def test_legacy_takeover_requires_both_explicit_gates() -> None:
    assert not shadow.legacy_shadow_enabled({})
    assert not shadow.legacy_shadow_enabled({"SHADOW_ORIGINAL": "1"})
    assert not shadow.legacy_shadow_enabled({"LEGACY_SHADOW": "1"})
    assert shadow.legacy_shadow_enabled(
        {"LEGACY_SHADOW": "1", "SHADOW_ORIGINAL": "1"}
    )


def test_clean_capture_resolution_never_redirects_the_physical_camera(tmp_path) -> None:
    physical = tmp_path / "physical-camera"
    physical.touch()
    values = {
        "PHYSICAL_CAMERA": str(physical),
        "SHADOW_ORIGINAL": "1",
        "STATE_DIR": str(tmp_path / "state"),
    }

    assert resolve_capture_device(values) == physical


def test_explicit_legacy_capture_resolution_uses_the_preserved_node(tmp_path) -> None:
    physical = tmp_path / "video7"
    physical.touch()
    state = tmp_path / "state"
    values = {
        "PHYSICAL_CAMERA": str(physical),
        "LEGACY_SHADOW": "1",
        "SHADOW_ORIGINAL": "1",
        "STATE_DIR": str(state),
    }

    assert resolve_capture_device(values, state) == state / "source" / "video7"


def test_clean_apply_only_points_the_alias_at_the_configured_loopback(
    monkeypatch, tmp_path
) -> None:
    loop = tmp_path / "video42"
    loop.touch()
    alias = tmp_path / "deep-live-cam"
    states: list[dict[str, object]] = []
    monkeypatch.setattr(shadow, "ALIAS", alias)
    monkeypatch.setattr(shadow, "write_state", lambda _name, state: states.append(state))

    assert shadow.apply({"LOOPBACK_NODE": str(loop)}) == 0
    assert alias.is_symlink()
    assert alias.readlink() == Path("video42")
    assert states[-1]["status"] == "ready"
    assert states[-1]["mode"] == "independent"
