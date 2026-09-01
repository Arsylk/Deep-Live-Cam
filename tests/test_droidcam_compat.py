from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "arch-linux" / "bin" / "droidcam-compat"


def fake_frontend(tmp_path: Path) -> Path:
    frontend = tmp_path / "frontend"
    frontend.write_text(
        "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n",
        encoding="utf-8",
    )
    frontend.chmod(0o755)
    return frontend


def test_gui_wrapper_injects_dedicated_device(tmp_path: Path) -> None:
    frontend = fake_frontend(tmp_path)
    env = os.environ.copy()
    env.update(
        DROIDCAM_REAL_GUI=str(frontend),
        DROIDCAM_VIDEO_DEVICE="/dev/null",
    )

    result = subprocess.run(
        [str(WRAPPER), "ordinary-argument"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["<-dev=/dev/null>", "<ordinary-argument>"]


def test_explicit_device_is_not_duplicated(tmp_path: Path) -> None:
    frontend = fake_frontend(tmp_path)
    env = os.environ.copy()
    env.update(
        DROIDCAM_REAL_GUI=str(frontend),
        DROIDCAM_VIDEO_DEVICE="/does/not/exist",
    )

    result = subprocess.run(
        [str(WRAPPER), "-dev=/dev/null", "ordinary-argument"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == ["<-dev=/dev/null>", "<ordinary-argument>"]


def test_cli_symlink_selects_cli_frontend(tmp_path: Path) -> None:
    frontend = fake_frontend(tmp_path)
    cli_wrapper = tmp_path / "droidcam-cli"
    cli_wrapper.symlink_to(WRAPPER)
    env = os.environ.copy()
    env.update(
        DROIDCAM_REAL_CLI=str(frontend),
        DROIDCAM_VIDEO_DEVICE="/dev/null",
    )

    result = subprocess.run(
        [str(cli_wrapper), "127.0.0.1", "4747"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.splitlines() == [
        "<-dev=/dev/null>",
        "<127.0.0.1>",
        "<4747>",
    ]


def test_missing_dedicated_node_fails_before_frontend(tmp_path: Path) -> None:
    frontend = fake_frontend(tmp_path)
    env = os.environ.copy()
    env.update(
        DROIDCAM_REAL_GUI=str(frontend),
        DROIDCAM_VIDEO_DEVICE=str(tmp_path / "missing-video-node"),
    )

    result = subprocess.run(
        [str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 66
    assert "dedicated camera node is unavailable" in result.stderr
