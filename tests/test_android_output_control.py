from __future__ import annotations

from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARCH_BIN = ROOT / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

import android_bridge  # noqa: E402


def completed(command, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        (None, False),
        ("v0.4.5", False),
        ("v0.4.6", True),
        ("v0.4.7", True),
        ("v0.4.8", True),
        ("v0.4.9", True),
        ("0.5.0-development", True),
        ("broken", False),
    ],
)
def test_output_control_capability_is_versioned(version, supported):
    assert android_bridge._module_supports_output_control(version) is supported


def test_hot_output_configuration_is_allowlisted_atomic_and_camera_safe(monkeypatch):
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def fake_adb(_adb, _serial, *args, timeout=5.0):
        del timeout
        calls.append(tuple(args))
        return completed(
            args,
            "enabled=1\nmirror=1\nrotation=270\n"
            "revision=8\nchanged=1\npersisted=1\n",
        )

    monkeypatch.setattr(android_bridge, "_adb", fake_adb)

    result = android_bridge.configure_output(
        {"enabled": True, "mirror": True, "rotation": 270}, "phone"
    )

    assert result == {
        "enabled": True,
        "mirror": True,
        "rotation": 270,
        "revision": 8,
        "changed": True,
        "persisted": True,
    }
    assert len(calls) == 1
    assert calls[0][0] == "shell"
    assert calls[0][1].startswith("su -c ")
    root_script = shlex.split(calls[0][1])[2]
    assert android_bridge.OUTPUT_CONTROL_FILE in root_script
    assert "requested_enabled='1'" in root_script
    assert "requested_mirror='1'" in root_script
    assert "requested_rotation='270'" in root_script
    assert 'mv -f "$tmp" "$control"' in root_script
    assert "am " not in root_script
    assert "force-stop" not in root_script
    assert "kill" not in root_script
    assert "/dev/video20" not in root_script
    assert "provider" not in root_script


def test_repeated_output_configuration_reports_idempotent_remote_result(monkeypatch):
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )
    monkeypatch.setattr(
        android_bridge,
        "_adb",
        lambda *_args, **_kwargs: completed(
            _args,
            "enabled=0\nmirror=0\nrotation=0\n"
            "revision=12\nchanged=0\npersisted=1\n",
        ),
    )

    result = android_bridge.configure_output({"enabled": False}, "phone")

    assert result["changed"] is False
    assert result["revision"] == 12
    assert result["enabled"] is False


def test_remote_control_script_persists_once_then_is_a_noop(monkeypatch, tmp_path):
    control_file = tmp_path / "phone output.conf"
    monkeypatch.setattr(
        android_bridge,
        "OUTPUT_CONTROL_FILE",
        str(control_file),
    )
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )

    def execute_remote_shell(_adb, _serial, *args, timeout=5.0):
        del timeout
        command = shlex.split(args[1])
        assert command[:2] == ["su", "-c"]
        return subprocess.run(
            ["sh", "-c", command[2]],
            check=False,
            capture_output=True,
            text=True,
        )

    monkeypatch.setattr(android_bridge, "_adb", execute_remote_shell)

    first = android_bridge.configure_output(
        {"enabled": False, "mirror": True, "rotation": 90}, "phone"
    )
    first_stat = control_file.stat()
    second = android_bridge.configure_output(
        {"enabled": False, "mirror": True, "rotation": 90}, "phone"
    )

    assert first["changed"] is True
    assert first["revision"] == 1
    assert second["changed"] is False
    assert second["revision"] == 1
    assert control_file.stat().st_ino == first_stat.st_ino
    assert control_file.read_text(encoding="utf-8") == (
        "version=1\n"
        "enabled=0\n"
        "mirror=1\n"
        "rotation=90\n"
        "revision=1\n"
    )


@pytest.mark.parametrize(
    "values,error",
    [
        ({}, "invalid or empty"),
        ({"unknown": True}, "invalid or empty"),
        ({"enabled": 1}, "enabled must be true or false"),
        ({"mirror": "true"}, "mirror must be true or false"),
        ({"rotation": True}, "rotation must be 0, 90, 180, or 270"),
        ({"rotation": 45}, "rotation must be 0, 90, 180, or 270"),
    ],
)
def test_output_configuration_rejects_noncanonical_values(values, error):
    with pytest.raises(android_bridge.AndroidBridgeError, match=error):
        android_bridge.configure_output(values, "phone")


def test_output_configuration_rejects_noncanonical_remote_response(monkeypatch):
    monkeypatch.setattr(
        android_bridge,
        "_require_device",
        lambda _serial, _host="": ("adb", "phone"),
    )
    monkeypatch.setattr(
        android_bridge,
        "_adb",
        lambda *_args, **_kwargs: completed(
            _args,
            "enabled=maybe\nmirror=0\nrotation=45\n"
            "revision=1\nchanged=1\npersisted=1\n",
        ),
    )

    with pytest.raises(android_bridge.AndroidBridgeError, match="invalid values"):
        android_bridge.configure_output({"enabled": True}, "phone")


def test_configure_output_cli_uses_unambiguous_output_flags(monkeypatch, capsys):
    configured: list[tuple[dict[str, bool | int], str, str]] = []
    monkeypatch.setattr(
        android_bridge,
        "configure_output",
        lambda values, serial="", host="": configured.append(
            (dict(values), serial, host)
        ),
    )
    monkeypatch.setattr(
        android_bridge,
        "collect_status",
        lambda *_args: {"available": True, "output_control": {"applied": True}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "android_bridge.py",
            "configure-output",
            "--serial",
            "phone",
            "--host",
            "192.168.1.12",
            "--output-enabled",
            "false",
            "--output-mirror",
            "true",
            "--output-rotation",
            "180",
        ],
    )

    assert android_bridge.main() == 0
    assert configured == [
        (
            {"enabled": False, "mirror": True, "rotation": 180},
            "phone",
            "192.168.1.12",
        )
    ]
    assert '"applied": true' in capsys.readouterr().out


def test_output_filters_cover_every_rotation_and_mirror_in_final_coordinates():
    common = ROOT / "android" / "vcam-module-overlay" / "bridge-output-common.sh"

    for rotation, prefix in (
        (0, "scale="),
        (90, "transpose=1,scale="),
        (180, "hflip,vflip,scale="),
        (270, "transpose=2,scale="),
    ):
        plain = subprocess.run(
            [
                "sh",
                "-c",
                f". {shlex.quote(str(common))}; "
                f"build_output_filter 0 {rotation} 1280 720 30",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        mirrored = subprocess.run(
            [
                "sh",
                "-c",
                f". {shlex.quote(str(common))}; "
                f"build_output_filter 1 {rotation} 1280 720 30",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        assert plain.startswith(prefix)
        assert "force_original_aspect_ratio=increase" in plain
        assert "crop=1280:720:(iw-ow)/2:(ih-oh)/2" in plain
        assert "pad=" not in plain
        assert "color=black" not in plain
        assert plain.endswith("fps=30")
        assert mirrored == plain.removesuffix(",fps=30") + ",hflip,fps=30"


@pytest.mark.parametrize("rotation", [90, 270])
def test_quarter_turns_fill_camera2_surface_without_black_bars(rotation):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the output-filter integration test")

    common = ROOT / "android" / "vcam-module-overlay" / "bridge-output-common.sh"
    video_filter = subprocess.run(
        [
            "sh",
            "-c",
            f". {shlex.quote(str(common))}; "
            f"build_output_filter 0 {rotation} 1280 720 30",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # A solid white source makes any letterbox/pillarbox pixels unambiguous.
    # Raw gray output also proves that the filter's final geometry remains the
    # Camera2 contract (exactly one 1280x720 plane).
    rendered = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=1280x720:r=30",
            "-vf",
            video_filter,
            "-frames:v",
            "1",
            "-pix_fmt",
            "gray",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    ).stdout

    assert len(rendered) == 1280 * 720
    assert min(rendered) >= 200


def test_selector_uses_persisted_policy_without_camera_lifecycle_commands():
    selector = (
        ROOT / "android" / "vcam-module-overlay" / "bridge-output-selector.sh"
    ).read_text(encoding="utf-8")

    assert "ANDROID_OUTPUT_CONTROL_FILE" in selector
    assert 'desired=placeholder' in selector
    assert 'start_worker "$desired"' in selector
    assert "build_output_filter" in selector
    assert "provider-supervisor" not in selector
    assert "rmmod" not in selector
    assert "insmod" not in selector
    assert "am force-stop" not in selector
    assert "/dev/video20" not in selector


def test_output_transform_hot_deployer_is_narrow_atomic_and_camera_safe():
    deployer_path = (
        ROOT
        / "android"
        / "vcam-module-overlay"
        / "deploy-output-transform-hot.sh"
    )
    deployer = deployer_path.read_text(encoding="utf-8")

    syntax = subprocess.run(
        ["sh", "-n", str(deployer_path)], capture_output=True, text=True
    )
    assert syntax.returncode == 0, syntax.stderr
    assert "ANDROID_ADB_SERIAL" in deployer
    assert 'awk \'NR > 1 && $2 == "device" { print $1; exit }\'' in deployer
    assert deployer.count(' push "$SOURCE" "$remote_stage"') == 1
    assert 'SOURCE="$SCRIPT_DIR/bridge-output-common.sh"' in deployer
    assert 'mv -f "$next" "$TARGET"' in deployer
    assert 'kill -0 "$new_selector"' in deployer
    assert "android-vcam-output-selector.pid" in deployer
    assert "android-vcam-output-worker.pid" in deployer

    for forbidden in (
        "android-vcam-provider.pid",
        "android-vcam-producer.pid",
        "provider-supervisor",
        "am force-stop",
        "am stopservice",
        "/dev/video20",
        "rmmod",
        "insmod",
        "killall",
        "pkill",
    ):
        assert forbidden not in deployer


def test_android_module_scripts_are_posix_shell_syntax_clean():
    scripts = sorted((ROOT / "android" / "vcam-module-overlay").glob("*.sh"))
    assert scripts
    for script in scripts:
        result = subprocess.run(
            ["sh", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"
