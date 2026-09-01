#!/usr/bin/env python3
"""Maintain the processed-camera alias, with an explicit legacy takeover mode.

The normal layout is one independent upstream v4l2loopback device at
``/dev/video42`` with ``/dev/deep-live-cam`` as its stable alias.  That path
never unbinds, hides, renumbers, or rewrites links for a physical camera.

Only the combination ``LEGACY_SHADOW=1`` and ``SHADOW_ORIGINAL=1`` enables the
retained compatibility implementation below.  In that legacy mode the
v4l2loopback devices ARE /dev/video0 and
/dev/video1 (video_nr=0,1, so even stat() reports the original 81:0/81:1),
and the real camera is rebound onto higher minors.  The clean installer never
selects this path unless ``--legacy-shadow`` was supplied.

The apply action is idempotent.  Clean installs normally get their alias
directly from udev; only explicit legacy mode asks udev to re-run this helper.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, DEFAULT_STATE_DIR, load_env_file, write_state

SYSFS = Path("/sys/class/video4linux")
MEDIA_SYSFS = Path("/sys/class/media")
ALIAS = Path("/dev/deep-live-cam")
DROIDCAM_ALIAS = Path("/dev/droidcam")
UVC_DRIVER = Path("/sys/bus/usb/drivers/uvcvideo")
TAKEOVER_NAMES = ("video0", "video1")


def legacy_shadow_enabled(config: dict[str, str]) -> bool:
    """Require an explicit second gate before any invasive compatibility work."""
    return (
        str(config.get("LEGACY_SHADOW", "0")) == "1"
        and str(config.get("SHADOW_ORIGINAL", "0")) == "1"
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)


def unescape_mount(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def mountpoints() -> set[str]:
    points: set[str] = set()
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return points
    for line in lines:
        fields = line.split(" ")
        if len(fields) >= 5:
            points.add(unescape_mount(fields[4]))
    return points


def unmount(path: Path) -> None:
    """Fully unmount a path, including stacked and busy mounts (fail loudly)."""
    for _attempt in range(10):
        if str(path) not in mountpoints():
            return
        if run(["umount", str(path)]).returncode == 0:
            continue
        run(["umount", "-l", str(path)])
        time.sleep(0.1)
    raise RuntimeError(f"could not unmount {path}")


def bind_mount(source: Path, target: Path) -> None:
    result = run(["mount", "--bind", str(source), str(target)])
    if result.returncode != 0:
        raise RuntimeError(f"mount --bind {source} -> {target}: {result.stderr.strip()}")


def create_char_node(path: Path, rdev: int, mode: int = 0o0660) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if not path.exists():
        os.mknod(path, stat.S_IFCHR | mode, rdev)
    # mknod applies the process umask; force the intended mode and group.
    os.chmod(path, mode)
    try:
        shutil.chown(path, group="video")
    except (KeyError, PermissionError):
        pass


def sysfs_device(name: str) -> Path | None:
    """Interface sysfs dir of a video node (e.g. .../usb2/2-4/2-4:1.0)."""
    try:
        return (SYSFS / name / "device").resolve()
    except OSError:
        return None


def sysfs_group(name: str) -> Path | None:
    """Physical USB device dir shared by all video nodes of one camera."""
    interface = sysfs_device(name)
    return interface.parent if interface is not None else None


def sysfs_rdev(name: str) -> int:
    raw = (SYSFS / name / "dev").read_text(encoding="utf-8").strip()
    major_s, minor_s = raw.split(":")
    return os.makedev(int(major_s), int(minor_s))


def is_virtual(name: str) -> bool:
    # v4l2loopback marks its class devices with custom sysfs attributes
    # (format/buffers/...). Keep the attribute check as the authoritative
    # test and the standard virtual-device path as a conservative fallback.
    if (SYSFS / name / "format").exists():
        return True
    device = sysfs_device(name)
    return device is not None and str(device).startswith("/sys/devices/virtual/")


def discover_virtual() -> list[str]:
    try:
        entries = list(SYSFS.iterdir())
    except OSError:
        return []
    return sorted(e.name for e in entries if e.name.startswith("video") and is_virtual(e.name))


def configured_droidcam_name(config: dict[str, str]) -> str | None:
    """Return the dedicated loopback sysname without trusting a path value."""
    if str(config.get("DROIDCAM_COMPAT", "0")) != "1":
        return None
    raw = str(config.get("DROIDCAM_VIDEO_NR", "50"))
    if not raw.isdigit():
        return None
    number = int(raw)
    if number < 10 or number > 255 or number == 42:
        return None
    return f"video{number}"


def takeover_layout_ready(virtual: list[str], camera: list[str]) -> bool:
    """Auxiliary loopbacks are valid as long as the shadow pair is intact."""
    required = set(TAKEOVER_NAMES)
    return required.issubset(virtual) and required.isdisjoint(camera)


def discover_camera(group: Path | None) -> list[str]:
    if group is None:
        return []
    try:
        entries = list(SYSFS.iterdir())
    except OSError:
        return []
    return sorted(
        e.name
        for e in entries
        if e.name.startswith("video")
        and not is_virtual(e.name)
        and sysfs_group(e.name) == group
    )


def discover_camera_group(serial: str) -> Path | None:
    """Find the configured physical camera directly from sysfs/udev.

    During boot the previous shadow pass may already have removed every /dev
    node and persistent symlink.  Sysfs still has the physical UVC devices, so
    use their udev ID_SERIAL property and fall back only when exactly one
    physical USB camera group is present.
    """
    groups: set[Path] = set()
    serial_groups: set[Path] = set()
    try:
        entries = list(SYSFS.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.startswith("video") or is_virtual(entry.name):
            continue
        group = sysfs_group(entry.name)
        if group is None:
            continue
        groups.add(group)
        if serial:
            result = run(
                ["udevadm", "info", "--query=property", "--path", str(entry)]
            )
            properties = dict(
                line.split("=", 1)
                for line in result.stdout.splitlines()
                if "=" in line
            )
            if properties.get("ID_SERIAL") == serial:
                serial_groups.add(group)
    if len(serial_groups) == 1:
        return next(iter(serial_groups))
    if len(groups) == 1:
        return next(iter(groups))
    return None


def derive_serial(physical: Path, recorded: dict[str, Any]) -> str:
    match = re.match(r"usb-(.+)-video-index\d+$", physical.name)
    if match:
        return match.group(1)
    return str(recorded.get("serial", ""))


def read_state(state_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((state_dir / "shadow.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def ensure_state_dirs(state_dir: Path) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.chown(state_dir, user="deep-live-cam", group="deep-live-cam")
    except (KeyError, PermissionError):
        pass
    source = state_dir / "source"
    source.mkdir(exist_ok=True)
    return source


def point_alias(target_name: str | None) -> None:
    try:
        ALIAS.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return
    if target_name is not None:
        ALIAS.symlink_to(target_name)


def discover_droidcam_virtual(
    config: dict[str, str], virtual: list[str]
) -> str | None:
    """Return the configured auxiliary node when it is a live loopback."""
    name = configured_droidcam_name(config)
    if name is None or name not in virtual:
        return None
    return name


def point_droidcam_alias(target_name: str | None) -> None:
    """Restore the stable producer alias after a takeover or udev event."""
    if DROIDCAM_ALIAS.exists() and not DROIDCAM_ALIAS.is_symlink():
        print(
            f"deep-live-cam shadow: refusing to replace non-symlink {DROIDCAM_ALIAS}",
            file=sys.stderr,
        )
        return
    try:
        DROIDCAM_ALIAS.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return
    if target_name is None or not (SYSFS / target_name / "dev").exists():
        return
    node = Path("/dev") / target_name
    if not node.exists():
        create_char_node(node, sysfs_rdev(target_name))
    DROIDCAM_ALIAS.symlink_to(target_name)


def cleanup_legacy(state_dir: Path, recorded: dict[str, Any]) -> None:
    """Remove any bind-mount leftovers from the previous shadow design."""
    stale = {Path(p) for p in recorded.get("nodes", [])}
    stale |= {Path(p) for p in recorded.get("preserved", [])}
    if recorded.get("loop_source"):
        stale.add(Path(recorded["loop_source"]))
    stale.add(state_dir / "loop")
    source_dir = state_dir / "source"
    if source_dir.is_dir():
        stale |= set(source_dir.glob("video*"))
    for path in sorted(stale):
        unmount(path)
    for staging in Path("/dev").glob(".deep-live-cam-source-*"):
        try:
            staging.unlink()
        except OSError:
            pass


def camera_links(serial: str, cam_names: set[str]) -> dict[str, int]:
    """Map each persistent v4l symlink of the real camera to its stream index.

    Matches by-id links through the serial and by-path links through their
    current target, so both families keep working after one was repointed.
    """
    links: dict[str, int] = {}
    # Ask udev for the links it would create even when /dev/v4l was removed by
    # a prior shadow pass.  This makes boot/reset recovery independent of the
    # current contents of /dev.
    for name in cam_names:
        result = run(
            ["udevadm", "info", "--query=symlink", "--path", str(SYSFS / name)]
        )
        for relative in result.stdout.split():
            match = re.search(r"-video-index(\d+)$", relative)
            if match and (not serial or serial in relative or "/by-path/" in f"/{relative}"):
                links[str(Path("/dev") / relative)] = int(match.group(1))
    for family in ("/dev/v4l/by-id", "/dev/v4l/by-path"):
        directory = Path(family)
        if not directory.is_dir():
            continue
        for link in directory.iterdir():
            match = re.search(r"-video-index(\d+)$", link.name)
            if not match:
                continue
            target_name = os.path.basename(os.path.realpath(link))
            if serial and serial in link.name:
                links[str(link)] = int(match.group(1))
            elif target_name in cam_names:
                links[str(link)] = int(match.group(1))
    return links


def takeover_module(cam_names: list[str]) -> None:
    """Rebind the physical camera so the loopback can own video0/video1.

    The camera's USB interfaces are unbound, the module is reloaded (it then
    registers video_nr=0,1 on the freed minors), and the camera is bound
    back, landing on the next free minors.
    """
    interfaces = sorted(
        {device.name for name in cam_names if (device := sysfs_device(name)) is not None}
    )
    if not interfaces:
        raise RuntimeError("no USB interfaces found for the physical camera")
    for interface in interfaces:
        try:
            (UVC_DRIVER / "unbind").write_text(interface, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"could not unbind {interface}: {exc}") from exc
    try:
        result = run(["modprobe", "-r", "v4l2loopback"])
        if result.returncode != 0:
            raise RuntimeError(
                "modprobe -r v4l2loopback failed (receiver still holding it?): "
                + result.stderr.strip()
            )
        result = run(["modprobe", "v4l2loopback"])
        if result.returncode != 0:
            raise RuntimeError("modprobe v4l2loopback failed: " + result.stderr.strip())
    finally:
        for interface in interfaces:
            try:
                (UVC_DRIVER / "bind").write_text(interface, encoding="utf-8")
            except OSError as exc:
                print(f"deep-live-cam shadow: rebind {interface} failed: {exc}", file=sys.stderr)
        run(["udevadm", "settle", "--timeout=10"])


def wait_for_camera(group: Path | None, timeout: float = 10.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        names = discover_camera(group)
        if names:
            return names
        time.sleep(0.2)
    return []


def hide_node(name: str) -> None:
    try:
        (Path("/dev") / name).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"deep-live-cam shadow: could not hide /dev/{name}: {exc}", file=sys.stderr)


def hide_media_nodes(group: Path | None) -> list[str]:
    hidden: list[str] = []
    if group is None or not MEDIA_SYSFS.is_dir():
        return hidden
    for entry in MEDIA_SYSFS.iterdir():
        try:
            if (entry / "device").resolve().parent == group:
                hide_node(entry.name)
                hidden.append(entry.name)
        except OSError:
            continue
    return hidden


def apply_clean(config: dict[str, str]) -> int:
    """Publish only the stable alias for the independent loopback device."""
    loop = Path(config.get("LOOPBACK_NODE", "/dev/video42"))
    if ALIAS.exists() and not ALIAS.is_symlink():
        detail = f"refusing to replace non-symlink {ALIAS}"
        write_state("shadow", {"status": "error", "detail": detail})
        print(f"deep-live-cam virtual camera: {detail}", file=sys.stderr)
        return 1
    point_alias(loop.name if loop.exists() else None)
    status = "ready" if loop.exists() else "waiting_virtual"
    write_state(
        "shadow",
        {
            "status": status,
            "mode": "independent",
            "device": str(loop),
            "alias": f"{ALIAS} -> {loop.name}" if loop.exists() else None,
        },
    )
    return 0


def apply(config: dict[str, str]) -> int:
    state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    if not legacy_shadow_enabled(config):
        return apply_clean(config)

    source_dir = ensure_state_dirs(state_dir)
    physical = Path(config.get("PHYSICAL_CAMERA", ""))
    recorded = read_state(state_dir)
    serial = derive_serial(physical, recorded)
    cleanup_legacy(state_dir, recorded)

    virtual = discover_virtual()
    group: Path | None = None
    if physical.exists():
        real_name = os.path.basename(os.path.realpath(physical))
        if real_name not in virtual and (SYSFS / real_name).exists():
            group = sysfs_group(real_name)
    if group is None and recorded.get("usb_group"):
        candidate = Path(recorded["usb_group"])
        if candidate.exists():
            group = candidate
    if group is None:
        group = discover_camera_group(serial)

    if group is None:
        processed = next((name for name in TAKEOVER_NAMES if name in virtual), None)
        point_alias(processed)
        point_droidcam_alias(configured_droidcam_name(config))
        write_state(
            "shadow",
            {"status": "waiting_camera", "detail": f"{physical} is absent", "serial": serial},
        )
        return 0

    if not virtual:
        write_state("shadow", {"status": "waiting_virtual", "detail": "v4l2loopback not loaded"})
        return 0

    cam_names = discover_camera(group)
    if not cam_names:
        write_state("shadow", {"status": "waiting_camera", "detail": "camera bound but no nodes yet"})
        return 0

    # Take over video0/video1 if the loopback does not already own them.
    if not takeover_layout_ready(virtual, cam_names):
        try:
            takeover_module(cam_names)
        except RuntimeError as exc:
            write_state("shadow", {"status": "error", "detail": str(exc)})
            print(f"deep-live-cam shadow: takeover failed: {exc}", file=sys.stderr)
            return 1
        virtual = discover_virtual()
        cam_names = wait_for_camera(group)
        if not set(TAKEOVER_NAMES).issubset(virtual):
            write_state(
                "shadow",
                {
                    "status": "error",
                    "detail": f"loopback owns {virtual}, missing shadow pair {TAKEOVER_NAMES}",
                },
            )
            return 1
        if not cam_names:
            write_state("shadow", {"status": "waiting_camera", "detail": "camera lost in takeover"})
            return 0

    # Map stream index -> current real node, preferring the persistent links.
    links = camera_links(serial, set(cam_names))
    mapping: dict[int, str] = {}
    for link, index in links.items():
        target_name = os.path.basename(os.path.realpath(link))
        if target_name in cam_names:
            mapping[index] = target_name
    # Persistent links may be in a transient, partially repointed state when
    # this oneshot is retriggered by udev.  Keep any trustworthy matches, but
    # fill every missing stream index from the remaining camera nodes.  A
    # partial mapping used to preserve only video1 and leave video0 as a plain
    # device node on nodev /run, which then failed to open with EACCES.
    unused_names = [name for name in cam_names if name not in mapping.values()]
    missing_indices = [index for index in range(len(cam_names)) if index not in mapping]
    for index, name in zip(missing_indices, unused_names):
        mapping[index] = name

    # Preserve the real nodes for the sender. /run is tmpfs with the nodev
    # mount flag, so plain mknod copies there can never be opened (EACCES for
    # everyone including root). Bind-mount a staging node from devtmpfs
    # instead: the bind carries the devtmpfs mount, which has no nodev flag.
    preserved: list[str] = []
    for index, name in sorted(mapping.items()):
        target = source_dir / f"video{index}"
        unmount(target)
        try:
            target.unlink()
        except OSError:
            pass
        staging = Path("/dev") / f".deep-live-cam-source-{name}"
        create_char_node(staging, sysfs_rdev(name))
        create_char_node(target, sysfs_rdev(name))
        bind_mount(staging, target)
        run(["mount", "--make-private", str(target)])
        try:
            staging.unlink()
        except OSError:
            pass
        preserved.append(str(target))
    for name in cam_names:
        hide_node(name)
    hidden_media = hide_media_nodes(group)

    primary = f"video{min(mapping)}"

    # Keep every loopback under the virtual device hierarchy.  Moving a live
    # class device under a USB interface couples its lifetime and power-order
    # dependencies to hardware that this service deliberately unbinds.  The
    # stable public identity comes from V4L2 metadata and udev links instead.
    droidcam_name = discover_droidcam_virtual(config, virtual)

    # Wait for earlier hide/trigger events before recreating canonical nodes;
    # a delayed remove event must not unlink a freshly-created node.
    result = run(["udevadm", "settle", "--timeout=10"])
    if result.returncode != 0:
        print(
            "deep-live-cam shadow: udev settle timed out during takeover",
            file=sys.stderr,
        )

    # The kernel devices still exist in sysfs even when udev removed their old
    # paths. Restore canonical nodes, persistent links, and the convenience
    # alias only after the udev queue is empty.
    for name in TAKEOVER_NAMES:
        if (SYSFS / name / "dev").exists():
            create_char_node(Path("/dev") / name, sysfs_rdev(name))
    point_droidcam_alias(droidcam_name)
    for link, index in links.items():
        path = Path(link)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink()
        except OSError:
            pass
        path.symlink_to(f"../../video{index}")
    point_alias(primary)

    write_state(
        "shadow",
        {
            "status": "shadowed",
            "serial": serial,
            "usb_group": str(group),
            "mapping": {str(index): name for index, name in sorted(mapping.items())},
            "camera_nodes": sorted(cam_names),
            "preserved": preserved,
            "links": sorted(links),
            "hidden_media": hidden_media,
            "alias": f"{ALIAS} -> {primary}",
            "droidcam_alias": (
                f"{DROIDCAM_ALIAS} -> {droidcam_name}" if droidcam_name else "disabled/unavailable"
            ),
        },
    )
    print(
        "deep-live-cam shadow: processed feed now IS "
        + ", ".join(f"/dev/video{i}" for i in sorted(mapping))
        + f" (real camera hidden at {', '.join(cam_names)})"
    )
    return 0


def remove(config: dict[str, str], status: str = "inactive") -> int:
    state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    recorded = read_state(state_dir)
    cleanup_legacy(state_dir, recorded)

    virtual = discover_virtual()
    droidcam_name = discover_droidcam_virtual(config, virtual)
    if droidcam_name is not None:
        run(["udevadm", "settle", "--timeout=10"])
    point_droidcam_alias(droidcam_name)

    source_dir = state_dir / "source"
    if source_dir.is_dir():
        for entry in source_dir.glob("video*"):
            try:
                entry.unlink()
            except OSError:
                pass

    # Restore the real camera's device nodes and persistent links.
    mapping = {int(k): v for k, v in recorded.get("mapping", {}).items()}
    for index, name in sorted(mapping.items()):
        node = Path("/dev") / name
        if not node.exists() and (SYSFS / name / "dev").exists():
            create_char_node(node, sysfs_rdev(name), mode=0o0660)
            run(["udevadm", "trigger", f"--sysname-match={name}"])
    run(["udevadm", "settle", "--timeout=10"])
    for link in recorded.get("links", []):
        match = re.search(r"-video-index(\d+)$", link)
        if not match:
            continue
        index = int(match.group(1))
        real_name = mapping.get(index)
        if real_name is None:
            continue
        path = Path(link)
        try:
            path.unlink()
        except OSError:
            pass
        path.symlink_to(f"../../{real_name}")

    loop = Path(config.get("LOOPBACK_NODE", "/dev/video42"))
    point_alias(loop.name if loop.exists() else None)
    write_state("shadow", {"status": status})
    print(f"deep-live-cam shadow: {status}")
    print("note: topology changes take effect after a controlled reboot", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "remove"))
    args = parser.parse_args()
    if os.geteuid() != 0:
        print("deep-live-cam shadow: must run as root", file=sys.stderr)
        return 1
    config = load_env_file(DEFAULT_CONFIG)
    # Serialize runs: udev triggers and manual restarts may overlap.
    state_dir = Path(config.get("STATE_DIR", str(DEFAULT_STATE_DIR)))
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(state_dir / "shadow.lock", os.O_WRONLY | os.O_CREAT, 0o644)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    if args.action == "apply":
        return apply(config)
    return remove(config)


if __name__ == "__main__":
    raise SystemExit(main())
