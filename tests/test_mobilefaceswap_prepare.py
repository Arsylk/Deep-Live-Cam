from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import prepare_mobilefaceswap_baseline as prepare  # noqa: E402


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for info, content in members:
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_safe_extract_copies_only_requested_regular_files(tmp_path):
    archive_path = tmp_path / "assets.tar"
    wanted = tarfile.TarInfo("checkpoints/wanted.bin")
    ignored = tarfile.TarInfo("checkpoints/ignored.bin")
    _write_tar(archive_path, [(wanted, b"wanted"), (ignored, b"ignored")])

    extracted = prepare.safe_extract_members(
        archive_path,
        tmp_path / "out",
        {"checkpoints/wanted.bin": len(b"wanted")},
    )

    assert extracted["checkpoints/wanted.bin"].read_bytes() == b"wanted"
    assert not (tmp_path / "out/checkpoints/ignored.bin").exists()


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "a\\b"])
def test_safe_extract_rejects_unsafe_member_paths(tmp_path, member_name):
    archive_path = tmp_path / "unsafe.tar"
    member = tarfile.TarInfo(member_name)
    _write_tar(archive_path, [(member, b"unsafe")])

    with pytest.raises(prepare.PreparationError, match="unsafe|backslash"):
        prepare.safe_extract_members(
            archive_path, tmp_path / "out", {"safe.bin": 1}
        )
    assert not (tmp_path.parent / "escape").exists()


def test_safe_extract_rejects_links_even_when_not_requested(tmp_path):
    archive_path = tmp_path / "links.tar"
    link = tarfile.TarInfo("unrequested-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../outside"
    _write_tar(archive_path, [(link, b"")])

    with pytest.raises(prepare.PreparationError, match="links are not accepted"):
        prepare.safe_extract_members(
            archive_path, tmp_path / "out", {"wanted.bin": 1}
        )


def test_official_archive_hash_mismatch_is_rejected(tmp_path):
    archive = tmp_path / "checkpoints.tar"
    archive.write_bytes(b"not the official archive")

    with pytest.raises(prepare.PreparationError, match="SHA-256 mismatch"):
        prepare.verify_official_archive(archive)


def test_manifest_hashes_artifacts_and_marks_baseline_experimental(tmp_path):
    source = tmp_path / "source.jpg"
    archive = tmp_path / "checkpoints.tar"
    onnx = tmp_path / "model.onnx"
    param = tmp_path / "model.param"
    model = tmp_path / "model.bin"
    pnnx = tmp_path / "pnnx"
    fixtures = {
        source: b"source",
        archive: b"archive",
        onnx: b"onnx",
        param: b"param",
        model: b"weights",
        pnnx: b"executable",
    }
    for path, content in fixtures.items():
        path.write_bytes(content)

    manifest = prepare.build_manifest(
        source_path=source,
        archive_path=archive,
        archive_sha256=hashlib.sha256(b"archive").hexdigest(),
        upstream_dir=tmp_path / "MobileFaceSwap",
        upstream_info={"commit": prepare.OFFICIAL_UPSTREAM_COMMIT},
        size=256,
        onnx_path=onnx,
        ncnn_param_path=param,
        ncnn_model_path=model,
        pnnx_path=pnnx,
    )

    assert manifest["quality_status"] == "experimental"
    assert manifest["default_backend"] is False
    assert manifest["auto_select"] is False
    assert manifest["source_sha256"] == hashlib.sha256(b"source").hexdigest()
    assert manifest["model_hashes"] == {
        "onnx": hashlib.sha256(b"onnx").hexdigest(),
        "ncnn_param": hashlib.sha256(b"param").hexdigest(),
        "ncnn_model": hashlib.sha256(b"weights").hexdigest(),
    }
    assert "No public native-256" in manifest["training_size_warning"]
    assert "no explicit model-weight license" in manifest["license_warning"]
    assert manifest["contract"]["outputs"][0]["uncomposited"] is True
    assert manifest["contract"]["outputs"][1]["name"] == "alpha"
    assert manifest["artifacts"]["ncnn"]["pnnx_sha256"] == hashlib.sha256(
        b"executable"
    ).hexdigest()
    json.dumps(manifest)


def test_parser_rejects_unsupported_size():
    with pytest.raises(SystemExit):
        prepare.parse_args(
            [
                "--checkpoint-tar",
                "checkpoints.tar",
                "--upstream-dir",
                "MobileFaceSwap",
                "--source-image",
                "source.jpg",
                "--output-dir",
                "cache",
                "--size",
                "128",
            ]
        )


def test_path_validation_does_not_import_or_require_paddle(tmp_path):
    upstream = tmp_path / "upstream"
    for relative in prepare.AUDITED_UPSTREAM_FILES:
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    archive = tmp_path / "checkpoints.tar"
    source = tmp_path / "source.jpg"
    archive.write_bytes(b"fixture")
    source.write_bytes(b"fixture")
    output = tmp_path / "cache"
    pnnx = tmp_path / "pnnx"
    pnnx.write_bytes(b"fixture")

    args = argparse.Namespace(
        checkpoint_tar=archive,
        upstream_dir=upstream,
        source_image=source,
        output_dir=output,
        size=224,
        pnnx=pnnx,
    )
    with pytest.raises(prepare.PreparationError, match="not executable"):
        prepare.validate_cli_paths(args)

    pnnx.chmod(pnnx.stat().st_mode | 0o100)
    prepare.validate_cli_paths(args)
    assert "paddle" not in prepare.__dict__
