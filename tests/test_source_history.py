from __future__ import annotations

import json
from pathlib import Path
import sys


ARCH_BIN = Path(__file__).resolve().parents[1] / "arch-linux" / "bin"
if str(ARCH_BIN) not in sys.path:
    sys.path.insert(0, str(ARCH_BIN))

from source_history import SourceHistoryStore  # noqa: E402


def test_history_persists_current_picture(tmp_path):
    store = SourceHistoryStore(tmp_path)
    current = store.remember(b"first-image", "first.png")
    store.remember(b"second-image", "second.jpg")

    reloaded = SourceHistoryStore(tmp_path)

    assert [entry.filename for entry in reloaded.entries()] == [
        "second.jpg",
        "first.png",
    ]
    assert reloaded.active_entry().filename == "second.jpg"
    assert current.cache_path.read_bytes() == b"first-image"


def test_an_existing_picture_can_be_reactivated_without_rewriting_it(tmp_path):
    store = SourceHistoryStore(tmp_path)
    first = store.remember(b"first-image", "first.png")
    store.remember(b"second-image", "second.jpg")
    original_mtime = first.cache_path.stat().st_mtime_ns

    activated = store.activate(first.identifier)
    reloaded = SourceHistoryStore(tmp_path)

    assert activated.identifier == first.identifier
    assert reloaded.active_entry().identifier == first.identifier
    assert first.cache_path.stat().st_mtime_ns == original_mtime


def test_history_deduplicates_and_evicts_old_cache_files(tmp_path):
    store = SourceHistoryStore(tmp_path, limit=2)
    first = store.remember(b"same-picture", "old-name.png")
    store.remember(b"other-picture", "other.png")
    renamed = store.remember(b"same-picture", "new-name.jpg")
    store.remember(b"third-picture", "third.png")

    assert renamed.identifier == first.identifier
    assert [entry.filename for entry in store.entries()] == [
        "third.png",
        "new-name.jpg",
    ]
    assert not (tmp_path / ".history.json.tmp").exists()
    cached_pictures = [path for path in tmp_path.iterdir() if path.name != "history.json"]
    assert sorted(path.suffix for path in cached_pictures) == [".jpg", ".png"]


def test_history_ignores_missing_and_unsafe_metadata_entries(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "valid.png").write_bytes(b"valid")
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "version": 1,
                "active_id": "unsafe",
                "items": [
                    {
                        "id": "unsafe",
                        "filename": "unsafe.png",
                        "cache_name": "../unsafe.png",
                        "used_at": 1,
                    },
                    {
                        "id": "missing",
                        "filename": "missing.png",
                        "cache_name": "missing.png",
                        "used_at": 2,
                    },
                    {
                        "id": "valid",
                        "filename": "valid.png",
                        "cache_name": "valid.png",
                        "used_at": 3,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    store = SourceHistoryStore(tmp_path)

    assert [entry.identifier for entry in store.entries()] == ["valid"]
    assert store.active_entry() is None
