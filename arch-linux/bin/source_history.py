#!/usr/bin/env python3
"""Persistent, local history for source-face pictures used by the control app."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


HISTORY_VERSION = 1
DEFAULT_HISTORY_LIMIT = 8
SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class SourceHistoryEntry:
    identifier: str
    filename: str
    cache_path: Path
    used_at: float


def default_source_history_directory() -> Path:
    """Return a per-user data directory that survives reboots."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "deep-live-cam" / "source-history"


class SourceHistoryStore:
    """Keep a bounded, content-addressed history without relying on originals."""

    def __init__(self, directory: Path, limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if limit < 1:
            raise ValueError("history limit must be positive")
        self.directory = Path(directory)
        self.limit = limit
        self.metadata_path = self.directory / "history.json"
        self._items: list[dict[str, Any]] = []
        self._active_id: str | None = None
        self._load()

    def entries(self) -> list[SourceHistoryEntry]:
        return [self._entry(item) for item in self._items]

    def active_entry(self) -> SourceHistoryEntry | None:
        item = next(
            (item for item in self._items if item["id"] == self._active_id),
            None,
        )
        return self._entry(item) if item is not None else None

    def find(self, identifier: str) -> SourceHistoryEntry | None:
        item = next(
            (item for item in self._items if item["id"] == identifier),
            None,
        )
        return self._entry(item) if item is not None else None

    def activate(self, identifier: str) -> SourceHistoryEntry:
        """Make an existing cached picture current without rewriting its data."""
        item = next(
            (item for item in self._items if item["id"] == identifier),
            None,
        )
        if item is None:
            raise ValueError("source picture is not present in history")
        self._active_id = identifier
        self._save()
        return self._entry(item)

    def remember(self, data: bytes, filename: str) -> SourceHistoryEntry:
        """Cache an applied picture, make it current, and return its entry."""
        if not data:
            raise ValueError("source picture is empty")
        safe_name = (
            Path(filename).name.strip().replace("\r", " ").replace("\n", " ")
            or "source-picture"
        )[:240]
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            suffix = ".image"
        identifier = hashlib.sha256(data).hexdigest()
        cache_name = f"{identifier}{suffix}"
        used_at = time.time()

        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.directory, 0o700)
        cache_path = self.directory / cache_name
        if not cache_path.exists() or cache_path.stat().st_size != len(data):
            temporary = self.directory / f".{cache_name}.tmp"
            temporary.write_bytes(data)
            os.chmod(temporary, 0o600)
            os.replace(temporary, cache_path)

        superseded = [item for item in self._items if item["id"] == identifier]
        self._items = [item for item in self._items if item["id"] != identifier]
        self._items.insert(
            0,
            {
                "id": identifier,
                "filename": safe_name,
                "cache_name": cache_name,
                "used_at": used_at,
            },
        )
        evicted = self._items[self.limit :]
        self._items = self._items[: self.limit]
        self._active_id = identifier
        self._save()

        retained_names = {item["cache_name"] for item in self._items}
        for item in superseded + evicted:
            if item["cache_name"] in retained_names:
                continue
            try:
                (self.directory / item["cache_name"]).unlink()
            except FileNotFoundError:
                pass
        return self._entry(self._items[0])

    def _entry(self, item: dict[str, Any]) -> SourceHistoryEntry:
        return SourceHistoryEntry(
            identifier=str(item["id"]),
            filename=str(item["filename"]),
            cache_path=self.directory / str(item["cache_name"]),
            used_at=float(item["used_at"]),
        )

    def _load(self) -> None:
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return
        if not isinstance(payload, dict) or payload.get("version") != HISTORY_VERSION:
            return

        loaded: list[dict[str, Any]] = []
        raw_items = payload.get("items", [])
        if isinstance(raw_items, list):
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                identifier = raw.get("id")
                filename = raw.get("filename")
                cache_name = raw.get("cache_name")
                used_at = raw.get("used_at")
                if not all(
                    isinstance(value, str)
                    for value in (identifier, filename, cache_name)
                ):
                    continue
                if Path(cache_name).name != cache_name:
                    continue
                try:
                    numeric_used_at = float(used_at)
                except (TypeError, ValueError):
                    continue
                if not (self.directory / cache_name).is_file():
                    continue
                loaded.append(
                    {
                        "id": identifier,
                        "filename": filename,
                        "cache_name": cache_name,
                        "used_at": numeric_used_at,
                    }
                )
                if len(loaded) >= self.limit:
                    break

        active_id = payload.get("active_id")
        self._items = loaded
        self._active_id = (
            active_id
            if isinstance(active_id, str)
            and any(item["id"] == active_id for item in loaded)
            else None
        )

    def _save(self) -> None:
        payload = {
            "version": HISTORY_VERSION,
            "active_id": self._active_id,
            "items": self._items,
        }
        temporary = self.directory / ".history.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.metadata_path)
