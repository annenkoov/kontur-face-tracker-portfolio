"""Explicitly labeled, local-only face crops for two independent workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
from uuid import uuid4

import cv2
import numpy as np

from .personal_states import FEATURE_NAMES, FEATURE_VERSION, PERSONAL_LABELS, FeatureVector, feature_vector
from .preprocessing import PADDED_CROP_ID, padded_face_crop


LABELS = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
MIN_SAVE_QUALITY = 0.45
MAX_SAMPLE_AGE_SECONDS = 2.0
_MANAGED_CROP_FILENAME = re.compile(r"[0-9a-f]{32}\.jpg\Z")


def observation_is_fresh(timestamp: float, now: float, max_age: float = MAX_SAMPLE_AGE_SECONDS, *,
                         monotonic_timestamp: float | None = None,
                         monotonic_now: float | None = None) -> bool:
    """Reject stale/future observations; use monotonic time for live age when available."""
    if monotonic_timestamp is not None and monotonic_now is not None:
        age = float(monotonic_now) - float(monotonic_timestamp)
    else:
        age = float(now) - float(timestamp)
    return math.isfinite(age) and 0 <= age <= max_age


@dataclass(frozen=True)
class EncodedFaceCrop:
    jpeg: bytes
    face_box: tuple[int, int, int, int]


def encode_face_crop(frame: np.ndarray, face_box: tuple[int, int, int, int]) -> EncodedFaceCrop:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Frame must be a non-empty BGR image")
    if len(face_box) != 4:
        raise ValueError("Face box must contain x1, y1, x2, y2")
    crop, local_box = padded_face_crop(frame, face_box)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Could not encode face crop")
    return EncodedFaceCrop(encoded.tobytes(), local_box)


@dataclass(frozen=True)
class LabeledSample:
    sample_id: str
    path: Path
    label: str
    created_at: str
    quality: float
    model_version: str
    preprocessing: str
    session_id: str
    face_box: tuple[int, int, int, int] | None = None
    label_kind: str = "expression"
    features: FeatureVector | None = None


def default_sample_directory() -> Path:
    """Return a per-user application-data path, outside the source tree."""
    base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    return Path(base) / "Kontur" / "labeled_faces"


def _read_features(version: str | None, raw: str | None) -> FeatureVector | None:
    if version != FEATURE_VERSION or not raw:
        return None
    try:
        values = json.loads(raw)
        return feature_vector(values) if isinstance(values, dict) else None
    except (TypeError, ValueError, RecursionError):
        return None


def _is_reparse_point(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _safe_file(path: Path) -> bool:
    try:
        info = path.lstat()
        return (stat.S_ISREG(info.st_mode) and not _is_reparse_point(info) and
                info.st_nlink <= 1 and path.resolve(strict=True) == path)
    except FileNotFoundError:
        return True


def _managed_crop_path(directory: Path, filename: str) -> Path | None:
    if not isinstance(filename, str) or not _MANAGED_CROP_FILENAME.fullmatch(filename):
        return None
    path = directory / filename
    return path if _safe_file(path) else None


class LabeledSampleStore:
    """Save only crops explicitly submitted by the user; never records video."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_sample_directory()
        self.root = self.root.expanduser().resolve()
        self.faces_dir = self.root / "faces"
        self.personal_faces_dir = self.root / "personal_faces"
        self.root.mkdir(parents=True, exist_ok=True)
        self._crop_directory("expression", create=True)
        manifest = self.root / "manifest.sqlite3"
        if not _safe_file(manifest):
            raise OSError("Refusing to open a redirected sample manifest")
        self.db = sqlite3.connect(manifest)
        self.db.row_factory = sqlite3.Row
        try:
            # Remove deleted row content from reused database pages.
            self.db.execute("PRAGMA secure_delete=ON")
            self.db.execute("PRAGMA foreign_keys=ON")
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                self.db.execute("""CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    quality REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    preprocessing TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    face_x1 INTEGER, face_y1 INTEGER, face_x2 INTEGER, face_y2 INTEGER
                )""")
                columns = {row["name"] for row in self.db.execute("PRAGMA table_info(samples)")}
                additions = {
                    "face_x1": "INTEGER", "face_y1": "INTEGER",
                    "face_x2": "INTEGER", "face_y2": "INTEGER",
                    "label_kind": "TEXT NOT NULL DEFAULT 'expression'",
                    "features_json": "TEXT", "features_version": "TEXT",
                }
                for name, definition in additions.items():
                    if name not in columns:
                        self.db.execute(f"ALTER TABLE samples ADD COLUMN {name} {definition}")
        except Exception:
            self.db.close()
            raise

    def save_crop(self, frame: np.ndarray, face_box: tuple[int, int, int, int], label: str,
                  quality: float, model_version: str, preprocessing: str,
                  session_id: str) -> LabeledSample:
        return self.save_expression_crop(encode_face_crop(frame, face_box), label, quality,
                                         model_version, preprocessing, session_id)

    def save_expression_crop(self, crop: EncodedFaceCrop, label: str, quality: float,
                             model_version: str, preprocessing: str, session_id: str) -> LabeledSample:
        if label not in LABELS:
            raise ValueError(f"Unknown expression label: {label}")
        return self._save(crop, label, quality, model_version, preprocessing, session_id, "expression")

    def save_personal_sample(self, frame: np.ndarray, face_box: tuple[int, int, int, int],
                             label: str, quality: float, session_id: str,
                             features: Mapping[str, float]) -> LabeledSample:
        if label not in PERSONAL_LABELS:
            raise ValueError(f"Unknown personal label: {label}")
        if not session_id or session_id == "unknown-session":
            raise ValueError("A capture session is required for a personal sample")
        return self._save(encode_face_crop(frame, face_box), label, quality, FEATURE_VERSION,
                          "mediapipe-mesh-bbox / BGR-JPEG92", session_id, "personal", feature_vector(features))

    def _crop_directory(self, kind: str, *, create: bool = False) -> Path:
        # A sibling junction can make a sweep operate on the other workflow.
        # Check the exact children, including reparse points on Windows.
        for directory in (self.faces_dir, self.personal_faces_dir):
            try:
                info = directory.lstat()
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
                info = directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info) or
                    directory.resolve(strict=True) != directory):
                raise OSError("Refusing to access a redirected face-crop directory")
        return self.personal_faces_dir if kind == "personal" else self.faces_dir

    def _save(self, crop: EncodedFaceCrop, label: str, quality: float, model_version: str,
              preprocessing: str, session_id: str, kind: str, features: FeatureVector | None = None) -> LabeledSample:
        quality = float(quality)
        if not MIN_SAVE_QUALITY <= quality <= 1:
            raise ValueError("Face quality is too low to save a useful sample")

        sample_id = uuid4().hex
        filename = f"{sample_id}.jpg"
        local_box = crop.face_box
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        preprocessing = f"{preprocessing};saved={PADDED_CROP_ID}"
        features_json = (json.dumps(dict(zip(FEATURE_NAMES, features)), allow_nan=False)
                         if features is not None else None)
        version = FEATURE_VERSION if features is not None else None
        created_file = False
        with self.db:
            # Serialize crop mutations with other store instances and orphan cleanup.
            self.db.execute("BEGIN IMMEDIATE")
            destination = _managed_crop_path(self._crop_directory(kind), filename)
            if destination is None:
                raise OSError("Refusing to overwrite an unsafe face-crop path")
            try:
                with destination.open("xb") as image_file:
                    created_file = True
                    image_file.write(crop.jpeg)
                self.db.execute(
                    """INSERT INTO samples
                    (sample_id, filename, label, created_at, quality, model_version, preprocessing,
                     session_id, face_x1, face_y1, face_x2, face_y2, label_kind, features_json, features_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sample_id, filename, label, created_at, quality, str(model_version),
                     preprocessing, str(session_id), *local_box, kind, features_json, version),
                )
                self.db.commit()
            except Exception:
                if created_file:
                    owned_path = _managed_crop_path(self._crop_directory(kind), filename)
                    if owned_path is not None:
                        owned_path.unlink(missing_ok=True)
                raise
        return LabeledSample(sample_id, destination, label, created_at, quality,
                             str(model_version), preprocessing, str(session_id), local_box, kind, features)

    def counts(self) -> dict[str, int]:
        return self._counts("expression", LABELS)

    def personal_counts(self) -> dict[str, int]:
        return self._counts("personal", PERSONAL_LABELS)

    def _counts(self, kind: str, labels: tuple[str, ...]) -> dict[str, int]:
        counts = dict.fromkeys(labels, 0)
        for row in self.db.execute(
            "SELECT label, COUNT(*) AS amount FROM samples WHERE label_kind=? GROUP BY label", (kind,)
        ):
            counts[row["label"]] = int(row["amount"])
        return counts

    def list_samples(self) -> list[LabeledSample]:
        return self._list("expression")

    def list_personal_samples(self) -> list[LabeledSample]:
        return self._list("personal")

    def _list(self, kind: str) -> list[LabeledSample]:
        directory = self._crop_directory(kind)
        rows = self.db.execute(
            "SELECT * FROM samples WHERE label_kind=? ORDER BY created_at, sample_id", (kind,)
        )
        samples = []
        for row in rows:
            path = _managed_crop_path(directory, row["filename"])
            if path is None:
                continue
            coordinates = (row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"])
            face_box = tuple(int(value) for value in coordinates) if all(value is not None for value in coordinates) else None
            features = (_read_features(row["features_version"], row["features_json"]) if kind == "personal" else None)
            samples.append(LabeledSample(row["sample_id"], path,
                                         row["label"], row["created_at"], float(row["quality"]),
                                         row["model_version"], row["preprocessing"], row["session_id"],
                                         face_box, kind, features))
        return samples

    def expression_preview_path(self, sample_id: str) -> Path | None:
        row = self.db.execute(
            "SELECT filename FROM samples WHERE sample_id=? AND label_kind='expression'", (sample_id,)
        ).fetchone()
        if row is None:
            return None
        path = _managed_crop_path(self._crop_directory("expression"), row["filename"])
        return path if path is not None and path.is_file() else None

    def delete_all(self) -> int:
        return self._delete_all("expression")

    def delete_all_personal(self) -> int:
        return self._delete_all("personal")

    def _delete_all(self, kind: str) -> int:
        rows = self.db.execute("SELECT sample_id FROM samples WHERE label_kind=?", (kind,)).fetchall()
        # Commit each deletion so an inaccessible crop does not leave earlier
        # deletions rolled back in the manifest after their files were removed.
        removed = sum(self._delete_sample(row["sample_id"], kind, discard_unsafe=True) for row in rows)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            # Sweep interrupted saves in this workflow only. Concurrent saves
            # since the initial selection have rows and must be preserved.
            filenames = {row["filename"] for row in self.db.execute(
                "SELECT filename FROM samples WHERE label_kind=?", (kind,)
            )}
            directory = self._crop_directory(kind)
            for entry in directory.glob("*.jpg"):
                if entry.name not in filenames:
                    path = _managed_crop_path(directory, entry.name)
                    if path is not None:
                        path.unlink(missing_ok=True)
        return removed

    def update_label(self, sample_id: str, label: str) -> bool:
        if label not in LABELS:
            raise ValueError(f"Unknown expression label: {label}")
        with self.db:
            cursor = self.db.execute(
                "UPDATE samples SET label=? WHERE sample_id=? AND label_kind='expression'", (label, sample_id)
            )
            return cursor.rowcount == 1

    def delete_sample(self, sample_id: str) -> bool:
        return self._delete_sample(sample_id, "expression")

    def delete_personal_sample(self, sample_id: str) -> bool:
        return self._delete_sample(sample_id, "personal")

    def _delete_sample(self, sample_id: str, kind: str, *, discard_unsafe: bool = False) -> bool:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT filename FROM samples WHERE sample_id=? AND label_kind=?", (sample_id, kind)
            ).fetchone()
            if row is None:
                return False
            path = _managed_crop_path(self._crop_directory(kind), row["filename"])
            if path is not None:
                path.unlink(missing_ok=True)
            elif not discard_unsafe:
                raise OSError("Refusing to delete an unsafe face-crop path")
            # Bulk removal can clear a corrupt row without accessing its path.
            self.db.execute("DELETE FROM samples WHERE sample_id=? AND label_kind=?", (sample_id, kind))
            return True

    def close(self) -> None:
        self.db.close()
