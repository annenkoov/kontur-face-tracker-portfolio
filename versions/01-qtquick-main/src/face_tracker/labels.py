"""Explicitly labeled, local-only face crops used for personal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

import cv2
import numpy as np

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


def default_sample_directory() -> Path:
    """Return a per-user application-data path, outside the source tree."""
    base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
    return Path(base) / "Kontur" / "labeled_faces"


class LabeledSampleStore:
    """Save only crops explicitly submitted by the user; never records video."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_sample_directory()
        self.root = self.root.expanduser()
        self.faces_dir = self.root / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "manifest.sqlite3")
        self.db.row_factory = sqlite3.Row
        # The manifest contains personal labels and face-crop filenames; clear deleted
        # row content from reused database pages for future deletions.
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.execute("PRAGMA foreign_keys=ON")
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
        for name in ("face_x1", "face_y1", "face_x2", "face_y2"):
            if name not in columns:
                self.db.execute(f"ALTER TABLE samples ADD COLUMN {name} INTEGER")
        self.db.commit()

    def save_crop(self, frame: np.ndarray, face_box: tuple[int, int, int, int], label: str,
                  quality: float, model_version: str, preprocessing: str,
                  session_id: str) -> LabeledSample:
        if label not in LABELS:
            raise ValueError(f"Unknown expression label: {label}")
        if not math.isfinite(float(quality)) or quality < MIN_SAVE_QUALITY or quality > 1:
            raise ValueError("Face quality is too low to save a useful sample")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            raise ValueError("Frame must be a non-empty BGR image")
        if len(face_box) != 4:
            raise ValueError("Face box must contain x1, y1, x2, y2")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in face_box)
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            raise ValueError("Face box is outside the frame or empty")
        crop, local_box = padded_face_crop(frame, (x1, y1, x2, y2))
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("Could not encode face crop")

        sample_id = uuid4().hex
        filename = f"{sample_id}.jpg"
        destination = self.faces_dir / filename
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            # Exclusive creation prevents accidental overwrite, even in a UUID collision.
            with destination.open("xb") as image_file:
                image_file.write(encoded.tobytes())
            self.db.execute(
                """INSERT INTO samples
                (sample_id, filename, label, created_at, quality, model_version, preprocessing,
                 session_id, face_x1, face_y1, face_x2, face_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (sample_id, filename, label, created_at, float(quality), str(model_version),
                 f"{preprocessing};saved={PADDED_CROP_ID}", str(session_id), *local_box),
            )
            self.db.commit()
        except Exception:
            destination.unlink(missing_ok=True)
            self.db.rollback()
            raise
        return LabeledSample(sample_id, destination, label, created_at, float(quality),
                             str(model_version), f"{preprocessing};saved={PADDED_CROP_ID}", str(session_id), local_box)

    def counts(self) -> dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for row in self.db.execute("SELECT label, COUNT(*) AS amount FROM samples GROUP BY label"):
            counts[row["label"]] = int(row["amount"])
        return counts

    def list_samples(self) -> list[LabeledSample]:
        rows = self.db.execute("SELECT * FROM samples ORDER BY created_at, sample_id").fetchall()
        samples = []
        for row in rows:
            coordinates = (row["face_x1"], row["face_y1"], row["face_x2"], row["face_y2"])
            face_box = tuple(int(value) for value in coordinates) if all(value is not None for value in coordinates) else None
            samples.append(LabeledSample(row["sample_id"], self.faces_dir / row["filename"], row["label"],
                                         row["created_at"], float(row["quality"]), row["model_version"],
                                         row["preprocessing"], row["session_id"], face_box))
        return samples

    def delete_all(self) -> int:
        rows = self.db.execute("SELECT filename FROM samples").fetchall()
        # Remove manifest-owned files and UUID-named orphan crops from interrupted saves.
        # Limit the sweep to direct children of this app-owned directory.
        for row in rows:
            filename = row["filename"]
            if Path(filename).name == filename and _MANAGED_CROP_FILENAME.fullmatch(filename):
                (self.faces_dir / filename).unlink(missing_ok=True)
        for path in self.faces_dir.glob("*.jpg"):
            if _MANAGED_CROP_FILENAME.fullmatch(path.name):
                path.unlink(missing_ok=True)
        self.db.execute("DELETE FROM samples")
        self.db.commit()
        return len(rows)

    def update_label(self, sample_id: str, label: str) -> bool:
        if label not in LABELS:
            raise ValueError(f"Unknown expression label: {label}")
        cursor = self.db.execute("UPDATE samples SET label=? WHERE sample_id=?", (label, sample_id))
        self.db.commit()
        return cursor.rowcount == 1

    def delete_sample(self, sample_id: str) -> bool:
        row = self.db.execute("SELECT filename FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
        if row is None:
            return False
        filename = row["filename"]
        if Path(filename).name != filename or not _MANAGED_CROP_FILENAME.fullmatch(filename):
            raise RuntimeError("Refusing to delete a path outside the managed face-crop directory")
        (self.faces_dir / filename).unlink(missing_ok=True)
        self.db.execute("DELETE FROM samples WHERE sample_id=?", (sample_id,))
        self.db.commit()
        return True

    def close(self) -> None:
        self.db.close()
