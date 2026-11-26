"""Local SQLite journal for sessions and sustained facial-tension episodes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import os

from .events import StateTransition, TrackerState
from .scoring import ScoreSnapshot


def default_database_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    return base / "FaceStressTracker" / "tracker.sqlite3"


@dataclass(frozen=True)
class SessionSummary:
    session_id: int
    started_at: float
    ended_at: float | None
    interrupted: bool
    episode_count: int
    rating_count: int = 0
    usable_seconds: float | None = None
    usable_observations: int | None = None


class JournalStore:
    def __init__(self, database_path: str | Path | None = None) -> None:
        self.path = Path(database_path) if database_path else default_database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        # Clear deleted row content from reused database pages going forward.
        self._connection.execute("PRAGMA secure_delete = ON")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                interrupted INTEGER NOT NULL DEFAULT 0,
                usable_seconds REAL,
                usable_observations INTEGER
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                started_at REAL NOT NULL,
                ended_at REAL,
                duration REAL,
                peak_index REAL NOT NULL,
                average_quality REAL NOT NULL,
                quality_samples INTEGER NOT NULL DEFAULT 1,
                interrupted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS self_ratings (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at REAL NOT NULL,
                observed_at REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 0 AND 10),
                face_index REAL NOT NULL CHECK(face_index BETWEEN 0 AND 100),
                face_quality REAL NOT NULL CHECK(face_quality BETWEEN 0 AND 1)
            );
            """
        )
        episode_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(episodes)")}
        for name, definition in (
            ("entry_observations", "INTEGER NOT NULL DEFAULT 0"),
            ("entry_observed_span", "REAL NOT NULL DEFAULT 0"),
            ("entry_mean_interval", "REAL"),
            ("entry_max_interval", "REAL"),
            ("exit_observations", "INTEGER NOT NULL DEFAULT 0"),
            ("exit_observed_span", "REAL NOT NULL DEFAULT 0"),
            ("exit_mean_interval", "REAL"),
            ("exit_max_interval", "REAL"),
        ):
            if name not in episode_columns:
                self._connection.execute(f"ALTER TABLE episodes ADD COLUMN {name} {definition}")
        session_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(sessions)")}
        for name, definition in (("usable_seconds", "REAL"), ("usable_observations", "INTEGER")):
            if name not in session_columns:
                self._connection.execute(f"ALTER TABLE sessions ADD COLUMN {name} {definition}")
        rating_columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(self_ratings)")}
        if "rating_mode" not in rating_columns:
            self._connection.execute(
                "ALTER TABLE self_ratings ADD COLUMN rating_mode TEXT NOT NULL DEFAULT 'visible' "
                "CHECK(rating_mode IN ('visible', 'blind'))"
            )
        self._connection.commit()

    def start_session(self, started_at: float) -> int:
        cursor = self._connection.execute("INSERT INTO sessions(started_at) VALUES (?)", (started_at,))
        self._connection.commit()
        return int(cursor.lastrowid)

    def record_transition(self, session_id: int, transition: StateTransition) -> None:
        if transition.state == TrackerState.TENSION:
            self._connection.execute(
                """INSERT INTO episodes(session_id, started_at, peak_index, average_quality,
                   entry_observations, entry_observed_span, entry_mean_interval, entry_max_interval)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (session_id, transition.timestamp, transition.index, transition.quality,
                 transition.qualifying_observations, transition.qualifying_span_seconds,
                 transition.mean_inter_observation_seconds, transition.max_inter_observation_seconds),
            )
        else:
            self._connection.execute(
                """UPDATE episodes SET ended_at=?, duration=MAX(0, ? - started_at),
                   exit_observations=?, exit_observed_span=?, exit_mean_interval=?, exit_max_interval=?
                   WHERE id=(SELECT id FROM episodes WHERE session_id=? AND ended_at IS NULL
                             AND interrupted=0 ORDER BY started_at DESC LIMIT 1)""",
                (transition.timestamp, transition.timestamp, transition.qualifying_observations,
                 transition.qualifying_span_seconds, transition.mean_inter_observation_seconds,
                 transition.max_inter_observation_seconds, session_id),
            )
        self._connection.commit()

    def update_open_episode(self, session_id: int, snapshot: ScoreSnapshot, now: float) -> None:
        if not snapshot.usable or snapshot.index is None:
            return
        self._connection.execute(
            """UPDATE episodes SET peak_index=MAX(peak_index, ?),
               average_quality=(average_quality*quality_samples + ?) / (quality_samples+1),
               quality_samples=quality_samples+1
               WHERE id=(SELECT id FROM episodes WHERE session_id=? AND ended_at IS NULL
                         AND interrupted=0 ORDER BY started_at DESC LIMIT 1)
               AND started_at <= ?""",
            (snapshot.index, snapshot.quality, session_id, now),
        )
        self._connection.commit()

    def finish_session(self, session_id: int, ended_at: float, interrupted: bool = False,
                       usable_seconds: float | None = None,
                       usable_observations: int | None = None) -> None:
        open_episode = self._connection.execute(
            "SELECT id FROM episodes WHERE session_id=? AND ended_at IS NULL AND interrupted=0 LIMIT 1",
            (session_id,),
        ).fetchone()
        was_interrupted = interrupted or open_episode is not None
        if open_episode is not None:
            self._connection.execute("UPDATE episodes SET interrupted=1 WHERE id=?", (open_episode["id"],))
        self._connection.execute(
            "UPDATE sessions SET ended_at=?, interrupted=?, usable_seconds=?, usable_observations=? WHERE id=?",
            (ended_at, int(was_interrupted), usable_seconds, usable_observations, session_id),
        )
        self._connection.commit()

    def interrupt_open_episode(self, session_id: int, ended_at: float) -> None:
        """Close the last trusted episode boundary when observations break."""
        self._connection.execute(
            """UPDATE episodes SET ended_at=?, duration=MAX(0, ? - started_at), interrupted=1
               WHERE id=(SELECT id FROM episodes WHERE session_id=? AND ended_at IS NULL
                         AND interrupted=0 ORDER BY started_at DESC LIMIT 1)""",
            (ended_at, ended_at, session_id),
        )
        self._connection.commit()

    def record_self_rating(self, session_id: int, created_at: float, observed_at: float, rating: int,
                           face_index: float, face_quality: float, rating_mode: str = "visible") -> int:
        """Store an explicit user rating beside the simultaneous face-only signal."""
        timestamp = float(created_at)
        observed_timestamp = float(observed_at)
        score = float(face_index)
        quality = float(face_quality)
        mode = str(rating_mode)
        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError("Оценка должна быть целым числом от 0 до 10")
        value = rating
        if (not math.isfinite(timestamp) or not math.isfinite(observed_timestamp) or
                not math.isfinite(score) or not math.isfinite(quality)
                or not 0 <= value <= 10 or score < 0 or score > 100 or quality < 0 or quality > 1 or
                mode not in {"visible", "blind"}):
            raise ValueError("Некорректная отметка состояния")
        duplicate = self._connection.execute(
            "SELECT 1 FROM self_ratings WHERE session_id=? AND observed_at=? LIMIT 1",
            (int(session_id), observed_timestamp),
        ).fetchone()
        if duplicate:
            raise ValueError("Для этого кадра личная отметка уже сохранена")
        cursor = self._connection.execute(
            "INSERT INTO self_ratings(session_id, created_at, observed_at, rating, face_index, face_quality, "
            "rating_mode) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (int(session_id), timestamp, observed_timestamp, value, score, quality, mode),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def list_self_ratings(self, session_id: int) -> list[dict]:
        rows = self._connection.execute(
            "SELECT * FROM self_ratings WHERE session_id=? ORDER BY created_at, id", (int(session_id),)
        ).fetchall()
        return [dict(row) for row in rows]

    def list_all_self_ratings(self) -> list[dict]:
        rows = self._connection.execute(
            """SELECT r.*, s.started_at AS session_started_at
               FROM self_ratings r JOIN sessions s ON s.id=r.session_id
               ORDER BY r.created_at, r.id"""
        ).fetchall()
        return [dict(row) for row in rows]

    def list_sessions(self, limit: int = 50) -> list[SessionSummary]:
        rows = self._connection.execute(
            """SELECT s.*, COUNT(DISTINCT e.id) AS episode_count,
                      (SELECT COUNT(*) FROM self_ratings r WHERE r.session_id=s.id) AS rating_count
               FROM sessions s
               LEFT JOIN episodes e ON e.session_id=s.id
               GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?""",
            (max(1, int(limit)),),
        ).fetchall()
        return [SessionSummary(int(row["id"]), row["started_at"], row["ended_at"],
                               bool(row["interrupted"]), int(row["episode_count"]),
                               int(row["rating_count"]),
                               float(row["usable_seconds"]) if row["usable_seconds"] is not None else None,
                               int(row["usable_observations"])
                               if row["usable_observations"] is not None else None) for row in rows]

    def list_episodes(self, session_id: int) -> list[dict]:
        rows = self._connection.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY started_at", (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_all(self) -> None:
        self._connection.execute("DELETE FROM episodes")
        self._connection.execute("DELETE FROM sessions")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
