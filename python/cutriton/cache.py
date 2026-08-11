from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    key: str
    status: str
    payload: dict[str, Any]
    updated_at: float


class JitDecisionCache:
    """Process-safe persistent decisions; Triton owns compiled binary caching."""

    def __init__(self, directory: Path | str, max_bytes: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "decisions.sqlite3"
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decisions (
                      key TEXT PRIMARY KEY,
                      status TEXT NOT NULL,
                      payload TEXT NOT NULL,
                      updated_at REAL NOT NULL,
                      last_access REAL NOT NULL
                    )
                    """
                )
        except sqlite3.DatabaseError:
            corrupt = self.path.with_suffix(f".corrupt-{int(time.time())}")
            if self.path.exists():
                self.path.replace(corrupt)
            for suffix in ("-wal", "-shm"):
                companion = self.path.with_name(self.path.name + suffix)
                if companion.exists():
                    companion.replace(corrupt.with_name(corrupt.name + suffix))
            with self._connection() as connection:
                connection.execute(
                    "CREATE TABLE decisions (key TEXT PRIMARY KEY, status TEXT NOT NULL, "
                    "payload TEXT NOT NULL, updated_at REAL NOT NULL, last_access REAL NOT NULL)"
                )

    def get(self, key: str) -> CacheEntry | None:
        now = time.time()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status, payload, updated_at FROM decisions WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE decisions SET last_access = ? WHERE key = ?", (now, key))
        try:
            payload = json.loads(row[1])
        except (TypeError, json.JSONDecodeError):
            self.delete(key)
            return None
        return CacheEntry(key, row[0], payload, float(row[2]))

    def put(self, key: str, status: str, payload: dict[str, Any]) -> None:
        if status not in {"enabled", "disabled", "failed"}:
            raise ValueError("unsupported cache status")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        now = time.time()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO decisions(key, status, payload, updated_at, last_access)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET status=excluded.status,
                  payload=excluded.payload, updated_at=excluded.updated_at,
                  last_access=excluded.last_access
                """,
                (key, status, serialized, now, now),
            )
        self.prune()

    def delete(self, key: str) -> None:
        with self._lock, self._connection() as connection:
            connection.execute("DELETE FROM decisions WHERE key = ?", (key,))

    def prune(self) -> None:
        files = [item for item in self.directory.rglob("*") if item.is_file()]
        total = sum(item.stat().st_size for item in files)
        if total <= self.max_bytes:
            return
        with self._lock, self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            remove = max(1, int(count * 0.1))
            connection.execute(
                "DELETE FROM decisions WHERE key IN "
                "(SELECT key FROM decisions ORDER BY last_access ASC LIMIT ?)",
                (remove,),
            )
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        protected = {
            self.path.resolve(),
            self.path.with_name(self.path.name + "-wal").resolve(),
            self.path.with_name(self.path.name + "-shm").resolve(),
        }
        candidates = sorted(
            (item for item in files if item.resolve() not in protected),
            key=lambda item: item.stat().st_atime,
        )
        target = int(self.max_bytes * 0.9)
        for candidate in candidates:
            if total <= target:
                break
            try:
                size = candidate.stat().st_size
                candidate.unlink()
                total -= size
            except FileNotFoundError:
                continue

    def count(self) -> int:
        with self._lock, self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0])
