from __future__ import annotations
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

try:
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:
    psycopg2 = None

import yaml

logger = logging.getLogger(__name__)

_POOL: Any = None
_POOL_LOCK = threading.Lock()


class _SQLiteCursorWrapper:
    def __init__(self, cur: sqlite3.Cursor) -> None:
        self._cur = cur

    def __enter__(self) -> "_SQLiteCursorWrapper":
        return self

    def __exit__(self, *args: Any) -> None:
        self._cur.close()

    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        clean_sql = (
            sql.replace("%s", "?")
            .replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            .replace("NOW() - INTERVAL '1 hour'", "datetime('now', '-1 hour')")
            .replace("NOW()", "datetime('now')")
        )
        return self._cur.execute(clean_sql, params)

    def executemany(self, sql: str, seq_of_params: Any) -> sqlite3.Cursor:
        clean_sql = (
            sql.replace("%s", "?")
            .replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        )
        return self._cur.executemany(clean_sql, seq_of_params)

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()


class _SQLiteConnWrapper:
    def __init__(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)

    def cursor(self) -> _SQLiteCursorWrapper:
        return _SQLiteCursorWrapper(self._conn.cursor())

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _cfg() -> dict:
    raw = yaml.safe_load(Path("config.yaml").read_text())
    return raw.get("database", {})


def _get_sqlite_path() -> Path:
    try:
        raw = yaml.safe_load(Path("config.yaml").read_text())
        q = (raw.get("training", {}) or {}).get("queue_db")
        if q:
            return Path(q)
    except Exception:
        pass
    return Path("data/training.db")


def _password(cfg: dict) -> str:
    password = os.environ.get("DATABASE_PASSWORD") or cfg.get("password")
    if not password:
        raise RuntimeError(
            "Database password not configured. Set the DATABASE_PASSWORD "
            "environment variable (see .env.example) or add 'password' "
            "under 'database:' in config.yaml for local development."
        )
    return password


def get_pool() -> Any:
    global _POOL
    if psycopg2 is None:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            cfg = _cfg()
            _POOL = psycopg2.pool.ThreadedConnectionPool(
                minconn=int(cfg.get("pool_min_conn", 2)),
                maxconn=int(cfg.get("pool_max_conn", 10)),
                host=cfg.get("host", "localhost"),
                port=int(cfg.get("port", 5432)),
                dbname=cfg.get("name", "girivinity"),
                user=cfg.get("user", "postgres"),
                password=_password(cfg),
            )
            logger.info("PostgreSQL pool created → %s:%s/%s", cfg.get("host"), cfg.get("port"), cfg.get("name"))
    return _POOL


@contextmanager
def get_conn() -> Generator[Any, None, None]:
    if psycopg2 is None:
        conn = _SQLiteConnWrapper(_get_sqlite_path())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def execute(sql: str, params: tuple = ()) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def executemany(sql: str, params_list: list[tuple]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            if psycopg2 is not None and hasattr(psycopg2, "extras"):
                psycopg2.extras.execute_batch(cur, sql, params_list)
            else:
                cur.executemany(sql, params_list)


def fetchone(sql: str, params: tuple = ()) -> tuple | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def fetchall(sql: str, params: tuple = ()) -> list[tuple]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def close_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL and psycopg2 is not None:
            _POOL.closeall()
            _POOL = None
            logger.info("PostgreSQL pool closed")
