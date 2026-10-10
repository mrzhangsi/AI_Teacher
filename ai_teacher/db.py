"""会话与 tick 记录：SQLite 落库。

图片默认不落盘（隐私 + 省磁盘），只存判定结果。
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    stats TEXT
);

CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts REAL NOT NULL,
    l0_verdict TEXT,
    model_used INTEGER NOT NULL DEFAULT 0,
    studying INTEGER,
    state TEXT NOT NULL,
    reason TEXT,
    interval_s REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ticks_session ON ticks(session_id, ts);
"""


def _locked(fn):
    """给 Store 的读方法加锁。

    写方法内部各自加锁了（execute + commit 必须在同一个临界区），
    读方法用这个装饰器统一处理，避免漏加。
    functools.wraps 保留原名与 docstring —— 测试会靠方法名做断言。
    """

    @functools.wraps(fn)
    def wrapper(self: "Store", *args: object, **kwargs: object):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapper


class Store:
    """薄封装的 SQLite 存储。

    **线程安全**：`check_same_thread=False` + 一把锁。原因是 GUI 里Store
    在后台线程创建、却可能从主线程读取统计；而任何「在A 线程建连接、
    到 B 线程用」的写法都会触发 `ProgrammingError: SQLite objects created
    in a thread can only be used in that same thread`。

    监控线程会高频写tick，主线程读报告，用锁保证不会交叉损坏。
    写入量很小（每秒最多一两条），锁的开销可以忽略。
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 允许跨线程使用；真正的并发保护靠 _lock
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """老库补列。

        CREATE TABLE IF NOT EXISTS 不会给已存在的表添加新列，
        所以升级字段必须显式 ALTER，否则老用户的库会一直报 no such column。
        """

        with self._lock:
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)")}
            if "stats" not in cols:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN stats TEXT")

    def start_session(self, goal: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (goal, started_at) VALUES (?, ?)",
                (goal, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id)
            )
            self._conn.commit()

    def save_session_stats(self, session_id: int, stats: dict[str, Any]) -> None:
        """把内存统计写回会话行，供事后用 report 命令查看。"""

        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET stats = ? WHERE id = ?",
                (json.dumps(stats, ensure_ascii=False), session_id),
            )
            self._conn.commit()

    def add_tick(
        self,
        session_id: int,
        *,
        l0_verdict: str,
        model_used: bool,
        studying: bool | None,
        state: str,
        reason: str,
        interval_s: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ticks
                    (session_id, ts, l0_verdict, model_used, studying, state, reason,
                     interval_s, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    time.time(),
                    l0_verdict,
                    int(model_used),
                    None if studying is None else int(studying),
                    state,
                    reason,
                    interval_s,
                    prompt_tokens,
                    completion_tokens,
                ),
            )
            self._conn.commit()

    @_locked
    def last_session(self) -> sqlite3.Row | None:
        cur = self._conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")
        return cur.fetchone()

    @_locked
    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        cur = self._conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    @_locked
    def session_goal(self, session_id: int) -> str | None:
        """取某次会话的目标；不存在返回 None。"""

        row = self._conn.execute(
            "SELECT goal FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row["goal"] if row else None

    @_locked
    def session_stats(self, session_id: int) -> dict[str, Any] | None:
        """读回某次会话落库的统计；老会话没存则返回 None。"""

        row = self._conn.execute(
            "SELECT stats FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row or not row["stats"]:
            return None
        try:
            return json.loads(row["stats"])
        except (json.JSONDecodeError, TypeError):
            return None

    @_locked
    def session_summary(self, session_id: int) -> dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS n,
                   SUM(studying) AS focus_n,
                   SUM(model_used) AS api_n,
                   SUM(CASE WHEN state = 'distracted' THEN 1 ELSE 0 END) AS distracted_n,
                   COALESCE(SUM(prompt_tokens), 0) AS pt,
                   COALESCE(SUM(completion_tokens), 0) AS ct
            FROM ticks WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        data = dict(row or {})
        n = data.get("n") or 0
        data["focus_ratio"] = (data.get("focus_n") or 0) / n if n else 0.0
        data["avg_prompt_tokens"] = (data.get("pt") or 0) / data["api_n"] if data.get("api_n") else 0
        return data

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

