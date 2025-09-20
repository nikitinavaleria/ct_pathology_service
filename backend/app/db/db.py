import os
import threading
from typing import Any, Optional, Sequence, Callable, Dict
import psycopg
from psycopg import OperationalError, InterfaceError
from psycopg.rows import dict_row

class DB_Connector:

    def __init__(self, conn_info: Dict[str, Any]):
        self.conn_info = dict(conn_info)
        self._conn: Optional[psycopg.Connection] = None
        self._lock = threading.RLock()
        self.connect()

    def connect(self) -> None:
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                return
            self._conn = psycopg.connect(**self.conn_info)
            # быстрая проверка
            with self._conn.cursor() as cur:
                cur.execute("SELECT 1")

    def close(self) -> None:
        with self._lock:
            if self._conn and not self._conn.closed:
                self._conn.close()
            self._conn = None

    def _ensure_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self.connect()
        return self._conn  # type: ignore[return-value]

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except (OperationalError, InterfaceError):
            self.close()
            self.connect()
            return fn()

    def fetch_one(self, sql: str, params: Optional[Sequence[Any]] = None) -> Optional[dict]:
        def _do():
            with self._lock:
                conn = self._ensure_conn()
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params or ())
                    return cur.fetchone()
        return self._with_retry(_do)

    def fetch_all(self, sql: str, params: Optional[Sequence[Any]] = None) -> list[dict]:
        def _do():
            with self._lock:
                conn = self._ensure_conn()
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params or ())
                    return list(cur.fetchall())
        return self._with_retry(_do)

    def scalar(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        row = self.fetch_one(sql, params)
        return None if row is None else next(iter(row.values()))

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> int:
        """DDL/DML без RETURNING. Возвращает количество затронутых строк."""
        def _do():
            with self._lock:
                conn = self._ensure_conn()
                with conn.cursor() as cur:
                    cur.execute(sql, params or ())
                    affected = cur.rowcount
                conn.commit()
                return affected
        return self._with_retry(_do)

    def execute_returning(self, sql: str, params: Optional[Sequence[Any]] = None) -> Optional[dict]:
        """DML с RETURNING. Возвращает первую строку как dict (или None)."""
        def _do():
            with self._lock:
                conn = self._ensure_conn()
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params or ())
                    row = cur.fetchone()
                conn.commit()
                return row
        return self._with_retry(_do)