"""Database helpers using MySQL backend with automatic migration."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pymysql

DB_FILE = Path("db.json")
METRIC_DB = Path("metrics.sqlite")


class MySQL:
    def __init__(self) -> None:
        host = os.getenv("MYSQL_HOST", "127.0.0.1")
        port = int(os.getenv("MYSQL_PORT", "3306"))
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        db_name = os.getenv("MYSQL_DB", "monitoring")

        # first connect without DB to create it
        conn = pymysql.connect(host=host, port=port, user=user, password=password, autocommit=True)
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4")
        conn.close()

        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=db_name,
            charset="utf8mb4",
            autocommit=True,
        )
        self.lock = threading.Lock()
        self._init_schema()
        self._maybe_migrate()

    def execute(self, query: str, params: tuple | None = None):
        """Execute a query and return a lightweight result object."""
        query = query.replace("?", "%s")
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(query, params or ())
            rows = cur.fetchall() if cur.description else []
            cur.close()
        class _Res:
            def __init__(self, r):
                self._rows = r
            def fetchone(self):
                return self._rows[0] if self._rows else None
            def fetchall(self):
                return self._rows
        return _Res(rows)

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS metrics(
                       secret      VARCHAR(255),
                       ts          BIGINT,
                       cpu         DOUBLE,
                       ram         DOUBLE,
                       gpu         DOUBLE,
                       vram        DOUBLE,
                       ram_used    DOUBLE,
                       ram_total   DOUBLE,
                       swap        DOUBLE,
                       swap_used   DOUBLE,
                       swap_total  DOUBLE,
                       vram_used   DOUBLE,
                       vram_total  DOUBLE,
                       cpu_temp    DOUBLE,
                       gpu_temp    DOUBLE,
                       net_up      DOUBLE,
                       net_down    DOUBLE,
                       uptime      BIGINT,
                       disks       TEXT,
                       top_procs   TEXT,
                       KEY idx_secret_ts(secret, ts)
                   )"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS state(
                       id    INT PRIMARY KEY,
                       data  LONGTEXT
                   )"""
            )
            try:
                cur.execute("ALTER TABLE state MODIFY data LONGTEXT")
            except Exception:
                pass
            cur.execute("INSERT IGNORE INTO state(id, data) VALUES(1, '{}')")

    def _maybe_migrate(self) -> None:
        migrated = False
        if DB_FILE.exists():
            try:
                data = json.loads(DB_FILE.read_text())
                with self.conn.cursor() as cur:
                    cur.execute("UPDATE state SET data=%s WHERE id=1", (json.dumps(data),))
                DB_FILE.rename(DB_FILE.with_suffix(".bak"))
                migrated = True
            except Exception:
                pass
        if METRIC_DB.exists():
            try:
                sq = sqlite3.connect(METRIC_DB)
                rows = sq.execute("SELECT * FROM metrics").fetchall()
                cols = [d[1] for d in sq.execute("PRAGMA table_info(metrics)").fetchall()]
                placeholders = ",".join(["%s"] * len(cols))
                with self.conn.cursor() as cur:
                    cur.executemany(
                        f"INSERT INTO metrics({','.join(cols)}) VALUES({placeholders})",
                        [tuple(r) for r in rows],
                    )
                sq.close()
                METRIC_DB.rename(METRIC_DB.with_suffix(".bak"))
                migrated = True
            except Exception:
                pass
        if migrated:
            print("Migrated existing data to MySQL")


sql = MySQL()


def purge_old_metrics(days: int = 30) -> None:
    cutoff = int(time.time()) - days * 86400
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute("DELETE FROM metrics WHERE ts < %s", (cutoff,))


def record_metric(secret: str, data: Dict[str, Any]) -> None:
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute(
            """INSERT INTO metrics(
                   secret, ts, cpu, ram, gpu, vram,
                   ram_used, ram_total, swap, swap_used, swap_total,
                   vram_used, vram_total, cpu_temp, gpu_temp,
                   net_up, net_down, uptime, disks, top_procs
               ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (
                secret,
                int(time.time()),
                data.get("cpu"),
                data.get("ram"),
                data.get("gpu"),
                data.get("vram"),
                data.get("ram_used"),
                data.get("ram_total"),
                data.get("swap"),
                data.get("swap_used"),
                data.get("swap_total"),
                data.get("vram_used"),
                data.get("vram_total"),
                data.get("cpu_temp"),
                data.get("gpu_temp"),
                data.get("net_up"),
                data.get("net_down"),
                data.get("uptime"),
                None,
                None,
            ),
        )


def _avg(vals: List[float | None]) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _avg_chunk(chunk: List[tuple]) -> tuple[int, float | None, float | None, float | None, float | None, float | None, float | None]:
    ts = chunk[-1][0]
    cpu = _avg([r[1] for r in chunk])
    ram = _avg([r[2] for r in chunk])
    gpu = _avg([r[3] for r in chunk])
    vram = _avg([r[4] for r in chunk])
    up = _avg([r[5] for r in chunk])
    down = _avg([r[6] for r in chunk])
    return ts, cpu, ram, gpu, vram, up, down


def fetch_metrics(secret: str, since: int) -> List[tuple[int, float]]:
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute(
            "SELECT ts, cpu, ram, gpu, vram, net_up, net_down FROM metrics WHERE secret=%s AND ts>=%s ORDER BY ts ASC",
            (secret, since),
        )
        rows = cur.fetchall()
    if not rows:
        return []

    grouped: List[tuple] = []
    chunk: List[tuple] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk(chunk))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk(chunk))
    return grouped


def _avg_chunk_full(chunk: List[tuple], avg_fn) -> Dict[str, Any]:
    r_last = chunk[-1]
    return {
        "ts": r_last[0],
        "cpu": avg_fn([r[1] for r in chunk]),
        "ram": avg_fn([r[2] for r in chunk]),
        "gpu": avg_fn([r[3] for r in chunk]),
        "vram": avg_fn([r[4] for r in chunk]),
        "net_up": avg_fn([r[5] for r in chunk]),
        "net_down": avg_fn([r[6] for r in chunk]),
        "ram_used": avg_fn([r[7] for r in chunk]),
        "ram_total": r_last[8],
        "vram_used": avg_fn([r[9] for r in chunk]),
        "vram_total": r_last[10],
    }


def fetch_metrics_full(secret: str, since: int) -> List[Dict[str, Any]]:
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute(
            "SELECT ts, cpu, ram, gpu, vram, net_up, net_down, ram_used, ram_total, vram_used, vram_total FROM metrics WHERE secret=%s AND ts>=%s ORDER BY ts ASC",
            (secret, since),
        )
        rows = cur.fetchall()
    if not rows:
        return []

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    grouped: List[Dict[str, Any]] = []
    chunk: List[tuple] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk_full(chunk, avg))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk_full(chunk, avg))
    return grouped


def load_db() -> Dict[str, Any]:
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute("SELECT data FROM state WHERE id=1")
        row = cur.fetchone()
    data = json.loads(row[0] if row and row[0] else "{}")
    data.setdefault("secrets", {})
    data.setdefault("active", {})
    data.setdefault("alerts", {})
    data.setdefault("alert_last", {})
    return data


def save_db(db: Dict[str, Any]) -> None:
    with sql.lock, sql.conn.cursor() as cur:
        cur.execute("UPDATE state SET data=%s WHERE id=1", (json.dumps(db),))

