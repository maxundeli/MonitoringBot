"""Функции работы с БД."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

DB_FILE = Path("db.json")
METRIC_DB = Path("metrics.sqlite")


def _init_metric_db() -> sqlite3.Connection:
    """Return connection to metrics DB, upgrading schema if needed."""
    con = sqlite3.connect(METRIC_DB, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute(
        """CREATE TABLE IF NOT EXISTS metrics(
               secret TEXT,
               ts     INTEGER,
               cpu    REAL,
               ram    REAL,
               gpu    REAL,
               vram   REAL,
               ram_used   REAL,
               ram_total  REAL,
               swap       REAL,
               swap_used  REAL,
               swap_total REAL,
               vram_used  REAL,
               vram_total REAL,
               cpu_temp   REAL,
               gpu_temp   REAL,
               net_up     REAL,
               net_down   REAL,
               uptime     INTEGER,
               disks      TEXT,
               top_procs  TEXT
        )"""
    )
    # дополняем недостающие поля при обновлении версии
    cols = [r[1] for r in con.execute("PRAGMA table_info(metrics)")]
    if "net_up" not in cols:
        con.execute("ALTER TABLE metrics ADD COLUMN net_up REAL")
    if "net_down" not in cols:
        con.execute("ALTER TABLE metrics ADD COLUMN net_down REAL")
    if "top_procs" not in cols:
        con.execute("ALTER TABLE metrics ADD COLUMN top_procs TEXT")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_secret_ts ON metrics(secret, ts)"
    )
    return con


sql = _init_metric_db()


def purge_old_metrics(days: int = 30):
    cutoff = int(time.time()) - days * 86400
    sql.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))


def record_metric(secret: str, data: Dict[str, Any]):
    sql.execute(
        """INSERT INTO metrics(
               secret, ts, cpu, ram, gpu, vram,
               ram_used, ram_total, swap, swap_used, swap_total,
               vram_used, vram_total, cpu_temp, gpu_temp,
               net_up, net_down, uptime, disks, top_procs
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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


def _avg_chunk(chunk: List[sqlite3.Row]) -> tuple[int, float | None, float | None, float | None, float | None, float | None, float | None]:
    ts = chunk[-1][0]
    cpu = _avg([r[1] for r in chunk])
    ram = _avg([r[2] for r in chunk])
    gpu = _avg([r[3] for r in chunk])
    vram = _avg([r[4] for r in chunk])
    up = _avg([r[5] for r in chunk])
    down = _avg([r[6] for r in chunk])
    return ts, cpu, ram, gpu, vram, up, down


def fetch_metrics(secret: str, since: int) -> List[tuple[int, float]]:
    rows = sql.execute(
        "SELECT ts, cpu, ram, gpu, vram, net_up, net_down FROM metrics WHERE secret=? AND ts>=? ORDER BY ts ASC",
        (secret, since),
    ).fetchall()

    if not rows:
        return []

    grouped = []
    chunk: list[sqlite3.Row] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk(chunk))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk(chunk))
    return grouped


def _avg_chunk_full(chunk: List[sqlite3.Row], avg_fn) -> Dict[str, Any]:
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
    rows = sql.execute(
        "SELECT ts, cpu, ram, gpu, vram, net_up, net_down, ram_used, ram_total, vram_used, vram_total FROM metrics WHERE secret=? AND ts>=? ORDER BY ts ASC",
        (secret, since),
    ).fetchall()
    if not rows:
        return []

    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    grouped = []
    chunk: List[sqlite3.Row] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk_full(chunk, avg))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk_full(chunk, avg))
    return grouped


def load_db() -> Dict[str, Any]:
    if DB_FILE.exists():
        data = json.loads(DB_FILE.read_text())
    else:
        data = {}
    data.setdefault("secrets", {})
    data.setdefault("active", {})
    data.setdefault("alerts", {})
    data.setdefault("alert_last", {})
    return data


def save_db(db: Dict[str, Any]):
    DB_FILE.write_text(json.dumps(db, indent=2))
