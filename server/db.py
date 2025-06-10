import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys
import subprocess
import shutil
import atexit

import pymysql

DB_FILE = Path("db.json")
METRIC_DB = Path("metrics.sqlite")

# Параметры подключения к MySQL. Сервер создаёт базу автоматически.
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASS = ""
MYSQL_DB = "monitoring"

# Локальный экземпляр MySQL хранит данные в подкаталоге проекта
MYSQL_DATA = Path("mysql_data")
MYSQL_PROC: subprocess.Popen | None = None


def _start_mysql() -> None:
    """Запустить локальный mysqld, если он ещё не запущен."""
    global MYSQL_PROC
    if MYSQL_PROC and MYSQL_PROC.poll() is None:
        return

    mysqld = shutil.which("mysqld")
    if not mysqld:
        print("\u274c mysqld не найден. Установите MySQL или MariaDB", file=sys.stderr)
        raise SystemExit(1)

    MYSQL_DATA.mkdir(exist_ok=True)
    if not (MYSQL_DATA / "mysql").exists():
        subprocess.run([mysqld, "--initialize-insecure", f"--datadir={MYSQL_DATA}"], check=True)

    MYSQL_PROC = subprocess.Popen(
        [
            mysqld,
            f"--datadir={MYSQL_DATA}",
            f"--port={MYSQL_PORT}",
            "--bind-address=127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(30):
        try:
            con = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASS,
                autocommit=True,
            )
            con.close()
            break
        except pymysql.err.OperationalError:
            time.sleep(1)
    else:
        print("\u274c \u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c MySQL", file=sys.stderr)
        raise SystemExit(1)


def _stop_mysql() -> None:
    if MYSQL_PROC and MYSQL_PROC.poll() is None:
        MYSQL_PROC.terminate()
        try:
            MYSQL_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            MYSQL_PROC.kill()


atexit.register(_stop_mysql)


def _ensure_database() -> None:
    """Запустить сервер и создать базу при первом запуске."""
    _start_mysql()
    try:
        con = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            autocommit=True,
            cursorclass=pymysql.cursors.Cursor,
        )
    except pymysql.err.OperationalError as exc:
        print(
            f"\u274c \u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u044c\u0441\u044f \u043a MySQL ({MYSQL_HOST}:{MYSQL_PORT}): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with con.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}`")
    con.close()


def _get_conn() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASS,
        database=MYSQL_DB,
        autocommit=True,
        cursorclass=pymysql.cursors.Cursor,
    )


def _maybe_migrate_sqlite(mysql_con: pymysql.connections.Connection) -> None:
    if not METRIC_DB.exists():
        return
    with mysql_con.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM metrics")
        if cur.fetchone()[0] > 0:
            return
    import sqlite3
    con = sqlite3.connect(METRIC_DB)
    rows = con.execute("SELECT * FROM metrics").fetchall()
    if not rows:
        return
    with mysql_con.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO metrics (
                secret, ts, cpu, ram, gpu, vram,
                ram_used, ram_total, swap, swap_used, swap_total,
                vram_used, vram_total, cpu_temp, gpu_temp,
                net_up, net_down, uptime, disks, top_procs
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )
    METRIC_DB.rename(METRIC_DB.with_suffix(".sqlite.bak"))


def _maybe_migrate_json(mysql_con: pymysql.connections.Connection) -> None:
    if not DB_FILE.exists():
        return
    with mysql_con.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM secrets")
        if cur.fetchone()[0] > 0:
            return
    data = json.loads(DB_FILE.read_text())
    secrets = data.get("secrets", {})
    active = data.get("active", {})
    alerts = data.get("alerts", {})
    alert_last = data.get("alert_last", {})
    with mysql_con.cursor() as cur:
        for secret, entry in secrets.items():
            cur.execute(
                "INSERT INTO secrets(secret, owners, nickname, pending) VALUES (%s,%s,%s,%s)",
                (
                    secret,
                    json.dumps(entry.get("owners", [])),
                    entry.get("nickname"),
                    json.dumps(entry.get("pending", [])),
                ),
            )
        for chat_id, s in active.items():
            cur.execute(
                "INSERT INTO active(chat_id, secret) VALUES (%s,%s)",
                (str(chat_id), s),
            )
        for uid, cfg in alerts.items():
            for s, metrics in cfg.items():
                for metric, thr in metrics.items():
                    cur.execute(
                        "INSERT INTO alerts(uid, secret, metric, threshold) VALUES (%s,%s,%s,%s)",
                        (uid, s, metric, thr),
                    )
        for key, ts in alert_last.items():
            cur.execute(
                "INSERT INTO alert_last(id, ts) VALUES (%s,%s)",
                (key, int(ts)),
            )
    DB_FILE.rename(DB_FILE.with_suffix(".json.bak"))


def _init_mysql() -> pymysql.connections.Connection:
    _ensure_database()
    try:
        con = _get_conn()
    except pymysql.err.OperationalError as exc:
        print(
            f"\u274c \u041d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0438\u0442\u044c\u0441\u044f \u043a MySQL ({MYSQL_HOST}:{MYSQL_PORT}): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with con.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                secret VARCHAR(64),
                ts INT,
                cpu FLOAT,
                ram FLOAT,
                gpu FLOAT,
                vram FLOAT,
                ram_used FLOAT,
                ram_total FLOAT,
                swap FLOAT,
                swap_used FLOAT,
                swap_total FLOAT,
                vram_used FLOAT,
                vram_total FLOAT,
                cpu_temp FLOAT,
                gpu_temp FLOAT,
                net_up FLOAT,
                net_down FLOAT,
                uptime INT,
                disks TEXT,
                top_procs TEXT,
                INDEX secret_ts(secret, ts)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS secrets (
                secret VARCHAR(64) PRIMARY KEY,
                owners TEXT,
                nickname TEXT,
                pending TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS active (
                chat_id VARCHAR(32) PRIMARY KEY,
                secret VARCHAR(64)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                uid VARCHAR(32),
                secret VARCHAR(64),
                metric VARCHAR(16),
                threshold FLOAT,
                PRIMARY KEY(uid, secret, metric)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_last (
                id VARCHAR(64) PRIMARY KEY,
                ts INT
            )
            """
        )
    _maybe_migrate_sqlite(con)
    _maybe_migrate_json(con)
    return con


sql = _init_mysql()


def purge_old_metrics(days: int = 30) -> None:
    cutoff = int(time.time()) - days * 86400
    with sql.cursor() as cur:
        cur.execute("DELETE FROM metrics WHERE ts < %s", (cutoff,))


def record_metric(secret: str, data: Dict[str, Any]) -> None:
    with sql.cursor() as cur:
        cur.execute(
            """
            INSERT INTO metrics(
                secret, ts, cpu, ram, gpu, vram,
                ram_used, ram_total, swap, swap_used, swap_total,
                vram_used, vram_total, cpu_temp, gpu_temp,
                net_up, net_down, uptime, disks, top_procs
            ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
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


def _avg_chunk(chunk: List[Tuple]) -> Tuple[int, float | None, float | None, float | None, float | None, float | None, float | None]:
    ts = chunk[-1][0]
    cpu = _avg([r[1] for r in chunk])
    ram = _avg([r[2] for r in chunk])
    gpu = _avg([r[3] for r in chunk])
    vram = _avg([r[4] for r in chunk])
    up = _avg([r[5] for r in chunk])
    down = _avg([r[6] for r in chunk])
    return ts, cpu, ram, gpu, vram, up, down


def fetch_metrics(secret: str, since: int) -> List[Tuple[int, float]]:
    with sql.cursor() as cur:
        cur.execute(
            "SELECT ts, cpu, ram, gpu, vram, net_up, net_down FROM metrics WHERE secret=%s AND ts>=%s ORDER BY ts ASC",
            (secret, since),
        )
        rows = cur.fetchall()
    if not rows:
        return []
    grouped = []
    chunk: List[Tuple] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk(chunk))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk(chunk))
    return grouped


def _avg_chunk_full(chunk: List[Tuple], avg_fn) -> Dict[str, Any]:
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
    with sql.cursor() as cur:
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

    grouped = []
    chunk: List[Tuple] = []
    for r in rows:
        chunk.append(r)
        if len(chunk) == 6:
            grouped.append(_avg_chunk_full(chunk, avg))
            chunk = []
    if chunk:
        grouped.append(_avg_chunk_full(chunk, avg))
    return grouped


def load_db() -> Dict[str, Any]:
    data = {
        "secrets": {},
        "active": {},
        "alerts": {},
        "alert_last": {},
    }
    with sql.cursor() as cur:
        cur.execute("SELECT secret, owners, nickname, pending FROM secrets")
        for secret, owners, nickname, pending in cur.fetchall():
            data["secrets"][secret] = {
                "owners": json.loads(owners or "[]"),
                "nickname": nickname,
                "pending": json.loads(pending or "[]"),
            }
        cur.execute("SELECT chat_id, secret FROM active")
        for chat_id, s in cur.fetchall():
            data["active"][str(chat_id)] = s
        cur.execute("SELECT uid, secret, metric, threshold FROM alerts")
        for uid, s, metric, thr in cur.fetchall():
            ucfg = data["alerts"].setdefault(uid, {})
            scfg = ucfg.setdefault(s, {})
            scfg[metric] = thr
        cur.execute("SELECT id, ts FROM alert_last")
        for key, ts in cur.fetchall():
            data["alert_last"][key] = ts
    return data


def save_db(db: Dict[str, Any]) -> None:
    with sql.cursor() as cur:
        cur.execute("DELETE FROM secrets")
        for secret, entry in db.get("secrets", {}).items():
            cur.execute(
                "INSERT INTO secrets(secret, owners, nickname, pending) VALUES(%s,%s,%s,%s)",
                (
                    secret,
                    json.dumps(entry.get("owners", [])),
                    entry.get("nickname"),
                    json.dumps(entry.get("pending", [])),
                ),
            )
        cur.execute("DELETE FROM active")
        for chat_id, s in db.get("active", {}).items():
            cur.execute(
                "INSERT INTO active(chat_id, secret) VALUES(%s,%s)",
                (str(chat_id), s),
            )
        cur.execute("DELETE FROM alerts")
        for uid, cfg in db.get("alerts", {}).items():
            for s, metrics in cfg.items():
                for metric, thr in metrics.items():
                    cur.execute(
                        "INSERT INTO alerts(uid, secret, metric, threshold) VALUES(%s,%s,%s,%s)",
                        (uid, s, metric, thr),
                    )
        cur.execute("DELETE FROM alert_last")
        for key, ts in db.get("alert_last", {}).items():
            cur.execute(
                "INSERT INTO alert_last(id, ts) VALUES(%s,%s)",
                (key, int(ts)),
            )

