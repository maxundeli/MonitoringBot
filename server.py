from __future__ import annotations

"""remote_bot_server"""
import asyncio
import io
import json
import logging
import os
import re
import secrets
from html import escape
from telegram.constants import ParseMode
import time
import sqlite3
import string
from telegram import InputFile
import subprocess
import sys
import threading
from pathlib import Path
import numpy as np
from statistics import median
from datetime import datetime, timedelta
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# ────────────────────────── CONFIG ─────────────────────────────────────────
ENV_FILE = Path(".env")
DB_FILE = Path("db.json")
METRIC_DB = Path("metrics.sqlite")
API_PORT = int(os.getenv("PORT", "8000"))

# последний текстовый статус от клиентов (speedtest и пр.)
LATEST_TEXT: Dict[str, str] = {}

# ссылка на экземпляр Telegram-приложения для отправки уведомлений
TG_APP = None

CERT_FILE = Path(os.getenv("SSL_CERT", "cert.pem"))
KEY_FILE = Path(os.getenv("SSL_KEY", "key.pem"))

# matplotlib без X-сервера
matplotlib.use("Agg")

# ────────────────────────── helpers ────────────────────────────────────────

def _load_dotenv() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

def _ensure_ssl() -> None:
    if CERT_FILE.exists() and KEY_FILE.exists():
        return
    logging.info("🔒 Generating self-signed TLS certificate…")
    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(KEY_FILE),
                "-out",
                str(CERT_FILE),
                "-days",
                "825",
                "-nodes",
                "-subj",
                "/CN=localhost",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info("✅ Certificate created.")
    except Exception as exc:
        logging.warning("⚠️  TLS cert generation failed: %s", exc)

_load_dotenv()
_ensure_ssl()

TOKEN = os.getenv("BOT_TOKEN") or input("Enter Telegram BOT_TOKEN: ").strip()
if not TOKEN:
    print("❌ BOT_TOKEN required.")
    sys.exit(1)
if "BOT_TOKEN" not in os.environ:
    ENV_FILE.write_text(
        (ENV_FILE.read_text() if ENV_FILE.exists() else "") + f"BOT_TOKEN={TOKEN}\n"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("remote-bot")

UNIT_NAMES = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]

def human_bytes(num: float) -> str:
    for unit in UNIT_NAMES:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} EiB"

def disk_bar(p: float, length: int = 10) -> str:
    filled = int(round(p * length / 100))
    return "█" * filled + "░" * (length - filled)
async def check_speedtest_done(ctx: ContextTypes.DEFAULT_TYPE):
    job  = ctx.job
    data = job.data

    secret  = data["secret"]
    chat_id = data["chat_id"]
    msg_id  = data["msg_id"]

    entry = load_db()["secrets"].get(secret, {})

    if "speedtest" in entry.get("pending", []):
        return

    status: str = LATEST_TEXT.get(secret, "")
    if "Speedtest" not in status:

        start_ts = data.setdefault("start_ts", time.time())
        TIMEOUT  = 3 * 60
        if time.time() - start_ts > TIMEOUT:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="⚠️  Speedtest занял много времени и был прерван.",
            )
            job.schedule_removal()
        return

    # ─── 3) Результат получен – выкладываем и выходим ───────────────────────────
    await ctx.bot.edit_message_text(
        chat_id=chat_id,
        message_id=msg_id,
        text=status,
        parse_mode="Markdown",
    )
    job.schedule_removal()

# ───────────────────── SQLite helpers ──────────────────────────────────────
def _init_metric_db() -> sqlite3.Connection:
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
               disks      TEXT
        )"""
    )
    cols = [r[1] for r in con.execute("PRAGMA table_info(metrics)")]
    if "net_up" not in cols:
        con.execute("ALTER TABLE metrics ADD COLUMN net_up REAL")
    if "net_down" not in cols:
        con.execute("ALTER TABLE metrics ADD COLUMN net_down REAL")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_secret_ts ON metrics(secret, ts)"
    )
    return con

sql = _init_metric_db()

def purge_old_metrics(days: int = 30):
    """Удалить метрики старше указанного количества дней."""
    cutoff = int(time.time()) - days * 86400
    cur = sql.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))
    log.info("🧹 Purged %d old metric rows", cur.rowcount)

def record_metric(secret: str, data: Dict[str, Any]):
    sql.execute(
        """INSERT INTO metrics(
               secret, ts, cpu, ram, gpu, vram,
               ram_used, ram_total, swap, swap_used, swap_total,
               vram_used, vram_total, cpu_temp, gpu_temp,
               net_up, net_down, uptime, disks
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
            json.dumps(data.get("disks")),
        ),
    )

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

def _avg(val_list: List[float | None]) -> float | None:
    vals = [v for v in val_list if v is not None]
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


async def _purge_loop():
    while True:
        purge_old_metrics()
        await asyncio.sleep(86400)


async def maybe_send_alerts(secret: str, data: Dict[str, Any]):
    """Check alert thresholds and notify owners if exceeded."""
    db = load_db()
    alerts = db.get("alerts", {})
    changed = False
    for uid, secrets_cfg in alerts.items():
        cfg = secrets_cfg.get(secret)
        if not cfg:
            continue
        for metric, thr in cfg.items():
            val = data.get(metric)
            if val is None:
                continue
            if val >= thr:
                key = f"{uid}:{secret}:{metric}"
                last = db.get("alert_last", {}).get(key, 0)
                if time.time() - last >= 300:
                    name = db.get("secrets", {}).get(secret, {}).get("nickname", secret)
                    msg = f"⚠️ {name}: {metric.upper()} {val:.1f}% ≥ {thr}%"
                    if TG_APP:
                        TG_APP.create_task(TG_APP.bot.send_message(chat_id=int(uid), text=msg))
                    db.setdefault("alert_last", {})[key] = time.time()
                    changed = True
    if changed:
        save_db(db)

# ──────────────────────── JSON DB helpers ──────────────────────────────────
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

# ───────────────────────- Telegram command handlers ────────────────────────
OWNER_HELP = (
    "Команды:\n"
    "/newkey <имя> – создать ключ.\n"
    "/linkkey <ключ> – подписаться.\n"
    "/set <ключ> – сделать активным.\n"
    "/list – показать ключи.\n"
    "/status – статус + кнопки.\n"
    "/renamekey <ключ> <имя> – переименовать.\n"
    "/delkey <ключ/имя> – удалить.\n"
    "/setalert <ключ/имя> <метрика> <порог> – настроить алерт.\n"
    "/delalert <ключ/имя> <метрика> – удалить алерт."
)
def gen_secret(n: int = 20):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

def is_owner(entry: Dict[str, Any], user_id: int) -> bool:
    return user_id in entry.get("owners", [])

def format_status(row: sqlite3.Row) -> str:
    lines = [
        "💻 *PC stats*",
        f"⏳ Uptime: {timedelta(seconds=int(row['uptime'] or 0))}",
        "*━━━━━━━━━━━CPU━━━━━━━━━━━*",
        f"🖥️ CPU: {row['cpu']:.1f}%",
        f"🌡️ CPU Temp: {row['cpu_temp']:.1f} °C" if row['cpu_temp'] is not None else "🌡️ CPU Temp: N/A",
        "*━━━━━━━━━━━RAM━━━━━━━━━━━*",
        f"🧠 RAM: {human_bytes(row['ram_used'])} / {human_bytes(row['ram_total'])} ({row['ram']:.1f}%)",
        f"🧠 SWAP: {human_bytes(row['swap_used'])} / {human_bytes(row['swap_total'])} ({row['swap']:.1f}%)",
    ]
    if row['net_up'] is not None and row['net_down'] is not None:
        lines.extend([
            "*━━━━━━━━━━━NET━━━━━━━━━━━*",
            f"📡 Net: ↑ {human_bytes(row['net_up'])}/s ↓ {human_bytes(row['net_down'])}/s",
        ])
    if row['gpu'] is not None:
        lines.extend([
            "*━━━━━━━━━━━GPU━━━━━━━━━━━*",
            f"🎮 GPU: {row['gpu']:.1f}%",
        ])
        if row['vram_used'] is not None:
            lines.append(
                f"🗄️ VRAM: {row['vram_used']:.0f} / {row['vram_total']:.0f} MiB ({row['vram']:.1f}%)"
            )
        if row['gpu_temp'] is not None:
            lines.append(f"🌡️ GPU Temp: {row['gpu_temp']:.0f} °C")

    disks = json.loads(row['disks']) if row['disks'] else []
    if disks:
        lines.append("*━━━━━━━━━━━DISKS━━━━━━━━━━*")
        for d in disks:
            line = (
                f"💾 {d['mount']}: {disk_bar(d['percent'])} "
                f"{d['percent']:.0f}% ({human_bytes(d['used'])} / {human_bytes(d['total'])})"
            )
            if d['percent'] >= 90:
                line += "❗"
            lines.append(line)
    return "\n".join(lines)

# ───────────────────────- UI helpers ───────────────────────────────────────
def status_keyboard(secret: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔢 Список", callback_data=f"list"),
             InlineKeyboardButton("🔃 Обновить", callback_data=f"status:{secret}"),
        ],
            [InlineKeyboardButton("📊 Все", callback_data=f"graph:all:{secret}")],
            [
                InlineKeyboardButton("📊 CPU", callback_data=f"graph:cpu:{secret}"),
                InlineKeyboardButton("📈 RAM", callback_data=f"graph:ram:{secret}"),
                InlineKeyboardButton("🎮 GPU",  callback_data=f"graph:gpu:{secret}"),
                InlineKeyboardButton("🗄️ VRAM", callback_data=f"graph:vram:{secret}"),
                InlineKeyboardButton("📡 Net", callback_data=f"graph:net:{secret}"),
            ],
            [InlineKeyboardButton("🏎️ Speedtest", callback_data=f"speedtest:{secret}")],
            [
                InlineKeyboardButton("🔄 Reboot",   callback_data=f"reboot:{secret}"),
                InlineKeyboardButton("⏻ Shutdown", callback_data=f"shutdown:{secret}"),
            ],
        ]
    )

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-монитор.\n" + OWNER_HELP)

async def cmd_newkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db = load_db()
    uid = update.effective_user.id
    if ctx.args:
        name = " ".join(ctx.args)[:30]
        # проверяем уникальность имени
        for e in db["secrets"].values():
            if is_owner(e, uid) and e.get("nickname") == name:
                return await update.message.reply_text("❌ Имя уже занято.")
    else:
        base = "key"
        nums = []
        for e in db["secrets"].values():
            if is_owner(e, uid) and (n := e.get("nickname")) and n.startswith(base):
                tail = n[len(base):]
                if tail.isdigit():
                    nums.append(int(tail))
        num = 1
        while num in nums:
            num += 1
        name = f"{base}{num}"

    secret = gen_secret()
    db["secrets"][secret] = {
        "owners": [uid],
        "nickname": name,
        "pending": [],
    }
    db["active"][str(update.effective_chat.id)] = secret
    save_db(db)
    await update.message.reply_text(
        f"Создан секрет `{secret}` (название: {name}).", parse_mode="Markdown"
    )

async def cmd_linkkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Синтаксис: /linkkey <key>")
    secret = ctx.args[0]
    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry:
        return await update.message.reply_text("🚫 Ключ не найден.")
    if update.effective_user.id in entry["owners"]:
        return await update.message.reply_text("✔️ Уже есть доступ.")
    entry["owners"].append(update.effective_user.id)
    db["active"][str(update.effective_chat.id)] = secret
    save_db(db)
    await update.message.reply_text("✅ Ключ добавлен и сделан активным.")

async def cmd_setactive(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("/set <key>")
    secret = ctx.args[0]
    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry or not is_owner(entry, update.effective_user.id):
        return await update.message.reply_text("🚫 Нет доступа.")
    db["active"][str(update.effective_chat.id)] = secret
    save_db(db)
    await update.message.reply_text(f"✅ Активный: `{secret}`", parse_mode="Markdown")

async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db   = load_db()
    uid  = update.effective_user.id
    now  = int(time.time())

    rows = []
    for secret, entry in db["secrets"].items():
        if not is_owner(entry, uid):
            continue

        name = entry.get("nickname") or secret

        row = sql.execute(
            "SELECT ts, cpu, ram FROM metrics "
            "WHERE secret=? ORDER BY ts DESC LIMIT 1",
            (secret,),
        ).fetchone()

        if row:
            ts, cpu, ram = row
            fresh = (now - ts) < 300
            info  = f"🖥️{cpu:.0f}% CPU, 🧠{ram:.0f}% RAM"
        else:
            fresh = False
            info  = "нет данных"

        uptime = "-"
        if row:
            up = sql.execute(
                "SELECT uptime FROM metrics WHERE secret=? ORDER BY ts DESC LIMIT 1",
                (secret,),
            ).fetchone()
            if up and up[0] is not None:
                uptime = str(timedelta(seconds=int(up[0])))

        marker = " <b>❗️ДАННЫЕ УСТАРЕЛИ❗</b>" if not fresh else ""
        rows.append(
            f"<b>{escape(name)}</b> – <code>{escape(secret)}</code>"
            f"\n• {info}, ⏳ {escape(uptime)}{marker}"
            f"\n"
        )

    buttons = [
        InlineKeyboardButton(
            entry.get("nickname") or s,
            callback_data=f"status:{s}",
        )
        for s, entry in list(db["secrets"].items())[:12]
        if is_owner(entry, uid)
    ]
    keyboard = InlineKeyboardMarkup([buttons[i:i + 4] for i in range(0, len(buttons), 4)])

    active = db["active"].get(str(update.effective_chat.id))
    head   = "Твои ключи:" if rows else "Ключей нет. /newkey создаст."
    if active:
        head += f"\n<b>Активный:</b> <code>{escape(active)}</code>"

    await update.message.reply_text(
        head + "\n" + "\n".join(rows),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )

def resolve_secret(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> str | None:
    db = load_db()
    secret = ctx.args[0] if ctx.args else db["active"].get(str(update.effective_chat.id))
    entry = db["secrets"].get(secret) if secret else None
    if not entry or not is_owner(entry, update.effective_user.id):
        return None
    return secret

async def cmd_renamekey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 2:
        return await update.message.reply_text("Синтаксис: /renamekey <key> <new_name>")
    secret, new_name = ctx.args[0], " ".join(ctx.args[1:])[:30]
    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry or not is_owner(entry, update.effective_user.id):
        return await update.message.reply_text("🚫 Нет доступа.")
    uid = update.effective_user.id
    for s, e in db["secrets"].items():
        if s != secret and is_owner(e, uid) and e.get("nickname") == new_name:
            return await update.message.reply_text("❌ Имя уже занято.")
    entry["nickname"] = new_name
    save_db(db)
    await update.message.reply_text(f"✅ `{secret}` → {new_name}", parse_mode="Markdown")

async def cmd_delkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Синтаксис: /delkey <ключ или имя>")
    arg = " ".join(ctx.args).strip()
    db = load_db()
    uid = update.effective_user.id

    # аргумент может быть полным ключом
    entry = db["secrets"].get(arg)
    if entry:
        if not is_owner(entry, uid):
            return await update.message.reply_text("🚫 Нет доступа.")
        secret = arg
    else:
        matches = [s for s, e in db["secrets"].items() if is_owner(e, uid) and e.get("nickname") == arg]
        if not matches:
            return await update.message.reply_text("Ключ не найден.")
        if len(matches) > 1:
            return await update.message.reply_text("Несколько ключей с таким именем. Укажи полный ключ.")
        secret = matches[0]

    db["secrets"].pop(secret, None)
    for chat, s in list(db["active"].items()):
        if s == secret:
            db["active"].pop(chat)
    save_db(db)
    sql.execute("DELETE FROM metrics WHERE secret=?", (secret,))
    LATEST_TEXT.pop(secret, None)
    await update.message.reply_text(f"🗑️ Удалён ключ {secret}")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    secret = resolve_secret(update, ctx)
    if not secret:
        return await update.message.reply_text("Нет доступа или активного ключа.")
    row = sql.execute(
        "SELECT * FROM metrics WHERE secret=? ORDER BY ts DESC LIMIT 1",
        (secret,),
    ).fetchone()
    if not row:
        return await update.message.reply_text("Нет данных от агента.")
    await update.message.reply_text(
        format_status(row), parse_mode="Markdown", reply_markup=status_keyboard(secret)
    )

async def cmd_setalert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) != 3:
        return await update.message.reply_text(
            "Синтаксис: /setalert <ключ/имя> <метрика> <порог>"
        )

    key, metric, thr = ctx.args
    metric = metric.lower()
    if metric not in {"cpu", "ram", "gpu", "vram"}:
        return await update.message.reply_text("Неизвестная метрика.")
    try:
        threshold = float(thr)
    except ValueError:
        return await update.message.reply_text("Порог должен быть числом.")

    db = load_db()
    uid = str(update.effective_user.id)

    secret = None
    entry = db["secrets"].get(key)
    if entry and is_owner(entry, update.effective_user.id):
        secret = key
    else:
        for s, e in db["secrets"].items():
            if is_owner(e, update.effective_user.id) and e.get("nickname") == key:
                secret = s
                break
    if not secret:
        return await update.message.reply_text("Ключ не найден.")

    alerts = db.setdefault("alerts", {})
    user_cfg = alerts.setdefault(uid, {})
    metric_cfg = user_cfg.setdefault(secret, {})
    metric_cfg[metric] = threshold
    save_db(db)
    await update.message.reply_text(
        f"✅ Алерт для {metric.upper()} {threshold}% сохранён."
    )

async def cmd_delalert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) != 2:
        return await update.message.reply_text(
            "Синтаксис: /delalert <ключ/имя> <метрика>"
        )

    key, metric = ctx.args
    metric = metric.lower()
    if metric not in {"cpu", "ram", "gpu", "vram"}:
        return await update.message.reply_text("Неизвестная метрика.")

    db = load_db()
    uid = str(update.effective_user.id)

    secret = None
    entry = db["secrets"].get(key)
    if entry and is_owner(entry, update.effective_user.id):
        secret = key
    else:
        for s, e in db["secrets"].items():
            if is_owner(e, update.effective_user.id) and e.get("nickname") == key:
                secret = s
                break
    if not secret:
        return await update.message.reply_text("Ключ не найден.")

    alerts = db.get("alerts", {})
    user_cfg = alerts.get(uid)
    if not user_cfg or secret not in user_cfg or metric not in user_cfg[secret]:
        return await update.message.reply_text("Алерт не найден.")

    user_cfg[secret].pop(metric, None)
    if not user_cfg[secret]:
        user_cfg.pop(secret)
    if not user_cfg:
        alerts.pop(uid)

    last_key = f"{uid}:{secret}:{metric}"
    db.get("alert_last", {}).pop(last_key, None)

    save_db(db)
    await update.message.reply_text("🗑️ Алерт удалён")

# ───────────────────- Plot helpers ─────────────────────────────────────────
def _find_gaps(ts, factor: float = 2.0):
    if len(ts) < 2:
        return [(0, len(ts) - 1)], [], 0

    intervals = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    med = median(intervals) if intervals else 0
    if med <= 0:
        med = max(intervals) if intervals else 60  # fallback 60 s
    thr = med * factor

    segments, gaps = [], []
    start = 0
    for i, dt in enumerate(intervals, start=1):
        if dt > thr:
            segments.append((start, i - 1))
            gaps.append((ts[i - 1], ts[i]))
            start = i
    segments.append((start, len(ts) - 1))
    return segments, gaps, thr


def _plot_segments(ax, ts, ys, segments, *args, **kwargs):
    first = True
    col = None
    for s, e in segments:
        if first:
            line, = ax.plot(ts[s:e+1], ys[s:e+1], *args, **kwargs)
            col = line.get_color()
            first = False
        else:
            kw = dict(kwargs)
            kw.pop("label", None)
            kw["color"] = col
            ax.plot(ts[s:e+1], ys[s:e+1], *args, **kw)
def _make_figure(seconds: int):
    """Return figure sized proportionally to the requested time span."""
    days = seconds / 86_400
    width = 12
    if days >= 1:
        width *= 1.5 if days < 1.5 else days
    dpi = 500
    fig, ax = plt.subplots(figsize=(width, 6), dpi=dpi)

    base = 9
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 4,
        "axes.labelsize": base + 2,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
    })
    return fig, ax
def plot_metric(secret: str, metric: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]

    idx_map = {
        "cpu": (1, "CPU %", "%", (0, 100)),
        "ram": (2, "RAM %", "%", (0, 100)),
        "gpu": (3, "GPU %", "%", (0, 100)),
        "vram": (4, "VRAM %", "%", (0, 100)),
        "net_up": (5, "Net Up", "B/s", None),
        "net_down": (6, "Net Down", "B/s", None),
    }
    col_idx, label, ylab, ylim = idx_map[metric]
    ys = [np.nan if rows[i][col_idx] is None else rows[i][col_idx] for i in range(len(rows))]

    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    _plot_segments(ax, ts, ys, segments, linewidth=1.5)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_title(f"{label} за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.3)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_net(secret: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]
    up = [np.nan if r[5] is None else r[5] for r in rows]
    down = [np.nan if r[6] is None else r[6] for r in rows]
    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    _plot_segments(ax, ts, up, segments, label="Up", linewidth=1.2)
    _plot_segments(ax, ts, down, segments, label="Down", linewidth=1.2)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_title(f"Net за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel("B/s")
    ax.grid(True, linestyle="--", linewidth=0.3)
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
def plot_all_metrics(secret: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]
    segments, gaps, _ = _find_gaps(ts)

    cpu = [r[1] for r in rows]
    ram = [r[2] for r in rows]
    gpu = [np.nan if r[3] is None else r[3] for r in rows]
    vram = [np.nan if r[4] is None else r[4] for r in rows]

    plt.style.use("dark_background")

    fig, ax = _make_figure(seconds)


    for ys, lab in ((cpu, "CPU %"), (ram, "RAM %")):
        _plot_segments(ax, ts, ys, segments, label=lab, linewidth=1.2)
    if not all(np.isnan(g) for g in gpu):
        _plot_segments(ax, ts, gpu, segments, label="GPU %", linewidth=1.2)
    if not all(np.isnan(v) for v in vram):
        _plot_segments(ax, ts, vram, segments, label="VRAM %", linewidth=1.2)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_ylim(0, 100)
    ax.set_title(f"Все метрики за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel("%")
    ax.grid(True, linestyle="--", linewidth=0.3)
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# ─────────────────────- Callback handler ───────────────────────────────────
async def cb_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    parts = q.data.split(":")
    action = parts[0]
    db = load_db()

    # ───── status / reboot / shutdown (старые) ─────
    if action == "status":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            return await q.edit_message_text("🚫 Нет доступа.")
        row = sql.execute(
            "SELECT * FROM metrics WHERE secret=? ORDER BY ts DESC LIMIT 1",
            (secret,),
        ).fetchone()
        if not row:
            return await q.edit_message_text("Нет данных от агента.")
        return await q.edit_message_text(
            format_status(row), parse_mode="Markdown", reply_markup=status_keyboard(secret)
        )
    if action in {"reboot", "shutdown"}:
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            return await q.edit_message_text("🚫 Нет доступа.")
        entry.setdefault("pending", []).append(action)
        save_db(db)
        return await q.edit_message_text(f"☑️ *{action}* поставлена в очередь.", parse_mode="Markdown")
    if action == "list":
        uid = q.from_user.id
        now = int(time.time())
        rows = []

        for secret, entry in db["secrets"].items():
            if not is_owner(entry, uid):
                continue

            name = entry.get("nickname") or secret

            row = sql.execute(
                "SELECT ts, cpu, ram FROM metrics "
                "WHERE secret=? ORDER BY ts DESC LIMIT 1",
                (secret,),
            ).fetchone()

            if row:
                ts, cpu, ram = row
                fresh = (now - ts) < 300
                info = f"🖥️{cpu:.0f}% CPU, 🧠{ram:.0f}% RAM"
            else:
                fresh = False
                info = "нет данных"

            uptime = "-"
            up = sql.execute(
                "SELECT uptime FROM metrics WHERE secret=? ORDER BY ts DESC LIMIT 1",
                (secret,),
            ).fetchone()
            if up and up[0] is not None:
                uptime = str(timedelta(seconds=int(up[0])))

            marker = " <b>❗️ДАННЫЕ УСТАРЕЛИ❗</b>" if not fresh else ""
            rows.append(
                f"<b>{escape(name)}</b> – <code>{escape(secret)}</code>"
                f"\n• {info}, ⏳ {escape(uptime)}{marker}\n"
            )

        # те же кнопочки, но теперь они уедут в reply_markup
        buttons = [
            InlineKeyboardButton(
                entry.get("nickname") or s,
                callback_data=f"status:{s}",
            )
            for s, entry in db["secrets"].items()
            if is_owner(entry, uid)
        ]
        keyboard = InlineKeyboardMarkup(
            [buttons[i:i + 4] for i in range(0, len(buttons), 4)]
        )

        active = db["active"].get(str(q.message.chat_id))
        head = "Твои ключи:" if rows else "Ключей нет. /newkey создаст."
        if active:
            head += f"\n<b>Активный:</b> <code>{escape(active)}</code>"

        await q.edit_message_text(
            head + "\n" + "\n".join(rows),
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
        return



    if action == "speedtest":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        entry.setdefault("pending", []).append("speedtest")
        save_db(db)

        await q.answer()
        msg = await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text="⏳ Тестируем скорость…"
        )

        ctx.job_queue.run_repeating(
            callback=check_speedtest_done,
            interval=3,
            data={
                "secret": secret,
                "chat_id": msg.chat_id,
                "msg_id": msg.message_id,
            },
        )
        return
    # ───── graph selection ─────
    if action == "graph":
        metric = parts[1]


        if len(parts) == 3:  # graph:<metric>:<secret>
            secret = parts[2]
            kb = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("10 мин", callback_data=f"graph:{metric}:600:{secret}"),
                        InlineKeyboardButton("1 час", callback_data=f"graph:{metric}:3600:{secret}"),
                        InlineKeyboardButton("24 ч", callback_data=f"graph:{metric}:86400:{secret}"),
                        InlineKeyboardButton("7 д", callback_data=f"graph:{metric}:604800:{secret}"),
                    ],
                    [InlineKeyboardButton("◀️ Назад", callback_data=f"status:{secret}")],
                ]
            )
            return await q.edit_message_reply_markup(reply_markup=kb)


        seconds = int(parts[2])
        secret = parts[3]

        if metric == "all":
            buf = plot_all_metrics(secret, seconds)
            caption = f"Все метрики за {timedelta(seconds=seconds)}"
        elif metric == "net":
            buf = plot_net(secret, seconds)
            caption = f"NET за {timedelta(seconds=seconds)}"
        else:
            buf = plot_metric(secret, metric, seconds)
            caption = f"{metric.upper()} за {timedelta(seconds=seconds)}"

        if not buf:
            return await q.edit_message_text("Данных за этот период нет.")

        caption = f"{metric.upper()} за {timedelta(seconds=seconds)}"
        if seconds >= 86400:
            doc = InputFile(buf, filename=f"{metric}_{seconds}.png")
            await ctx.bot.send_document(
                chat_id=q.message.chat_id,
                document=doc,
                caption=caption,
            )
        else:
            await ctx.bot.send_photo(
                chat_id=q.message.chat_id,
                photo=buf,
                caption=caption,
            )
        return

# ────────────────────────── FastAPI for agents ─────────────────────────────
app = FastAPI()

class PushPayload(BaseModel):
    cpu: float | None = None
    ram: float | None = None
    ram_used: float | None = None
    ram_total: float | None = None
    swap: float | None = None
    swap_used: float | None = None
    swap_total: float | None = None
    gpu: float | None = None
    vram: float | None = None
    vram_used: float | None = None
    vram_total: float | None = None
    cpu_temp: float | None = None
    gpu_temp: float | None = None
    net_up: float | None = None
    net_down: float | None = None
    uptime: int | None = None
    disks: list[dict] | None = None
    text: str | None = None

@app.post("/api/push/{secret}")
async def push(secret: str, payload: PushPayload):
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404)

    if payload.text:
        LATEST_TEXT[secret] = payload.text

    if payload.cpu is None or payload.ram is None:
        return {"ok": True}

    record_metric(secret, payload.model_dump())
    await maybe_send_alerts(secret, payload.model_dump())

    return {"ok": True}

@app.get("/api/pull/{secret}")
async def pull(secret: str):
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404)
    cmds = db["secrets"][secret].get("pending", [])
    db["secrets"][secret]["pending"] = []
    save_db(db)
    return {"commands": cmds}

# ────────────────────────── Bootstrap ──────────────────────────────────────
def start_uvicorn():
    kwargs = dict(host="0.0.0.0", port=API_PORT, log_level="info")
    if CERT_FILE.exists() and KEY_FILE.exists():
        kwargs.update(ssl_certfile=str(CERT_FILE), ssl_keyfile=str(KEY_FILE))
        log.info("🔐 TLS enabled.")
    else:
        log.warning("⚠️  TLS disabled.")
    uvicorn.run(app, **kwargs)

def main():
    threading.Thread(target=start_uvicorn, daemon=True).start()
    log.info("🌐 FastAPI on port %s", API_PORT)

    global TG_APP
    async def post_init(app: Application) -> None:
        app.create_task(_purge_loop())

    TG_APP = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    TG_APP.add_handler(CommandHandler(["start", "help"], cmd_start))
    TG_APP.add_handler(CommandHandler("newkey", cmd_newkey))
    TG_APP.add_handler(CommandHandler("linkkey", cmd_linkkey))
    TG_APP.add_handler(CommandHandler("set", cmd_setactive))
    TG_APP.add_handler(CommandHandler("list", cmd_list))
    TG_APP.add_handler(CommandHandler("status", cmd_status))
    TG_APP.add_handler(CommandHandler("renamekey", cmd_renamekey))
    TG_APP.add_handler(CommandHandler("delkey", cmd_delkey))
    TG_APP.add_handler(CommandHandler("setalert", cmd_setalert))
    TG_APP.add_handler(CommandHandler("delalert", cmd_delalert))
    TG_APP.add_handler(CallbackQueryHandler(cb_action))

    # очищаем старые метрики и запускаем периодическую уборку
    purge_old_metrics()

    log.info("🤖 Polling…")
    TG_APP.run_polling(allowed_updates=["message", "callback_query"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bye.")
