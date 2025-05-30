from __future__ import annotations

"""remote_bot_server"""

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
import time
from collections import defaultdict
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
SUM_CPU = defaultdict(float)
SUM_RAM = defaultdict(float)
SUM_GPU = defaultdict(float)
SUM_VRAM = defaultdict(float)
COUNTERS = defaultdict(int)

CERT_FILE = Path(os.getenv("SSL_CERT", "cert.pem"))
KEY_FILE = Path(os.getenv("SSL_KEY", "key.pem"))

# matplotlib без X-сервера
matplotlib.use("Agg")

# ────────────────────────── helpers ────────────────────────────────────────
CPU_RE = re.compile(r"CPU:\s*([\d.]+)%")
RAM_RE = re.compile(r"RAM:.*\(([\d.]+)%\)")
GPU_RE  = re.compile(r"GPU:\s*([\d.]+)%")
VRAM_RE = re.compile(r"VRAM:.*\(([\d.]+)%\)")
UPTIME_RE = re.compile(r"Uptime:\s*([^\n]+)")
COUNTERS: defaultdict[str, int] = defaultdict(int)

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

# ───────────────────── SQLite helpers ──────────────────────────────────────
def _init_metric_db() -> sqlite3.Connection:
    con = sqlite3.connect(METRIC_DB, check_same_thread=False, isolation_level=None)
    con.execute(
        """CREATE TABLE IF NOT EXISTS metrics(
               secret TEXT,
               ts     INTEGER,
               cpu    REAL,
               ram    REAL,
               gpu    REAL,
               vram   REAL
        )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_secret_ts ON metrics(secret, ts)"
    )
    return con

sql = _init_metric_db()

def record_metric(secret: str, cpu: float, ram: float,
                  gpu: float | None, vram: float | None):
    sql.execute(
        "INSERT INTO metrics(secret, ts, cpu, ram, gpu, vram) VALUES(?,?,?,?,?,?)",
        (secret, int(time.time()), cpu, ram, gpu, vram),
    )

def fetch_metrics(secret: str, since: int) -> List[tuple[int, float]]:
    rows = sql.execute(
        "SELECT ts, cpu, ram, gpu, vram FROM metrics WHERE secret=? AND ts>=? ORDER BY ts ASC",
        (secret, since),
    ).fetchall()
    return rows

# ──────────────────────── JSON DB helpers ──────────────────────────────────
def load_db() -> Dict[str, Any]:
    if DB_FILE.exists():
        data = json.loads(DB_FILE.read_text())
    else:
        data = {}
    data.setdefault("secrets", {})
    data.setdefault("active", {})
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
    "/renamekey <ключ> <имя> – переименовать."
)
def gen_secret(n: int = 20):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

def is_owner(entry: Dict[str, Any], user_id: int) -> bool:
    return user_id in entry.get("owners", [])

# ───────────────────────- UI helpers ───────────────────────────────────────
def status_keyboard(secret: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Все", callback_data=f"graph:all:{secret}")],
            [
                InlineKeyboardButton("📊 CPU",  callback_data=f"graph:cpu:{secret}"),
                InlineKeyboardButton("📈 RAM",  callback_data=f"graph:ram:{secret}"),
            ],
            [
                InlineKeyboardButton("🎮 GPU",  callback_data=f"graph:gpu:{secret}"),
                InlineKeyboardButton("🗄️ VRAM", callback_data=f"graph:vram:{secret}"),
            ],
            [
               InlineKeyboardButton("🏎️ Speedtest", callback_data=f"speedtest:{secret}"),
               InlineKeyboardButton("🔃 Обновить", callback_data=f"status:{secret}"),
            ],
            [
                InlineKeyboardButton("🔄 Reboot",   callback_data=f"reboot:{secret}"),
                InlineKeyboardButton("⏻ Shutdown", callback_data=f"shutdown:{secret}"),
            ],
        ]
    )

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот-монитор.\n" + OWNER_HELP)

async def cmd_newkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = " ".join(ctx.args)[:30] if ctx.args else "PC"
    db = load_db()
    secret = gen_secret()
    db["secrets"][secret] = {
        "owners": [update.effective_user.id],
        "nickname": name,
        "status": None,
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
        if entry.get("status"):
            m = UPTIME_RE.search(entry["status"])
            if m:
                uptime = m.group(1)

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
    entry["nickname"] = new_name
    save_db(db)
    await update.message.reply_text(f"✅ `{secret}` → {new_name}", parse_mode="Markdown")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    secret = resolve_secret(update, ctx)
    if not secret:
        return await update.message.reply_text("Нет доступа или активного ключа.")
    entry = load_db()["secrets"].get(secret)
    if not entry or not entry["status"]:
        return await update.message.reply_text("Нет данных от агента.")
    await update.message.reply_text(
        entry["status"], parse_mode="Markdown", reply_markup=status_keyboard(secret)
    )

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
    long_span = seconds >= 86_400  # ≥ 1 day
    dpi = 500 if long_span else 150
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

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
        "cpu": (1, "CPU %"),
        "ram": (2, "RAM %"),
        "gpu": (3, "GPU %"),
        "vram": (4, "VRAM %"),
    }
    col_idx, label = idx_map[metric]
    ys = [np.nan if rows[i][col_idx] is None else rows[i][col_idx] for i in range(len(rows))]

    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    _plot_segments(ax, ts, ys, segments, linewidth=1.5)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_title(f"{label} за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", linewidth=0.3)
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
        if not entry["status"]:
            return await q.edit_message_text("Нет данных от агента.")
        return await q.edit_message_text(
            entry["status"], parse_mode="Markdown", reply_markup=status_keyboard(secret)
        )
    if action in {"reboot", "shutdown"}:
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            return await q.edit_message_text("🚫 Нет доступа.")
        entry.setdefault("pending", []).append(action)
        save_db(db)
        return await q.edit_message_text(f"☑️ *{action}* поставлена в очередь.", parse_mode="Markdown")
    if action == "speedtest":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return

        # ставим команду в очередь для агента
        entry.setdefault("pending", []).append("speedtest")
        save_db(db)

        await q.answer()  # закрываем «часики» на кнопке
        await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text="⏳ Тестируем скорость…"
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

class StatusPayload(BaseModel):
    text: str

@app.post("/api/push/{secret}")
async def push(secret: str, payload: StatusPayload):
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404)

    # Сохраняем последний статус для /status в Telegram-боте
    db["secrets"][secret]["status"] = payload.text
    save_db(db)

    # ───── парсинг чисел ─────
    cpu_m  = CPU_RE.search(payload.text)
    ram_m  = RAM_RE.search(payload.text)
    gpu_m  = GPU_RE.search(payload.text)
    vram_m = VRAM_RE.search(payload.text)

    # CPU и RAM считаем обязательными: если их нет — просто выходим
    if not (cpu_m and ram_m):
        return {"ok": True}

    try:
        cpu_val  = float(cpu_m.group(1))
        ram_val  = float(ram_m.group(1))
        gpu_val  = float(gpu_m.group(1)) if gpu_m else None
        vram_val = float(vram_m.group(1)) if vram_m else None
    except ValueError:
        # Невалидные числа — игнорируем этот пуш
        return {"ok": True}

    # ───── аккумуляторы ─────
    SUM_CPU[secret]  += cpu_val
    SUM_RAM[secret]  += ram_val
    SUM_GPU[secret]  += gpu_val  if gpu_val  is not None else 0.0
    SUM_VRAM[secret] += vram_val if vram_val is not None else 0.0
    COUNTERS[secret] += 1

    # Каждые 6 пушей (примерно 1 минута при шаге 10 с) пишем усреднённое
    if COUNTERS[secret] >= 6:
        avg_cpu  = SUM_CPU[secret]  / COUNTERS[secret]
        avg_ram  = SUM_RAM[secret]  / COUNTERS[secret]
        avg_gpu  = (
            SUM_GPU[secret] / COUNTERS[secret] if gpu_m else None
        )
        avg_vram = (
            SUM_VRAM[secret] / COUNTERS[secret] if vram_m else None
        )

        record_metric(secret, avg_cpu, avg_ram, avg_gpu, avg_vram)

        # сбрасываем счётчики
        SUM_CPU[secret]  = 0.0
        SUM_RAM[secret]  = 0.0
        SUM_GPU[secret]  = 0.0
        SUM_VRAM[secret] = 0.0
        COUNTERS[secret] = 0

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

    app_tg = ApplicationBuilder().token(TOKEN).build()
    app_tg.add_handler(CommandHandler(["start", "help"], cmd_start))
    app_tg.add_handler(CommandHandler("newkey", cmd_newkey))
    app_tg.add_handler(CommandHandler("linkkey", cmd_linkkey))
    app_tg.add_handler(CommandHandler("set", cmd_setactive))
    app_tg.add_handler(CommandHandler("list", cmd_list))
    app_tg.add_handler(CommandHandler("status", cmd_status))
    app_tg.add_handler(CommandHandler("renamekey", cmd_renamekey))
    app_tg.add_handler(CallbackQueryHandler(cb_action))

    log.info("🤖 Polling…")
    app_tg.run_polling(allowed_updates=["message", "callback_query"])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bye.")
