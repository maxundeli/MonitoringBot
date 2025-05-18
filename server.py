from __future__ import annotations

"""remote_bot_server"""

import io
import json
import logging
import os
import re
import secrets
import sqlite3
import string
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
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
MAX_CPU: defaultdict[str, float] = defaultdict(float)
MAX_RAM: defaultdict[str, float] = defaultdict(float)

CERT_FILE = Path(os.getenv("SSL_CERT", "cert.pem"))
KEY_FILE = Path(os.getenv("SSL_KEY", "key.pem"))

# matplotlib без X-сервера
matplotlib.use("Agg")

# ────────────────────────── helpers ────────────────────────────────────────
CPU_RE = re.compile(r"CPU:\s*([\d.]+)%")
RAM_RE = re.compile(r"RAM:.*\(([\d.]+)%\)")
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
    con = sqlite3.connect(METRIC_DB, check_same_thread=False)
    con.execute(
        """CREATE TABLE IF NOT EXISTS metrics(
                secret TEXT,
                ts INTEGER,
                cpu REAL,
                ram REAL
        )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_secret_ts ON metrics(secret, ts)"
    )
    return con

sql = _init_metric_db()

def record_metric(secret: str, cpu: float, ram: float):
    log.debug("REC %s cpu=%s ram=%s", secret, cpu, ram)
    sql.execute(
        "INSERT INTO metrics(secret, ts, cpu, ram) VALUES(?,?,?,?)",
        (secret, int(time.time()), cpu, ram),
    )
    sql.commit()

def fetch_metrics(secret: str, since: int) -> List[tuple[int, float]]:
    rows = sql.execute(
        "SELECT ts, cpu, ram FROM metrics WHERE secret=? AND ts>=? ORDER BY ts ASC",
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
            [
                InlineKeyboardButton("📊 CPU", callback_data=f"graph:cpu:{secret}"),
                InlineKeyboardButton("📈 RAM", callback_data=f"graph:ram:{secret}"),
            ],
            [InlineKeyboardButton("🔃 Обновить", callback_data=f"status:{secret}")],
            [
                InlineKeyboardButton("🔄 Reboot", callback_data=f"reboot:{secret}"),
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
    db = load_db()
    uid = update.effective_user.id
    lines = [f"`{s}` – {e['nickname']}" for s, e in db["secrets"].items() if is_owner(e, uid)]
    active = db["active"].get(str(update.effective_chat.id))
    msg = ("Твои ключи:\n" + "\n".join(lines)) if lines else "Ключей нет. /newkey создаст."
    if active:
        msg += f"\n*Активный:* `{active}`"
    await update.message.reply_text(msg, parse_mode="Markdown")

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
def plot_metric(secret: str, metric: str, seconds: int) -> io.BytesIO | None:
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    log.info("Plot %s %s %s -> %d rows",
             secret, metric, seconds, len(rows))
    if not rows:
        return None
    ts = [datetime.fromtimestamp(r[0]) for r in rows]
    if metric == "cpu":
        ys = [r[1] for r in rows]
        label = "CPU %"
    else:
        ys = [r[2] for r in rows]
        label = "RAM %"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(ts, ys, linewidth=1.5)
    ax.set_title(f"{label} за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel("%")
    ax.grid(True, linestyle="--", linewidth=0.3)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
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
                    ],
                    [InlineKeyboardButton("◀️ Назад", callback_data=f"status:{secret}")],
                ]
            )
            return await q.edit_message_reply_markup(reply_markup=kb)


        seconds = int(parts[2])
        secret = parts[3]

        buf = plot_metric(secret, metric, seconds)
        if not buf:
            return await q.edit_message_text("Данных за этот период нет.")

        caption = f"{metric.upper()} за {timedelta(seconds=seconds)}"
        await ctx.bot.send_photo(chat_id=q.message.chat_id, photo=buf, caption=caption)
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
    db["secrets"][secret]["status"] = payload.text
    save_db(db)

    cpu_m = CPU_RE.search(payload.text)
    ram_m = RAM_RE.search(payload.text)
    if cpu_m and ram_m:
        try:
            cpu_val = float(cpu_m.group(1))
            ram_val = float(ram_m.group(1))
            MAX_CPU[secret] = max(MAX_CPU[secret], cpu_val)
            MAX_RAM[secret] = max(MAX_RAM[secret], ram_val)
        except ValueError:
            pass

    COUNTERS[secret] += 1
    if COUNTERS[secret] >= 6:
        COUNTERS[secret] = 0
        max_cpu = MAX_CPU[secret]
        max_ram = MAX_RAM[secret]
        MAX_CPU[secret] = 0.0
        MAX_RAM[secret] = 0.0
        record_metric(secret, max_cpu, max_ram)

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
