"""remote_bot_server.py – Telegram bot + FastAPI backend

Version 2025‑05‑17‑persist‑fix
──────────────────────────────
• Запоминает BOT_TOKEN в .env при первом вводе.
• Поддерживает /status с инлайновыми кнопками 🔄 reboot / ⏻ shutdown.
• Исправлен обрыв файла: функция main() снова запускает и HTTP API, и
  Telegram‑бота.

Dependencies: python‑telegram‑bot==20.* fastapi uvicorn[standard] psutil requests
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import string
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

# ────────────────────────── CONFIG ─────────────────────────────────────────
ENV_FILE = Path(".env")
DB_FILE = Path("db.json")
API_PORT = int(os.getenv("PORT", "8000"))


def _load_dotenv() -> None:
    """Populate os.environ from .env if variables not already set."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    TOKEN = input("Enter Telegram BOT_TOKEN: ").strip()
    if not TOKEN:
        print("❌ BOT_TOKEN is required. Exiting.")
        sys.exit(1)
    ENV_FILE.write_text((ENV_FILE.read_text() if ENV_FILE.exists() else "") + f"BOT_TOKEN={TOKEN}\n")
    print("🔏 BOT_TOKEN saved to .env")

# ────────────────────────── LOGGING ────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("remote-bot")

# ────────────────────────── Persistence helpers ───────────────────────────

def load_db() -> Dict[str, Any]:
    if DB_FILE.exists():
        data = json.loads(DB_FILE.read_text())
    else:
        data = {}
    data.setdefault("secrets", {})
    data.setdefault("active", {})
    return data


def save_db(db: Dict[str, Any]) -> None:
    DB_FILE.write_text(json.dumps(db, indent=2))

# ────────────────────────── Telegram command handlers ─────────────────────
OWNER_HELP = (
    "Команды:\n"
    "/newkey [имя] – создать секрет.\n"
    "/setactivekey <секрет> – выбрать активный ключ.\n"
    "/list – показать все ключи.\n"
    "/status [секрет] – метрики ПК + кнопки."
)


def gen_secret(n: int = 20) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для мониторинга ПК.\n" + OWNER_HELP)


async def cmd_newkey(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nickname = " ".join(ctx.args)[:30] if ctx.args else "PC"
    db = load_db()
    secret = gen_secret()
    db["secrets"][secret] = {
        "owner_id": update.effective_user.id,
        "nickname": nickname,
        "status": None,
        "pending": [],
    }
    db["active"][str(update.effective_chat.id)] = secret
    save_db(db)
    await update.message.reply_text(
        f"Секрет `{secret}` (название: {nickname}) создан и сделан активным.\n" + OWNER_HELP,
        parse_mode="Markdown",
    )


async def cmd_setactive(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text("Синтаксис: /setactivekey <секрет>")
    secret = ctx.args[0]
    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry or entry["owner_id"] != update.effective_user.id:
        return await update.message.reply_text("🚫 Ключ не найден или чужой.")
    db["active"][str(update.effective_chat.id)] = secret
    save_db(db)
    await update.message.reply_text(f"✅ Активный ключ: `{secret}`", parse_mode="Markdown")


async def cmd_list(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    db = load_db()
    lines = [f"`{s}` – {e['nickname']}" for s, e in db["secrets"].items() if e["owner_id"] == update.effective_user.id]
    active = db["active"].get(str(update.effective_chat.id))
    msg = ("Твои ключи:\n" + "\n".join(lines)) if lines else "Ключей нет. /newkey создаст."
    if active:
        msg += f"\n*Активный:* `{active}`"
    await update.message.reply_text(msg, parse_mode="Markdown")


# Helper: pick secret for current chat

def resolve_secret(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> str | None:
    db = load_db()
    secret = ctx.args[0] if ctx.args else db["active"].get(str(update.effective_chat.id))
    if not secret:
        return None
    entry = db["secrets"].get(secret)
    if not entry or entry["owner_id"] != update.effective_user.id:
        return None
    return secret


async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    secret = resolve_secret(update, ctx)
    if not secret:
        return await update.message.reply_text("Нет активного ключа и аргумент не передан.")
    entry = load_db()["secrets"].get(secret)
    if not entry or not entry["status"]:
        return await update.message.reply_text("😴 Нет данных от агента.")

    kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reboot", callback_data=f"reboot:{secret}"), InlineKeyboardButton("⏻ Shutdown", callback_data=f"shutdown:{secret}")]])
    await update.message.reply_text(entry["status"], parse_mode="Markdown", reply_markup=kb)


async def cb_action(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data:
        return
    await q.answer()
    log.info("Callback user=%s chat=%s data=%s", q.from_user.id, q.message.chat.id, q.data)

    try:
        action, secret = q.data.split(":", 1)
    except ValueError:
        return

    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry or entry["owner_id"] not in {q.from_user.id, q.message.chat.id}:
        return await q.edit_message_text("🚫 Ключ не найден или чужой.")

    if action not in {"reboot", "shutdown"}:
        return

    entry.setdefault("pending", []).append(action)
    save_db(db)
    await q.edit_message_text(f"☑️ Команда *{action}* поставлена в очередь.", parse_mode="Markdown")

# ────────────────────────── FastAPI for agents ─────────────────────────────
app = FastAPI()


class StatusPayload(BaseModel):
    text: str


@app.post("/api/push/{secret}")
async def push(secret: str, payload: StatusPayload):
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404, "secret unknown")
    db["secrets"][secret]["status"] = payload.text
    save_db(db)
    return {"ok": True}


@app.get("/api/pull/{secret}")
async def pull(secret: str):
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404, "secret unknown")
    cmds = db["secrets"][secret].get("pending", [])
    db["secrets"][secret]["pending"] = []
    save_db(db)
    return {"commands": cmds}

# ────────────────────────── Bootstrap ──────────────────────────────────────

def start_uvicorn():
    """Run FastAPI in a background thread."""
    uvicorn.run(app, host="0.0.0.0", port=API_PORT, log_level="info")


def main() -> None:
    """Entry-point that launches API *and* Telegram bot."""
    # 1. Start HTTP API in a daemon thread
    threading.Thread(target=start_uvicorn, daemon=True).start()
    log.info("🌐 FastAPI on port %s", API_PORT)

    # 2. Build Telegram application and add handlers
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler(["start", "help"], cmd_start))
    application.add_handler(CommandHandler("newkey", cmd_newkey))
    application.add_handler(CommandHandler("setactivekey", cmd_setactive))
    application.add_handler(CommandHandler("list", cmd_list))
    application.add_handler(CommandHandler("status", cmd_status))
    application.add_handler(CallbackQueryHandler(cb_action))

    # 3. Run polling (will block until Ctrl-C)
    log.info("🤖 Telegram polling…")
    application.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bye.")