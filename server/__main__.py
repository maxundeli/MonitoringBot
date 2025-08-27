from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor

"""remote_bot_server"""
import asyncio
import json
import logging
import os
import re
import secrets
from html import escape
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
import time
import string
from telegram import InputFile
import io
import subprocess
import sys
import threading
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from .db import sql, load_db, save_db, record_metric, purge_old_metrics
from .graphs import (
    parse_timespan,
    plot_custom,
    plot_metric,
    plot_net,
    plot_all_metrics,
)

# ────────────────────────── CONFIG ─────────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
API_PORT = int(os.getenv("PORT", "8000"))
# порт UDP-эхо сервера для проверки стабильности связи
UDP_TEST_PORT = int(os.getenv("UDP_TEST_PORT", "9999"))

# последний текстовый статус от клиентов (speedtest и пр.)
LATEST_TEXT: Dict[str, str] = {}
# диагностические отчёты, отправляемые агентом
LATEST_DIAG: Dict[str, Optional[str]] = {}
# временные метрики, отправляемые командой /status
LATEST_STATUS: Dict[str, Dict[str, Any]] = {}
# результаты тестов стабильности связи
LATEST_STAB: Dict[str, Dict[str, Any]] = {}
_MISSING = object()

# активные WebSocket-соединения с агентами
ACTIVE_WS: Dict[str, WebSocket] = {}

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

class UDPEchoProtocol(asyncio.DatagramProtocol):
    """Простой UDP-эхо сервер для проверки связи."""

    def __init__(self):
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr) -> None:  # type: ignore[override]
        if self.transport:
            self.transport.sendto(data, addr)


def start_udp_echo() -> None:
    async def _run() -> None:
        await asyncio.get_running_loop().create_datagram_endpoint(
            UDPEchoProtocol, local_addr=("0.0.0.0", UDP_TEST_PORT)
        )
        await asyncio.Future()

    asyncio.run(_run())

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

async def send_or_queue(secret: str, cmd: str) -> None:
    """Попытаться отправить команду агенту через WebSocket либо поставить в очередь."""
    ws = ACTIVE_WS.get(secret)
    if ws:
        try:
            await ws.send_json({"commands": [cmd]})
            return
        except Exception as exc:
            log.warning("WS send failed: %s", exc)
    db = load_db()
    entry = db["secrets"].setdefault(secret, {})
    entry.setdefault("pending", []).append(cmd)
    save_db(db)

UNIT_NAMES = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
# Для сетевой скорости используем биты
UNIT_NAMES_BITS = ["bit", "Kbit", "Mbit", "Gbit", "Tbit", "Pbit"]

def human_bytes(num: float) -> str:
    for unit in UNIT_NAMES:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} EiB"

def human_net_speed(num_bytes_per_sec: float) -> str:
    """Показать скорость в битах в секунду."""
    num = num_bytes_per_sec * 8
    for unit in UNIT_NAMES_BITS:
        if num < 1000:
            return f"{num:.1f} {unit}/s"
        num /= 1000
    return f"{num:.1f} Ebit/s"

def best_unit(max_val_bytes: float) -> tuple[float, str]:
    """Return scale factor and unit for network speeds in bits."""
    max_bits = max_val_bytes * 8
    scale_bits = 1.0
    idx = 0
    while idx < len(UNIT_NAMES_BITS) - 1 and max_bits >= 1000:
        max_bits /= 1000
        scale_bits *= 1000
        idx += 1
    unit = UNIT_NAMES_BITS[idx] + "/s"
    return scale_bits / 8, unit

def disk_bar(p: float, length: int = 10) -> str:
    filled = int(round(p * length / 100))
    return "█" * filled + "░" * (length - filled)

async def run_plot(func, *args):
    """Run plotting function in a temporary worker process."""
    workers = os.getenv("GRAPH_WORKERS", "1")
    try:
        num = int(workers)
    except ValueError:
        num = 1
    num = max(num, 1)
    ctx = mp.get_context("spawn")
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=num, mp_context=ctx) as ex:
        part = functools.partial(func, *args)
        return await loop.run_in_executor(ex, part)
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


async def check_diag_done(ctx: ContextTypes.DEFAULT_TYPE):
    job = ctx.job
    data = job.data

    secret = data["secret"]
    chat_id = data["chat_id"]
    msg_id = data["msg_id"]

    entry = load_db()["secrets"].get(secret, {})
    if "diag" in entry.get("pending", []):
        return

    result = LATEST_DIAG.get(secret, _MISSING)
    if result is _MISSING:
        start_ts = data.setdefault("start_ts", time.time())
        TIMEOUT = 3 * 60
        if time.time() - start_ts > TIMEOUT:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="⚠️ Диагностика заняла много времени и была прервана.",
            )
            job.schedule_removal()
        return
    if result is None:
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text="⚠️ Диагностика не удалась.",
        )
        LATEST_DIAG.pop(secret, None)
        job.schedule_removal()
        return

    txt = result

    buf = io.BytesIO(txt.encode())
    doc = InputFile(buf, filename="diagnostics.txt")
    await ctx.bot.edit_message_text(
        chat_id=chat_id,
        message_id=msg_id,
        text="📄 Диагностика готова.",
    )
    await ctx.bot.send_document(chat_id=chat_id, document=doc)
    LATEST_DIAG.pop(secret, None)
    job.schedule_removal()


async def check_stability_done(ctx: ContextTypes.DEFAULT_TYPE):
    job = ctx.job
    data = job.data

    secret = data["secret"]
    chat_id = data["chat_id"]
    msg_id = data["msg_id"]
    deadline = data["deadline"]

    result = LATEST_STAB.get(secret)
    if not result:
        if time.time() < deadline:
            return
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text="⚠️ Тест стабильности не завершился.",
        )
        job.schedule_removal()
        return

    LATEST_STAB.pop(secret, None)

    if result.get("error"):
        await ctx.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg_id,
            text="⚠️ Тест стабильности не удался.",
        )
        job.schedule_removal()
        return

    rtts = result.get("rtts") or []
    interval_ms = result.get("interval_ms", 0)
    start_ts = result.get("start_ts", time.time())
    sent = len(rtts)
    lost = sum(1 for r in rtts if r is None)
    loss_pct = (lost / sent * 100) if sent else 0
    pings = [r for r in rtts if r is not None]
    avg = sum(pings) / len(pings) if pings else 0
    diffs = [abs(pings[i] - pings[i - 1]) for i in range(1, len(pings))]
    jitter = sum(diffs) / len(diffs) if diffs else 0

    outages: List[tuple[datetime, datetime]] = []
    cur = None
    for idx, r in enumerate(rtts):
        if r is None:
            if cur is None:
                cur = idx
        else:
            if cur is not None:
                s = datetime.fromtimestamp(start_ts + cur * interval_ms / 1000)
                e = datetime.fromtimestamp(start_ts + idx * interval_ms / 1000)
                outages.append((s, e))
                cur = None
    if cur is not None:
        s = datetime.fromtimestamp(start_ts + cur * interval_ms / 1000)
        e = datetime.fromtimestamp(start_ts + sent * interval_ms / 1000)
        outages.append((s, e))

    lines = [
        f"📶 Потерь пакетов: {lost}/{sent} ({loss_pct:.1f}%)",
        f"Средний пинг: {avg:.1f} мс",
        f"Джиттер: {jitter:.1f} мс",
    ]
    if outages:
        lines.append("Разрывы:")
        for s, e in outages:
            lines.append(f"• {s:%H:%M:%S} – {e:%H:%M:%S}")
    report = "📶 Отчёт по стабильности\n" + "\n".join(lines)

    await ctx.bot.edit_message_text(
        chat_id=chat_id,
        message_id=msg_id,
        text=report,
    )

    times = [
        datetime.fromtimestamp(start_ts + i * interval_ms / 1000) for i in range(sent)
    ]
    ys = [r if r is not None else float("nan") for r in rtts]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, ys, linewidth=0.5)
    for s, e in outages:
        ax.axvspan(s, e, color="red", alpha=0.3)
    ax.set_ylabel("Пинг, мс")
    ax.set_xlabel("Время")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    await ctx.bot.send_photo(
        chat_id=chat_id,
        photo=InputFile(buf, filename="stability.png"),
    )

    job.schedule_removal()

async def check_status_done(ctx: ContextTypes.DEFAULT_TYPE):
    job = ctx.job
    data = job.data

    secret = data["secret"]
    chat_id = data["chat_id"]
    msg_id = data["msg_id"]

    row = LATEST_STATUS.pop(secret, None)
    if not row:
        start_ts = data.setdefault("start_ts", time.time())
        if time.time() - start_ts > 15:
            await ctx.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="⚠️ Статус не получен.",
            )
            job.schedule_removal()
        return

    await ctx.bot.edit_message_text(
        chat_id=chat_id,
        message_id=msg_id,
        text=format_status(row),
        parse_mode="Markdown",
        reply_markup=status_keyboard(secret),
    )
    job.schedule_removal()



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


# ───────────────────────- Telegram command handlers ────────────────────────
OWNER_HELP = (
    "Команды:\n"
    "/newkey <имя> – создать ключ.\n"
    "/linkkey <ключ> – подписаться.\n"
    "/set <ключ> – сделать активным.\n"
    "/list – показать ключи.\n"
    "/status – статус + кнопки.\n"
    "/plot <ключ/имя> <метрики> <интервал> [предел] [единицы]\n"
    "/renamekey <ключ> <имя> – переименовать.\n"
    "/delkey <ключ/имя> – удалить.\n"
    "/setalert <ключ/имя> <метрика> <порог> – настроить алерт.\n"
    "/delalert <ключ/имя> <метрика> – удалить алерт."
)
def gen_secret(n: int = 20):
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

def is_owner(entry: Dict[str, Any], user_id: int) -> bool:
    return user_id in entry.get("owners", [])

from typing import Sequence, Mapping

# ``format_status`` accepts either a DB row (tuple) or a mapping returned
# by the agent in ``oneshot`` mode.  The column order for tuples matches the
# table schema, while dicts use keys.
def format_status(row: Sequence[Any] | Mapping[str, Any]) -> str:
    if isinstance(row, Mapping):
        ts = row.get("ts")
        cpu = row.get("cpu")
        ram = row.get("ram")
        gpu = row.get("gpu")
        vram = row.get("vram")
        ram_used = row.get("ram_used")
        ram_total = row.get("ram_total")
        swap = row.get("swap")
        swap_used = row.get("swap_used")
        swap_total = row.get("swap_total")
        vram_used = row.get("vram_used")
        vram_total = row.get("vram_total")
        cpu_temp = row.get("cpu_temp")
        gpu_temp = row.get("gpu_temp")
        net_up = row.get("net_up")
        net_down = row.get("net_down")
        uptime = row.get("uptime")
        disks = row.get("disks")
        top_procs = row.get("top_procs")
    else:
        ts = row[1]
        cpu = row[2]
        ram = row[3]
        gpu = row[4]
        vram = row[5]
        ram_used = row[6]
        ram_total = row[7]
        swap = row[8]
        swap_used = row[9]
        swap_total = row[10]
        vram_used = row[11]
        vram_total = row[12]
        cpu_temp = row[13]
        gpu_temp = row[14]
        net_up = row[15]
        net_down = row[16]
        uptime = row[17]
        disks = row[18]
        top_procs = row[19]

    lines = [
        "💻 *PC stats*",
        f"🕒 Updated: {datetime.fromtimestamp(ts).strftime('%d.%m %H:%M:%S')}",
        f"⏳ Uptime: {timedelta(seconds=int(uptime or 0))}",
        "*━━━━━━━━━━━CPU━━━━━━━━━━━*",
        f"🖥️ CPU: {cpu:.1f}%",
        f"🌡️ CPU Temp: {cpu_temp:.1f} °C" if cpu_temp is not None else "🌡️ CPU Temp: N/A",
        "*━━━━━━━━━━━RAM━━━━━━━━━━━*",
        f"🧠 RAM: {human_bytes(ram_used)} / {human_bytes(ram_total)} ({ram:.1f}%)",
        f"🧠 SWAP: {human_bytes(swap_used)} / {human_bytes(swap_total)} ({swap:.1f}%)",
    ]
    procs = json.loads(top_procs) if top_procs else []
    if procs:
        lines.append("*━━━━━━━━━TOP CPU━━━━━━━━━*")
        for p in procs:
            name_raw = p.get('name', '')
            if name_raw and name_raw.lower() == 'system idle process':
                continue
            name = escape_markdown(name_raw[:20], version=1)
            lines.append(
                f"⚙️ {name}: 🖥️ {p['cpu']:.1f}% 🧠 {human_bytes(p['ram'])}"
            )
    if net_up is not None and net_down is not None:
        lines.extend([
            "*━━━━━━━━━━━NET━━━━━━━━━━━*",
            f"📡 Net: ↑ {human_net_speed(net_up)} ↓ {human_net_speed(net_down)}",
        ])
    if gpu is not None:
        lines.extend([
            "*━━━━━━━━━━━GPU━━━━━━━━━━━*",
            f"🎮 GPU: {gpu:.1f}%",
        ])
        if vram_used is not None:
            lines.append(
                f"🗄️ VRAM: {vram_used:.0f} / {vram_total:.0f} MiB ({vram:.1f}%)"
            )
        if gpu_temp is not None:
            lines.append(f"🌡️ GPU Temp: {gpu_temp:.0f} °C")

    disks = json.loads(disks) if disks else []
    if disks:
        lines.append("*━━━━━━━━━━━DISKS━━━━━━━━━━*")
        for d in disks:
            mount = escape_markdown(d['mount'], version=1)
            line = (
                f"💾 {mount}: {disk_bar(d['percent'])} "
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
            [InlineKeyboardButton("🏎️ Speedtest", callback_data=f"speedtest:{secret}"),
             InlineKeyboardButton("📋 Диагностика", callback_data=f"diag:{secret}")],
            [InlineKeyboardButton("📶 Стабильность", callback_data=f"stability:{secret}")],
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
    db = load_db()
    entry = db["secrets"].get(secret)
    if not entry or not is_owner(entry, update.effective_user.id):
        return await update.message.reply_text("🚫 Нет доступа.")
    row = sql.execute(
        "SELECT * FROM metrics WHERE secret=? ORDER BY ts DESC LIMIT 1",
        (secret,),
    ).fetchone()

    if row:
        msg = await update.message.reply_text(
            format_status(row),
            parse_mode="Markdown",
            reply_markup=status_keyboard(secret),
        )
    else:
        msg = await update.message.reply_text("Нет данных от агента.")

    await send_or_queue(secret, "status")

    ctx.job_queue.run_repeating(
        callback=check_status_done,
        interval=2,
        data={
            "secret": secret,
            "chat_id": msg.chat_id,
            "msg_id": msg.message_id,
        },
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

async def cmd_plot(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if len(ctx.args) < 3:
        return await update.message.reply_text(
            "Синтаксис: /plot <ключ/имя> <метрики> <интервал> [предел] [единицы]"
        )

    key = ctx.args[0]
    metrics = [m for m in ctx.args[1].split(";") if m]
    rest = ctx.args[2:]

    time_tokens = []
    while rest and re.fullmatch(r"\d+[smhd]", rest[0], re.I):
        time_tokens.append(rest.pop(0))
    if not time_tokens:
        return await update.message.reply_text("Неверный интервал")
    try:
        seconds = parse_timespan(time_tokens)
    except ValueError:
        return await update.message.reply_text("Неверный интервал")

    top = None
    unit = None
    if rest:
        try:
            top = float(rest[0])
            rest = rest[1:]
        except ValueError:
            top = None
        if rest:
            unit = rest[0].rstrip(';')
            rest = rest[1:]
    if rest:
        return await update.message.reply_text("Слишком много аргументов")

    db = load_db()
    uid = update.effective_user.id
    secret = None
    entry = db["secrets"].get(key)
    if entry and is_owner(entry, uid):
        secret = key
    else:
        for s, e in db["secrets"].items():
            if is_owner(e, uid) and e.get("nickname") == key:
                secret = s
                break
    if not secret:
        return await update.message.reply_text("Ключ не найден.")

    try:
        buf = await run_plot(plot_custom, secret, metrics, seconds, top, unit)
    except ValueError as exc:
        return await update.message.reply_text(f"❌ {exc}")
    if not buf:
        return await update.message.reply_text("Данных за этот период нет.")

    caption = f"{'/'.join([m.upper() for m in metrics])} за {timedelta(seconds=seconds)}"
    if seconds >= 86400:
        doc = InputFile(buf, filename="plot.png")
        await ctx.bot.send_document(chat_id=update.effective_chat.id, document=doc, caption=caption)
        buf.close()
    else:
        await ctx.bot.send_photo(chat_id=update.effective_chat.id, photo=buf, caption=caption)
        buf.close()


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

        orig = q.message.text or ""
        prefixes = ("💻", "⏳", "Нет данных", "⚠️")
        from_status = any(orig.startswith(p) for p in prefixes)

        if from_status:
            await q.edit_message_text(
                text=f"⏳ Обновляем...\n{orig}",
                reply_markup=q.message.reply_markup,
            )
        else:
            if row:
                await q.edit_message_text(
                    format_status(row),
                    parse_mode="Markdown",
                    reply_markup=status_keyboard(secret),
                )
            else:
                await q.edit_message_text("Нет данных от агента.")

        await send_or_queue(secret, "status")

        ctx.job_queue.run_repeating(
            callback=check_status_done,
            interval=2,
            data={
                "secret": secret,
                "chat_id": q.message.chat_id,
                "msg_id": q.message.message_id,
            },
        )
        return
    if action in {"reboot", "shutdown"}:
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            return await q.edit_message_text("🚫 Нет доступа.")
        await send_or_queue(secret, action)
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



    if action == "stability":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("50 ms", callback_data=f"stab_i:{secret}:50"),
                    InlineKeyboardButton("100 ms", callback_data=f"stab_i:{secret}:100"),
                    InlineKeyboardButton("200 ms", callback_data=f"stab_i:{secret}:200"),
                    InlineKeyboardButton("500 ms", callback_data=f"stab_i:{secret}:500"),
                    InlineKeyboardButton("1000 ms", callback_data=f"stab_i:{secret}:1000"),
                ],
                [InlineKeyboardButton("◀️ Назад", callback_data=f"status:{secret}")],
            ]
        )
        await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text="Интервал пакетов:",
            reply_markup=kb,
        )
        return

    if action == "stab_i":
        secret = parts[1]
        interval = int(parts[2])
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        kb = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "5 мин",
                        callback_data=f"stab_run:{secret}:{interval}:300",
                    ),
                    InlineKeyboardButton(
                        "15 мин",
                        callback_data=f"stab_run:{secret}:{interval}:900",
                    ),
                    InlineKeyboardButton(
                        "1 ч",
                        callback_data=f"stab_run:{secret}:{interval}:3600",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "6 ч",
                        callback_data=f"stab_run:{secret}:{interval}:21600",
                    ),
                    InlineKeyboardButton(
                        "12 ч",
                        callback_data=f"stab_run:{secret}:{interval}:43200",
                    ),
                    InlineKeyboardButton(
                        "1 д",
                        callback_data=f"stab_run:{secret}:{interval}:86400",
                    ),
                ],
                [InlineKeyboardButton("◀️ Назад", callback_data=f"stability:{secret}")],
            ]
        )
        await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text="Длительность теста:",
            reply_markup=kb,
        )
        return

    if action == "stab_run":
        secret = parts[1]
        interval = int(parts[2])
        duration = int(parts[3])
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        await send_or_queue(secret, f"stability {interval} {duration}")
        await q.answer()
        dur_text = (
            f"{duration // 86400} д"
            if duration >= 86400
            else f"{duration // 3600} ч"
            if duration >= 3600
            else f"{duration // 60} мин"
        )
        msg = await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text=f"⏳ Проверяем стабильность: {interval} мс, {dur_text}…",
        )
        ctx.job_queue.run_repeating(
            callback=check_stability_done,
            interval=30,
            data={
                "secret": secret,
                "chat_id": msg.chat_id,
                "msg_id": msg.message_id,
                "deadline": time.time() + duration + 60,
            },
        )
        return


    if action == "speedtest":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        await send_or_queue(secret, "speedtest")

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

    if action == "diag":
        secret = parts[1]
        entry = db["secrets"].get(secret)
        if not entry or not is_owner(entry, q.from_user.id):
            await q.answer("🚫 Нет доступа.", show_alert=True)
            return
        await send_or_queue(secret, "diag")

        await q.answer()
        msg = await ctx.bot.send_message(
            chat_id=q.message.chat_id,
            text="⏳ Собираем диагностику…",
        )

        ctx.job_queue.run_repeating(
            callback=check_diag_done,
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
            buf = await run_plot(plot_all_metrics, secret, seconds)
            caption = f"Все метрики за {timedelta(seconds=seconds)}"
        elif metric == "net":
            buf = await run_plot(plot_net, secret, seconds)
            caption = f"NET за {timedelta(seconds=seconds)}"
        else:
            buf = await run_plot(plot_metric, secret, metric, seconds)
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
            buf.close()
        else:
            await ctx.bot.send_photo(
                chat_id=q.message.chat_id,
                photo=buf,
                caption=caption,
            )
            buf.close()
        return

# ────────────────────────── FastAPI for agents ─────────────────────────────
app = FastAPI()

class StabilityPayload(BaseModel):
    start_ts: float | None = None
    interval_ms: int | None = None
    rtts: List[float | None] | None = None
    error: str | None = None


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
    top_procs: list[dict] | None = None
    oneshot: bool | None = None
    text: str | None = None
    diag: str | None = None
    diag_ok: bool | None = None
    stability: StabilityPayload | None = None


async def process_payload(secret: str, payload: PushPayload) -> None:
    """Обработать данные от агента."""
    db = load_db()
    if secret not in db["secrets"]:
        raise HTTPException(404)

    if payload.text:
        LATEST_TEXT[secret] = payload.text

    if payload.diag_ok is not None:
        LATEST_DIAG[secret] = payload.diag if payload.diag_ok else None
        return

    if payload.stability is not None:
        LATEST_STAB[secret] = payload.stability.model_dump()
        return

    if payload.cpu is None or payload.ram is None:
        return

    if payload.oneshot:
        data = payload.model_dump()
        data.pop("oneshot", None)
        data["disks"] = json.dumps(data.get("disks") or [])
        data["top_procs"] = json.dumps(data.get("top_procs") or [])
        data["ts"] = int(time.time())
        LATEST_STATUS[secret] = data
        await maybe_send_alerts(secret, data)
        return

    record_metric(secret, payload.model_dump())
    await maybe_send_alerts(secret, payload.model_dump())





@app.websocket("/ws/{secret}")
async def ws_endpoint(ws: WebSocket, secret: str):
    db = load_db()
    if secret not in db["secrets"]:
        await ws.close(code=1008)
        return
    await ws.accept()
    ACTIVE_WS[secret] = ws
    try:
        while True:
            data = await ws.receive_json()
            await process_payload(secret, PushPayload(**data))
            db = load_db()
            cmds = db["secrets"][secret].get("pending", [])
            db["secrets"][secret]["pending"] = []
            save_db(db)
            await ws.send_json({"commands": cmds})
    except WebSocketDisconnect:
        pass
    finally:
        ACTIVE_WS.pop(secret, None)

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
    threading.Thread(target=start_udp_echo, daemon=True).start()
    log.info("🌐 FastAPI on port %s", API_PORT)
    log.info("📡 UDP echo on port %s", UDP_TEST_PORT)

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
    TG_APP.add_handler(CommandHandler("plot", cmd_plot))
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
