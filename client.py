"""pc_agent.py – Lightweight PC agent sending stats to remote bot server.

v2025‑05‑17‑persist – теперь секрет (`AGENT_SECRET`) запоминается в `.env`.
Алгоритм:
  1. Пытаемся взять `AGENT_SECRET` из переменных окружения.
  2. Если нет – читаем из файла `.env` в текущей папке.
  3. Если всё ещё пусто – спрашиваем в консоли и тут же записываем в `.env`.
     При следующем запуске запоминать уже не придётся.

Остальной функционал (метрики, reboot/shutdown) не изменился.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List

import psutil
import requests

# ────────────────────────── CONFIG ─────────────────────────────────────────
ENV_FILE = Path(".env")


def _load_dotenv() -> None:
    """Populate os.environ from .env if variables not already set."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

SECRET = os.getenv("AGENT_SECRET")
if not SECRET:
    SECRET = input("Enter AGENT_SECRET: ").strip()
    if not SECRET:
        print("❌ AGENT_SECRET is required. Exiting.")
        sys.exit(1)
    # append to .env
    with ENV_FILE.open("a", encoding="utf-8") as f:
        f.write(f"AGENT_SECRET={SECRET}\n")
        print("🔏 AGENT_SECRET saved to .env")

SERVER = os.getenv("AGENT_SERVER", "http://localhost:8000")
INTERVAL = int(os.getenv("AGENT_INTERVAL", "30"))

# ────────────────────────── HELPERS ────────────────────────────────────────

def human_bytes(num: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def disk_bar(percent: float, length: int = 10) -> str:
    filled = int(round(percent * length / 100))
    return "■" * filled + "□" * (length - filled)


def gather_disks() -> List[str]:
    lines: List[str] = []
    seen = set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen or part.fstype.lower() in {"tmpfs", "devtmpfs"}:
            continue
        seen.add(part.mountpoint)
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except PermissionError:
            continue
        if usage.total == 0:
            continue
        bar = disk_bar(usage.percent)
        lines.append(
            f"💾 {part.mountpoint}: {bar} {usage.percent:.0f}% "
            f"({human_bytes(usage.used)} / {human_bytes(usage.total)})"
        )
    return lines


def gather_status() -> str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    uptime = time.time() - psutil.boot_time()
    temp = (
        f"{psutil.sensors_temperatures()['coretemp'][0].current:.1f} °C"
        if hasattr(psutil, "sensors_temperatures") and psutil.sensors_temperatures()
        else "N/A"
    )

    lines: List[str] = [
        "💻 *PC stats*",
        f"🖥️ CPU: {cpu:.1f}%",
        f"🌡️ Temp: {temp}",
        f"🧠 RAM: {human_bytes(mem.used)} / {human_bytes(mem.total)} ({mem.percent:.1f}%)",
    ]
    lines.extend(gather_disks())
    lines.append(f"⏳ Uptime: {str(timedelta(seconds=int(uptime)))}")
    return "\n".join(lines)


def push_status(text: str) -> None:
    try:
        r = requests.post(f"{SERVER}/api/push/{SECRET}", json={"text": text}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("push error:", e)


def pull_commands() -> List[str]:
    try:
        r = requests.get(f"{SERVER}/api/pull/{SECRET}", timeout=10)
        r.raise_for_status()
        return r.json().get("commands", [])
    except Exception as e:
        print("pull error:", e)
        return []


# ────────────────────────── ACTIONS ───────────────────────────────────────

def do_reboot():
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["shutdown", "/r", "/t", "0"], shell=False)
        else:
            subprocess.Popen(["sudo", "reboot"], shell=False)
    except Exception as e:
        print("reboot failed:", e)


def do_shutdown():
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["shutdown", "/s", "/t", "0"], shell=False)
        else:
            subprocess.Popen(["sudo", "shutdown", "-h", "now"], shell=False)
    except Exception as e:
        print("shutdown failed:", e)


# ────────────────────────── MAIN LOOP ─────────────────────────────────────
print("Agent started. Server:", SERVER)
while True:
    push_status(gather_status())

    for cmd in pull_commands():
        if cmd == "reboot":
            print("Reboot command received")
            push_status("⚡️ Rebooting now…")
            do_reboot()
        elif cmd == "shutdown":
            print("Shutdown command received")
            push_status("💤 Shutting down now…")
            do_shutdown()
    time.sleep(INTERVAL)
