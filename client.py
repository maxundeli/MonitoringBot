"""Lightweight agent that runs on a PC and reports to *remote_bot_server.py*.

* requirements: python >=3.8, psutil, requests
* configuration via env vars (or edit constants below):
    AGENT_SECRET   – secret obtained from /newkey
    AGENT_SERVER   – base URL of server (e.g. http://example.com:8000)
    AGENT_INTERVAL – seconds between status pushes (default 30)

Metrics sent every cycle:
🖥️ CPU usage   – numeric %
🌡️ Temperature – degrees Celsius
🧠 RAM usage   – human‑readable bytes & %
💾 Disks       – per‑mountpoint usage bar with % and bytes
⏳ Uptime      – human duration

Accepted commands from server:
  • reboot   – immediate reboot (admin rights required)
  • shutdown – immediate power‑off (admin rights required)

Disk usage bars are rendered with ten squares (■ = used, □ = free).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import timedelta
from typing import List

import psutil
import requests

# ---- config ---------------------------------------------------------------
SECRET = os.getenv("AGENT_SECRET", "DCLpF6dCtokePlWSPmNn")
SERVER = os.getenv("AGENT_SERVER", "http://localhost:8000")
INTERVAL = int(os.getenv("AGENT_INTERVAL", "30"))

if not SECRET:
    print("AGENT_SECRET env var missing")
    sys.exit(1)

# ---- helpers --------------------------------------------------------------

def human_bytes(num: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PiB"


def disk_bar(percent: float, length: int = 10) -> str:
    """Return a text bar of ■/□ squares representing percentage used."""
    filled = int(round(percent * length / 100))
    return "■" * filled + "□" * (length - filled)


def gather_disks() -> List[str]:
    """Collect per‑partition disk usage lines with emoji and bars."""
    lines: List[str] = []
    seen = set()
    for part in psutil.disk_partitions(all=False):
        # Skip duplicates & non‑physical mounts (e.g., /snap, virtiofs)
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
        line = (
            f"💾 {part.mountpoint}: {bar} {usage.percent:.0f}% "
            f"({human_bytes(usage.used)} / {human_bytes(usage.total)})"
        )
        lines.append(line)
    return lines


def gather_status() -> str:
    """Build a multi‑line Markdown status block for push()."""
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
        (
            f"🧠 RAM: {human_bytes(mem.used)} / {human_bytes(mem.total)} "
            f"({mem.percent:.1f}%)"
        ),
    ]

    lines.extend(gather_disks())
    lines.append(f"⏳ Uptime: {str(timedelta(seconds=int(uptime)))}")
    return "\n".join(lines)


def push_status(text: str) -> None:
    url = f"{SERVER}/api/push/{SECRET}"
    try:
        r = requests.post(url, json={"text": text}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("push error:", e)


def pull_commands() -> List[str]:
    url = f"{SERVER}/api/pull/{SECRET}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("commands", [])
    except Exception as e:
        print("pull error:", e)
        return []


# ---- actions --------------------------------------------------------------

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


# ---- main loop ------------------------------------------------------------
print("Agent started. Server:", SERVER)
while True:
    stats = gather_status()
    push_status(stats)

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
