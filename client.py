"""pc_agent.py – Lightweight PC agent"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List

import psutil
import requests

ENV_FILE = Path(".env")

# ────────────────────────── load .env → os.environ ─────────────────────────
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ────────────────────────── prompt helpers ─────────────────────────────────
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

def prompt_ip() -> str:
    while True:
        ip = input("Enter SERVER IPv4 [127.0.0.1]: ").strip() or "127.0.0.1"
        if IP_RE.match(ip):
            return ip
        print("❌ Invalid IPv4, try again (e.g. 192.168.1.42)")

# ────────────────────────── CONFIG values ──────────────────────────────────
SECRET = os.getenv("AGENT_SECRET") or input("Enter AGENT_SECRET: ").strip()
if not SECRET:
    print("AGENT_SECRET required"); sys.exit(1)

if "AGENT_SECRET" not in os.environ:
    ENV_FILE.write_text((ENV_FILE.read_text() if ENV_FILE.exists() else "") + f"AGENT_SECRET={SECRET}\n")

SERVER_IP = os.getenv("AGENT_SERVER_IP")
if not SERVER_IP:
    SERVER_IP = prompt_ip()
    ENV_FILE.write_text((ENV_FILE.read_text() if ENV_FILE.exists() else "") + f"AGENT_SERVER_IP={SERVER_IP}\n")

SERVER = f"http://{SERVER_IP}:8000"
INTERVAL = int(os.getenv("AGENT_INTERVAL", "30"))

print(f"Config → server {SERVER} interval {INTERVAL}s")

# ────────────────────────── metric helpers ────────────────────────────────

def human_bytes(num: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PiB"


def disk_bar(p: float, length=10) -> str:
    filled = int(round(p * length / 100))
    return "■" * filled + "□" * (length - filled)


def gather_disks() -> List[str]:
    lines, seen = [], set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen or part.fstype.lower() in {"tmpfs", "devtmpfs"}:
            continue
        seen.add(part.mountpoint)
        try:
            u = psutil.disk_usage(part.mountpoint)
        except PermissionError:
            continue
        if u.total == 0:
            continue
        lines.append(
            f"💾 {part.mountpoint}: {disk_bar(u.percent)} {u.percent:.0f}% ({human_bytes(u.used)} / {human_bytes(u.total)})"
        )
    return lines


def gather_status() -> str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    uptime = time.time() - psutil.boot_time()
    temp = (
        f"{psutil.sensors_temperatures()['coretemp'][0].current:.1f} °C"
        if hasattr(psutil, "sensors_temperatures") and psutil.sensors_temperatures()
        else "N/A"
    )
    lines = [
        "💻 *PC stats*",
        f"🖥️ CPU: {cpu:.1f}%",
        f"🌡️ CPU Temp: {temp}",
        f"🧠 RAM: {human_bytes(mem.used)} / {human_bytes(mem.total)} ({mem.percent:.1f}%)",
        f"🧠 SWAP: {human_bytes(swap.used)} / {human_bytes(swap.total)} ({swap.percent:.1f}%)",
    ] + gather_disks() + [f"⏳ Uptime: {timedelta(seconds=int(uptime))}"]
    return "\n".join(lines)

# ────────────────────────── network I/O ───────────────────────────────────

def push_status(txt: str):
    try:
        r = requests.post(f"{SERVER}/api/push/{SECRET}", json={"text": txt}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("push error:", e)


def pull_cmds() -> List[str]:
    try:
        r = requests.get(f"{SERVER}/api/pull/{SECRET}", timeout=10)
        r.raise_for_status(); return r.json().get("commands", [])
    except Exception as e:
        print("pull error:", e); return []

# ────────────────────────── actions ───────────────────────────────────────

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

# ────────────────────────── main loop ─────────────────────────────────────
print("Agent started →", SERVER)
while True:
    push_status(gather_status())
    for c in pull_cmds():
        if c == "reboot":
            print("cmd reboot"); push_status("⚡️ Rebooting…"); do_reboot()
        elif c == "shutdown":
            print("cmd shutdown"); push_status("💤 Shutting down…"); do_shutdown()
    time.sleep(int(INTERVAL))
