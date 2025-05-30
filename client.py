from __future__ import annotations

"""pc_agent_tls.py – Lightweight PC agent."""


import logging, threading
import os
import platform
try:
    import wmi
except ImportError:
    wmi = None
try:
    import speedtest
except ImportError:
    speedtest = None
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
import shutil, subprocess, re
import psutil
import requests
from requests import Session
from requests.exceptions import SSLError, ConnectionError
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

log = logging.getLogger("pc-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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

# ──────────────────── fingerprint pinning ────────────────────────
import hashlib, json, ssl, pathlib
FP_FILE = pathlib.Path.home() / ".bot_fingerprint.json"

def _cert_fp(cert_bin: bytes) -> str:
    return hashlib.sha256(cert_bin).hexdigest()

def _load_fp() -> str | None:
    if FP_FILE.exists():
        return json.loads(FP_FILE.read_text()).get("fp")

def _save_fp(fp: str):
    FP_FILE.write_text(json.dumps({"fp": fp}))

def _ctx_with_pinning(pinned: str | None) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    if pinned:
        # проверяем, что отпечаток совпадает
        def _verify_cb(conn, cert, errno, depth, ok):
            return ok and _cert_fp(cert.as_binary()) == pinned
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.set_verify(ssl.CERT_REQUIRED, _verify_cb)
    else:
        # первый запуск: временно без проверки
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx

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

PORT = int(os.getenv("AGENT_PORT", "8000"))

VERIFY_ENV = os.getenv("AGENT_VERIFY_SSL", "1").lower()
if VERIFY_ENV == "0":
    VERIFY_SSL: Optional[str | bool] = False
elif VERIFY_ENV == "force":
    VERIFY_SSL = True
else:
    VERIFY_SSL = True

CA_FILE = os.getenv("AGENT_CA_FILE")
if CA_FILE:
    VERIFY_SSL = CA_FILE  # requests accepts str path

SCHEME = "https"
SERVER = f"{SCHEME}://{SERVER_IP}:{PORT}"
INTERVAL = int(os.getenv("AGENT_INTERVAL", "5"))

log.info("Config → server %s verify=%s interval %ss", SERVER, VERIFY_SSL, INTERVAL)

# ────────────────────────── metric helpers ────────────────────────────────

UNIT_NAMES = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]

def human_bytes(num: float) -> str:
    for unit in UNIT_NAMES:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} EiB"


def disk_bar(p: float, length=10) -> str:
    filled = int(round(p * length / 100))
    return "█" * filled + "░" * (length - filled)


def gather_disks() -> List[str]:
    EXCL_FSTYPES        = {"tmpfs", "devtmpfs", "squashfs", "overlay", "aufs"}
    EXCL_DEV_PREFIXES   = ("/dev/loop",)                     # snap-loop’ы и пр.
    EXCL_MOUNT_PREFIXES = ("/snap", "/var/lib/docker", "/var/snap", "/boot")
    MIN_SIZE_BYTES      = 1 << 30                           # 1 ГиБ

    lines, seen = [], set()
    lines.append("*━━━━━━━━━━━DISKS━━━━━━━━━━*")

    for part in psutil.disk_partitions(all=False):
        if (part.mountpoint in seen
            or part.fstype.lower()            in EXCL_FSTYPES
            or part.device.startswith(EXCL_DEV_PREFIXES)
            or any(part.mountpoint.startswith(p) for p in EXCL_MOUNT_PREFIXES)):
            continue
        seen.add(part.mountpoint)

        try:
            u = psutil.disk_usage(part.mountpoint)
        except PermissionError:
            continue

        if u.total < MIN_SIZE_BYTES:
            continue

        disk_string = (
            f"💾 {part.mountpoint}: {disk_bar(u.percent)} "
            f"{u.percent:.0f}% ({human_bytes(u.used)} / {human_bytes(u.total)})"
        )
        if u.percent >= 90:
            disk_string += "❗"
        lines.append(disk_string)

    return lines
def gather_gpu() -> tuple[str, str, str, str] | None:
    # ── 1) pynvml ─────────────────────────────
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu          # %
        mem  = pynvml.nvmlDeviceGetMemoryInfo(h)                    # bytes
        temp = pynvml.nvmlDeviceGetTemperature(
            h, pynvml.NVML_TEMPERATURE_GPU)
        return (
            "*━━━━━━━━━━━GPU━━━━━━━━━━━*",
            f"🎮 GPU: {util:.1f}%",
            f"🗄️ VRAM: {mem.used/2**20:.0f} / {mem.total/2**20:.0f} MiB "
            f"({mem.used/mem.total*100:.1f}%)",
            f"🌡️ GPU Temp: {temp} °C"
        )
    except Exception:
        pass  # переходим к следующему способу

    # ── 2)
    if shutil.which("nvidia-smi"):
        try:
            util, used, total, temp = map(float, re.split(r",\s*",
                                                          subprocess.check_output(
                                                              ["nvidia-smi",
                                                               "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                                               "--format=csv,noheader,nounits"],
                                                              text=True, timeout=2
                                                          ).strip()
                                                          ))
            return (
                "*━━━━━━━━━━━GPU━━━━━━━━━━━*",
                f"🎮 GPU: {util:.1f}%",
                f"🗄️ VRAM: {used:.0f} / {total:.0f} MiB "
                f"({used/total*100:.1f}%)",
                f"🌡️ GPU Temp: {temp} °C"
            )
        except Exception:
            pass

    # ── 3) GPUtil
    try:
        import GPUtil
        gpu = GPUtil.getGPUs()[0]
        util = gpu.load * 100                           # 0-1 → %
        used = gpu.memoryUsed
        total = gpu.memoryTotal
        temp = gpu.temperature
        return (
            "*━━━━━━━━━━━GPU━━━━━━━━━━━*",
            f"🎮 GPU: {util:.1f}%",
            f"🗄️ VRAM: {used:.0f} / {total:.0f} MiB "
            f"({used/total*100:.1f}%)",
            f"🌡️ GPU Temp: {temp} °C"
        )
    except Exception:
        return None     # не удалось
def get_cpu_temp() -> str | None:
    # ── 1) стандартный psutil ─────────────────────────────
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name in ("coretemp", "k10temp", "cpu_thermal"):
                if name in temps and temps[name]:
                    return f"{temps[name][0].current:.1f} °C"
    except Exception:
        pass

    # ── 2) Windows: Open/Libre Hardware Monitor через WMI ─
    if platform.system() == "Windows" and wmi:
        for namespace in ("root\\OpenHardwareMonitor",
                          "root\\LibreHardwareMonitor"):
            try:
                c = wmi.WMI(namespace=namespace)
                sensors = c.Sensor()  # все датчики
                for s in sensors:
                    if s.SensorType == u"Temperature" and "CPU" in s.Name:
                        return f"{s.Value:.1f} °C"
            except Exception:
                continue

    return None
def gather_status() -> str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temp_val = get_cpu_temp()
    temp = temp_val if temp_val is not None else "N/A"

    uptime = time.time() - psutil.boot_time()
    lines = [
        "💻 *PC stats*",
        f"⏳ Uptime: {timedelta(seconds=int(uptime))}",
        "*━━━━━━━━━━━CPU━━━━━━━━━━━*",
        f"🖥️ CPU: {cpu:.1f}%",
        f"🌡️ CPU Temp: {temp}",
        "*━━━━━━━━━━━RAM━━━━━━━━━━━*",
        f"🧠 RAM: {human_bytes(mem.used)} / {human_bytes(mem.total)} ({mem.percent:.1f}%)",
        f"🧠 SWAP: {human_bytes(swap.used)} / {human_bytes(swap.total)} ({swap.percent:.1f}%)",

    ]
    gpu_lines = gather_gpu()
    disk_lines = gather_disks()
    if gpu_lines:
        lines.extend(gpu_lines)
    lines.extend(disk_lines)
    return "\n".join(lines)
def run_speedtest() -> tuple[float | None, float | None, float | None]:

    try:
        if speedtest:
            st = speedtest.Speedtest(secure=True)
            st.get_best_server()
            dl = st.download() / 1e6
            ul = st.upload()   / 1e6
            return dl, ul, st.results.ping

        import shutil, subprocess, json
        if shutil.which("speedtest"):
            out = subprocess.check_output(
                ["speedtest", "--format=json"], text=True, timeout=120
            )
            data = json.loads(out)
            dl = data["download"]["bandwidth"] * 8 / 1e6
            ul = data["upload"]["bandwidth"]  * 8 / 1e6
            ping = data["ping"]["latency"]
            return dl, ul, ping
    except Exception as exc:
        log.error("speedtest failed: %s", exc)
    return None, None, None
# ---------- async speedtest helper ----------
speedtest_running = False      # флаг «тест уже идёт»

def _speedtest_job():
    global speedtest_running
    push_status("⏳ Тестируем скорость…")
    dl, ul, ping = run_speedtest()
    if dl is not None:
        push_status(f"💨 Speedtest:\n"
                    f"↓ {dl:.1f} Mbit/s  ↑ {ul:.1f} Mbit/s  Ping {ping:.0f} ms")
    else:
        push_status("⚠️  Speedtest не удался.")
    speedtest_running = False
# ────── network layer: TLS TOFU + fingerprint pinning ────────────
import ssl, socket, json, hashlib, pathlib, logging, requests
from urllib.parse import urlparse
from requests.exceptions import SSLError

log      = logging.getLogger(__name__)
session  = requests.Session()
FP_FILE  = pathlib.Path.home() / ".bot_fingerprint.json"

def _fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()

def _load_fp() -> str | None:
    return json.loads(FP_FILE.read_text())["fp"] if FP_FILE.exists() else None

def _save_fp(fp: str) -> None:
    FP_FILE.write_text(json.dumps({"fp": fp}))

def _fetch_cert_der(parsed) -> bytes:
    host, port = parsed.hostname, parsed.port or 443
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
        s.settimeout(5)
        s.connect((host, port))
        return s.getpeercert(binary_form=True)

def _request(method: str, url: str, **kwargs):
    pinned = _load_fp()
    try:
        resp = session.request(method, url, verify=False,
                               timeout=10, stream=True, **kwargs)
    except SSLError as exc:
        log.error("TLS-ошибка: %s", exc)
        raise

    cert_der = None
    try:
        cert_der = resp.raw.connection.sock.getpeercert(binary_form=True)
    except AttributeError:
        cert_der = _fetch_cert_der(urlparse(url))

    current_fp = _fingerprint(cert_der)

    if pinned is None:
        _save_fp(current_fp)
        log.info("🎉  Cert saved, fp=%s…", current_fp[:16])
    elif pinned != current_fp:
        raise RuntimeError(
            f"TLS fingerprint mismatch! old={pinned[:16]}… new={current_fp[:16]}…"
        )

    return resp

def push_status(txt: str):
    try:
        r = _request("POST", f"{SERVER}/api/push/{SECRET}", json={"text": txt})
        r.raise_for_status()
    except Exception as e:
        log.error("push error: %s", e)


def pull_cmds() -> List[str]:
    try:
        r = _request("GET", f"{SERVER}/api/pull/{SECRET}")
        r.raise_for_status(); return r.json().get("commands", [])
    except Exception as e:
        log.error("pull error: %s", e); return []

# ────────────────────────── actions ───────────────────────────────────────

def do_reboot():
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["shutdown", "/r", "/t", "0"], shell=False)
        else:
            subprocess.Popen(["sudo", "reboot"], shell=False)
    except Exception as e:
        log.error("reboot failed: %s", e)


def do_shutdown():
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["shutdown", "/s", "/t", "0"], shell=False)
        else:
            subprocess.Popen(["sudo", "shutdown", "-h", "now"], shell=False)
    except Exception as e:
        log.error("shutdown failed: %s", e)

# ────────────────────────── main loop ─────────────────────────────────────

log.info("Agent started → %s", SERVER)
while True:
    push_status(gather_status())
    for c in pull_cmds():
        if c == "reboot":
            log.info("cmd reboot"); push_status("⚡️ Rebooting…"); do_reboot()
        elif c == "shutdown":
            log.info("cmd shutdown"); push_status("💤 Shutting down…"); do_shutdown()
        elif c == "speedtest":
            # запускаем тест в отдельном потоке, чтобы не блокировать основной цикл
            if not speedtest_running:
                log.info("cmd speedtest (async)")
                speedtest_running = True
                threading.Thread(target=_speedtest_job, daemon=True).start()
            else:
                push_status("🚧 Speedtest уже выполняется, дождитесь окончания.")
    time.sleep(INTERVAL)