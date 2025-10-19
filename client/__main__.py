from __future__ import annotations

"""pc_agent_tls.py – Lightweight PC agent."""


import logging, threading
import asyncio
import json
import os
import platform
import atexit
try:
    import wmi
except ImportError:
    wmi = None
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
import shutil
import subprocess
import socket
import math

import heapq
import psutil
from PIL import Image

if os.name == "nt":
    import pystray
else:
    pystray = None  # type: ignore
from client.worker import run_speedtest, run_diagnostics, submit, shutdown_executor

atexit.register(shutdown_executor)
atexit.register(lambda: TRAY_ICON and TRAY_ICON.stop())
# Кэш для вычисления CPU без задержки
PROC_CACHE: dict[int, tuple[float, float]] = {}
GPU_VENDOR: str | None = None
GPU_METRIC_FUNCS: list = []
NVML_INITED = False
NVML_HANDLE = None
CPU_CORES = psutil.cpu_count(logical=True) or psutil.cpu_count() or 1
import websockets
WS_LOOP: asyncio.AbstractEventLoop | None = None
WS_CONN: websockets.WebSocketClientProtocol | None = None
WS_MAIN_TASK: asyncio.Task | None = None
TRAY_ICON: object | None = None
WS_PENDING: list[dict] = []
WS_PENDING_LOCK = threading.Lock()
try:
    import pynvml
except Exception:
    pynvml = None

if os.name == "nt":
    import ctypes

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    SW_HIDE = 0
    SW_RESTORE = 9

    def _console_hwnd() -> int:
        return kernel32.GetConsoleWindow()

    def toggle_console() -> None:
        hwnd = _console_hwnd()
        if not hwnd:
            if kernel32.AllocConsole():
                sys.stdout = open("CONOUT$", "w", buffering=1)
                sys.stderr = open("CONOUT$", "w", buffering=1)
                sys.stdin = open("CONIN$", "r")
                hwnd = _console_hwnd()
        if hwnd:
            if user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, SW_HIDE)
            else:
                user32.ShowWindow(hwnd, SW_RESTORE)
                user32.SetForegroundWindow(hwnd)

    def _hide_console_on_minimize() -> None:
        hwnd = _console_hwnd()
        if not hwnd:
            return
        visible = True
        while True:
            time.sleep(0.5)
            if visible and user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_HIDE)
                visible = False
            elif not visible and user32.IsWindowVisible(hwnd):
                visible = True
            if not user32.IsWindow(hwnd):
                break
else:
    def toggle_console() -> None:
        pass


log = logging.getLogger("pc-agent")
_lvl = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _lvl, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ENV_FILE = Path(".env")
GPU_METHOD_KEY = "AGENT_GPU_METHOD"

# ────────────────────────── load .env → os.environ ─────────────────────────
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _update_env_file(key: str, value: str) -> None:
    """Persist the given key/value pair to .env, replacing existing value."""

    os.environ[key] = value
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text().splitlines()
    else:
        lines = []

    new_line = f"{key}={value}"
    replaced = False
    for idx, line in enumerate(lines):
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        if line.split("=", 1)[0].strip() == key:
            lines[idx] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)

    if lines:
        ENV_FILE.write_text("\n".join(lines) + "\n")
    else:
        ENV_FILE.write_text("")


GPU_METHOD: str | None = os.getenv(GPU_METHOD_KEY)


def _set_gpu_method(value: str) -> None:
    """Store the detected GPU metrics collection method."""

    global GPU_METHOD
    if GPU_METHOD == value:
        return
    GPU_METHOD = value
    _update_env_file(GPU_METHOD_KEY, value)


def _as_valid_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _plausible_gpu_metrics(data: dict) -> bool:
    """Heuristics to guard against bogus GPU telemetry results."""

    total = _as_valid_float(data.get("vram_total"))
    used = _as_valid_float(data.get("vram_used"))
    util = _as_valid_float(data.get("gpu"))

    if total is not None:
        if total <= 0 or total < 128 or total > 262144:  # 128 MB .. 256 GB
            return False
        if used is not None and not (0 <= used <= total * 1.2):
            return False
    elif used is not None and used < 0:
        return False

    if util is not None and not (0 <= util <= 110):
        return False

    return True


def _record_gpu_metrics(tag: str, data: dict) -> dict | None:
    if _plausible_gpu_metrics(data):
        _set_gpu_method(tag)
        return data
    log.warning("Discarding GPU metrics from %s due to implausible values: %s", tag, data)
    return None

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
UDP_PORT = int(os.getenv("AGENT_UDP_PORT", "9999"))

VERIFY_ENV = os.getenv("AGENT_VERIFY_SSL", "1").lower()
if VERIFY_ENV == "0":
    VERIFY_SSL: Optional[str | bool] = False
elif VERIFY_ENV == "force":
    VERIFY_SSL = True
else:
    VERIFY_SSL = True

CA_FILE = os.getenv("AGENT_CA_FILE")
if CA_FILE:
    VERIFY_SSL = CA_FILE

SCHEME = "https"
SERVER = f"{SCHEME}://{SERVER_IP}:{PORT}"
INTERVAL = int(os.getenv("AGENT_INTERVAL", "5"))
RECONNECT_DELAY = int(os.getenv("AGENT_RECONNECT_DELAY", "5"))

ICON_FILE = os.getenv("AGENT_ICON_FILE")
if not ICON_FILE:
    candidate = Path(__file__).with_name("icon.png")
    if candidate.exists():
        ICON_FILE = str(candidate)
if ICON_FILE and not Path(ICON_FILE).exists():
    log.warning("Tray icon not found: %s", ICON_FILE)
    ICON_FILE = None


def _tray_exit(icon, item) -> None:
    icon.visible = False
    icon.stop()
    if WS_LOOP:
        def _cancel_main() -> None:
            if WS_MAIN_TASK and not WS_MAIN_TASK.done():
                WS_MAIN_TASK.cancel()
        WS_LOOP.call_soon_threadsafe(_cancel_main)
    else:
        os._exit(0)


def start_tray_icon() -> object | None:
    """Запустить иконку в системном трее."""
    if os.name != "nt" or pystray is None:
        return None
    image = None
    try:
        if ICON_FILE:
            image = Image.open(ICON_FILE)
    except Exception as exc:
        log.error("Failed to load tray icon %s: %s", ICON_FILE, exc)
    if image is None:
        image = Image.new("RGB", (64, 64), color="blue")
    menu = pystray.Menu(
        pystray.MenuItem(
            "Показать/скрыть консоль",
            lambda icon, item: toggle_console(),
            default=True,
        ),
        pystray.MenuItem("Выход", _tray_exit),
    )
    icon = pystray.Icon("MonitoringBot", image, "MonitoringBot", menu)
    threading.Thread(target=icon.run, daemon=True).start()
    return icon

log.info(
    "Config → server %s verify=%s interval %ss reconnect %ss",
    SERVER,
    VERIFY_SSL,
    INTERVAL,
    RECONNECT_DELAY,
)





def gather_disks_metrics() -> List[dict]:
    EXCL_FSTYPES        = {"tmpfs", "devtmpfs", "squashfs", "overlay", "aufs"}
    EXCL_DEV_PREFIXES   = ("/dev/loop",)                     # snap-loop’ы и пр.
    EXCL_MOUNT_PREFIXES = ("/snap", "/var/lib/docker", "/var/snap", "/boot")
    MIN_SIZE_BYTES      = 1 << 30                           # 1 ГиБ

    res, seen = [], set()

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
        res.append({
            "mount": part.mountpoint,
            "percent": u.percent,
            "used": u.used,
            "total": u.total,
        })

    return res


PROC_SAMPLE_DELAY = 0.2


def _refresh_proc_cache() -> None:
    """Обновить кэш времени CPU для процессов без расчёта процентов."""

    now = time.time()
    alive: set[int] = set()
    for p in psutil.process_iter(["pid"]):
        try:
            cpu_time = sum(p.cpu_times()[:2])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        PROC_CACHE[p.pid] = (cpu_time, now)
        alive.add(p.pid)

    # удалить процессы, которых больше нет
    for pid in list(PROC_CACHE):
        if pid not in alive:
            PROC_CACHE.pop(pid, None)


def gather_top_processes(
    count: int = 5,
    *,
    instant: bool = False,
    sample_delay: float = PROC_SAMPLE_DELAY,
) -> List[dict]:
    """Вернуть топ процессов по загрузке CPU с учётом RAM.

    Процессы с одинаковым именем объединяются (суммируются их CPU и RAM).

    Если ``instant`` установлен, собирается дополнительный моментальный срез
    CPU с небольшой задержкой, что позволяет показать актуальную картину,
    не дожидаясь следующего вызова функции.
    """

    if instant:
        _refresh_proc_cache()
        if sample_delay > 0:
            time.sleep(sample_delay)

    now = time.time()
    aggregated: dict[str, dict[str, object]] = {}
    alive: set[int] = set()

    for p in psutil.process_iter(["pid", "name"]):
        try:
            name_raw = p.info.get("name") or str(p.pid)
            if name_raw.lower() == "system idle process":
                continue

            cpu_time = sum(p.cpu_times()[:2])
            prev = PROC_CACHE.get(p.pid)
            cpu = 0.0
            if prev:
                dt = now - prev[1]
                if dt > 0:
                    cpu = (cpu_time - prev[0]) / dt * 100
            PROC_CACHE[p.pid] = (cpu_time, now)
            alive.add(p.pid)
            cpu /= CPU_CORES

            key = name_raw.lower()
            agg = aggregated.setdefault(
                key,
                {"name": name_raw, "cpu": 0.0, "ram": 0, "count": 0, "pids": []},
            )
            agg["cpu"] += cpu
            agg["count"] += 1
            agg["pids"].append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for pid in list(PROC_CACHE):
        if pid not in alive:
            PROC_CACHE.pop(pid, None)

    top_entries = heapq.nlargest(count, aggregated.values(), key=lambda x: x["cpu"])

    for entry in top_entries:
        total_ram = 0
        for pid in entry["pids"]:
            try:
                total_ram += psutil.Process(pid).memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        entry["ram"] = total_ram

    res = []
    for data in top_entries:
        name = data["name"]
        if data["count"] > 1:
            name = f"{name} ({data['count']})"
        res.append({"name": name, "cpu": data["cpu"], "ram": data["ram"]})

    return res

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
def _nvidia_gpu_metrics() -> dict | None:
    """Try reading metrics using NVIDIA-specific tools."""

    preferred = (GPU_METHOD or "").lower()
    if preferred == "none":
        return None

    order: list[str]
    if preferred in {"nvidia:pynvml", "nvidia:cli", "nvidia:gputil"}:
        order = [preferred.split(":", 1)[1]]
    else:
        order = ["pynvml", "cli", "gputil"]

    for method in order:
        if method == "pynvml":
            if not (pynvml and NVML_INITED):
                continue
            try:
                h = NVML_HANDLE or pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                data = {
                    "gpu": util,
                    "vram_used": mem.used / 2 ** 20,
                    "vram_total": mem.total / 2 ** 20,
                    "vram": mem.used / mem.total * 100 if mem.total else None,
                    "gpu_temp": float(temp),
                }
                result = _record_gpu_metrics("nvidia:pynvml", data)
                if result:
                    return result
            except Exception:
                continue

        if method == "cli":
            if not shutil.which("nvidia-smi"):
                continue
            try:
                util, used, total, temp = map(
                    float,
                    re.split(
                        r",\s*",
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=2,
                        ).strip(),
                    ),
                )
                data = {
                    "gpu": util,
                    "vram_used": used,
                    "vram_total": total,
                    "vram": used / total * 100 if total else None,
                    "gpu_temp": temp,
                }
                result = _record_gpu_metrics("nvidia:cli", data)
                if result:
                    return result
            except Exception:
                continue

        if method == "gputil":
            try:
                import GPUtil

                gpu = GPUtil.getGPUs()[0]
                util = gpu.load * 100
                used = gpu.memoryUsed
                total = gpu.memoryTotal
                temp = gpu.temperature
                data = {
                    "gpu": util,
                    "vram_used": used,
                    "vram_total": total,
                    "vram": used / total * 100 if total else None,
                    "gpu_temp": temp,
                }
                result = _record_gpu_metrics("nvidia:gputil", data)
                if result:
                    return result
            except Exception:
                continue

    return None


def _windows_wmi_amd_metrics() -> dict | None:
    """Fallback metrics via Windows WMI performance counters."""
    if platform.system() != "Windows" or not wmi:
        return None
    try:
        c = wmi.WMI(namespace="root\\CIMV2")
        mems = c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUMemory()
        engines = c.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine()
        used = total = util = None
        if mems:
            used = float(mems[0].DedicatedUsage)
            total = float(mems[0].DedicatedLimit)
        if engines:
            vals = [int(e.UtilizationPercentage) for e in engines if "engtype_3d" in e.Name.lower()]
            if vals:
                util = sum(vals) / len(vals)
        data = {}
        if util is not None:
            data["gpu"] = util
        if used is not None:
            data["vram_used"] = used
        if total is not None:
            data["vram_total"] = total
            if used is not None:
                data["vram"] = used / total * 100 if total else None
        return data or None
    except Exception:
        return None


def _amd_gpu_metrics() -> dict | None:
    """Try reading metrics using AMD-specific tools."""

    preferred = (GPU_METHOD or "").lower()
    if preferred == "none":
        return None

    order: list[str]
    if preferred in {"amd:amdsmi", "amd:cli", "amd:wmi", "amd:adlxpy", "amd:pyadl"}:
        order = [preferred.split(":", 1)[1]]
    else:
        order = ["amdsmi", "cli", "wmi", "adlxpy", "pyadl"]

    for method in order:
        if method == "amdsmi":
            try:
                import amdsmi

                amdsmi.amdsmi_init()
                try:
                    handles = amdsmi.amdsmi_get_processor_handles()
                    if handles:
                        h = handles[0]
                        util = amdsmi.amdsmi_get_gpu_activity(h)["gfx_activity"]
                        vram = amdsmi.amdsmi_get_gpu_vram_usage(h)
                        used = vram["vram_used"] / 2 ** 20
                        total = vram["vram_total"] / 2 ** 20
                        temp = (
                            amdsmi.amdsmi_get_temp_metric(
                                h,
                                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                                amdsmi.AmdSmiTemperatureType.GPU_EDGE,
                            )["temperature"]
                            / 1000
                        )
                        data = {
                            "gpu": util,
                            "vram_used": used,
                            "vram_total": total,
                            "vram": used / total * 100 if total else None,
                            "gpu_temp": temp,
                        }
                        result = _record_gpu_metrics("amd:amdsmi", data)
                        if result:
                            return result
                finally:
                    try:
                        amdsmi.amdsmi_shut_down()
                    except Exception:
                        pass
            except Exception:
                continue

        if method == "cli":
            if not shutil.which("amd-smi"):
                continue
            try:
                out = subprocess.check_output(
                    ["amd-smi", "metric", "--json", "--gpu", "0"],
                    text=True,
                    timeout=2,
                )
                import json

                data = json.loads(out)["metric"][0]
                util = data["gfx_activity"]
                used = data["vram_usage"]["used_vram_bytes"] / 2 ** 20
                total = data["vram_usage"]["total_vram_bytes"] / 2 ** 20
                temp = data["temperature"]["edge_current_temp"] / 1000
                data = {
                    "gpu": util,
                    "vram_used": used,
                    "vram_total": total,
                    "vram": used / total * 100 if total else None,
                    "gpu_temp": temp,
                }
                result = _record_gpu_metrics("amd:cli", data)
                if result:
                    return result
            except Exception:
                continue

        if method == "wmi" and platform.system() == "Windows":
            data = _windows_wmi_amd_metrics()
            if data:
                result = _record_gpu_metrics("amd:wmi", data)
                if result:
                    return result

        if method == "adlxpy" and platform.system() == "Windows":
            try:
                import adlxpy

                helper = adlxpy.ADLXHelper()
                if helper.initialize():
                    try:
                        system = helper.get_system()
                        gpu = system.get_gpus().at(0)
                        perf = system.get_performance_monitoring_services()
                        metrics = perf.get_gpu_metrics(gpu)

                        util = metrics.gpu_utilization()
                        vram = metrics.vram_usage()
                        used = vram.vram_used() / 2 ** 20
                        total = vram.vram_total() / 2 ** 20
                        temp = metrics.gpu_temperatures().edge_current()

                        data = {
                            "gpu": util,
                            "vram_used": used,
                            "vram_total": total,
                            "vram": used / total * 100 if total else None,
                            "gpu_temp": temp,
                        }
                        result = _record_gpu_metrics("amd:adlxpy", data)
                        if result:
                            return result
                    finally:
                        helper.terminate()
            except Exception:
                continue

        if method == "pyadl" and platform.system() == "Windows":
            try:
                from pyadl import ADLManager

                devs = ADLManager.getInstance().getDevices()
                if devs:
                    dev = devs[0]
                    util = dev.getCurrentUsage()
                    temp = dev.getCurrentTemperature()
                    data = {"gpu": util, "gpu_temp": temp}
                    result = _record_gpu_metrics("amd:pyadl", data)
                    if result:
                        return result
            except Exception:
                continue

    return None


def detect_gpu_vendor() -> str | None:
    """Return 'nvidia', 'amd' or None if unknown."""
    if platform.system() == "Windows":
        if wmi:
            try:
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    name = (gpu.Name or "").lower()
                    vendor = (gpu.AdapterCompatibility or "").lower()
                    if "nvidia" in name or "nvidia" in vendor:
                        return "nvidia"
                    if (
                        "amd" in name
                        or "radeon" in name
                        or "advanced micro devices" in vendor
                    ):
                        return "amd"
            except Exception:
                pass
        try:
            kwargs = {}
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            out = subprocess.check_output(
                ["wmic", "path", "Win32_VideoController", "get", "Name"],
                text=True,
                timeout=2,
                **kwargs,
            ).lower()
            if "nvidia" in out:
                return "nvidia"
            if "amd" in out or "radeon" in out:
                return "amd"
        except Exception:
            pass
    else:
        try:
            if shutil.which("lspci"):
                out = subprocess.check_output(["lspci", "-nn"], text=True)
                for line in out.splitlines():
                    if " VGA " in line or "3d controller" in line.lower():
                        ll = line.lower()
                        if "nvidia" in ll:
                            return "nvidia"
                        if "amd" in ll or "radeon" in ll or "advanced micro devices" in ll:
                            return "amd"
        except Exception:
            pass
    return None


def init_gpu_metrics() -> None:
    """Определить производителя GPU и рабочие функции чтения метрик."""
    global GPU_VENDOR, GPU_METRIC_FUNCS, NVML_INITED, NVML_HANDLE

    method_hint = (GPU_METHOD or "").lower()

    if method_hint == "none":
        GPU_VENDOR = None
        GPU_METRIC_FUNCS = []
        return

    if method_hint.startswith("nvidia:"):
        GPU_VENDOR = "nvidia"
        candidates = [_nvidia_gpu_metrics]
    elif method_hint.startswith("amd:"):
        GPU_VENDOR = "amd"
        candidates = [_amd_gpu_metrics]
    else:
        GPU_VENDOR = detect_gpu_vendor()
        if GPU_VENDOR == "nvidia":
            candidates = [_nvidia_gpu_metrics]
        elif GPU_VENDOR == "amd":
            candidates = [_amd_gpu_metrics]
        else:
            candidates = [_nvidia_gpu_metrics, _amd_gpu_metrics]

    GPU_METRIC_FUNCS = []

    if GPU_VENDOR == "nvidia" and pynvml:
        try:
            pynvml.nvmlInit()
            NVML_INITED = True
            NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            atexit.register(pynvml.nvmlShutdown)
        except Exception:
            NVML_INITED = False
            NVML_HANDLE = None

    for fn in candidates:
        try:
            data = fn()
            if data:
                GPU_METRIC_FUNCS.append(fn)
        except Exception:
            continue

    if not GPU_METRIC_FUNCS:
        if not method_hint:
            _set_gpu_method("none")
            GPU_METRIC_FUNCS = []
        else:
            GPU_METRIC_FUNCS = candidates


def gather_gpu_metrics() -> dict | None:
    for fn in GPU_METRIC_FUNCS:
        try:
            data = fn()
            if data:
                return data
        except Exception:
            continue
    return None

# ──────────────────────── network usage ─────────────────────────-
NET_LAST = None

# список подстрок в названиях сетевых интерфейсов, которые надо игнорировать
# (по умолчанию исключаем loopback и типичные VPN/TUN адаптеры)
NET_IGNORE = [s.strip().lower() for s in os.getenv(
    "AGENT_NET_IGNORE",
    "lo,loopback,tun,tap,wg,tailscale"
).split(',') if s.strip()]

def _should_skip(name: str) -> bool:
    name_l = name.lower()
    return any(sub in name_l for sub in NET_IGNORE)

def gather_net_usage():
    """Посчитать сетевую скорость, исключив виртуальные интерфейсы."""
    global NET_LAST
    cur = psutil.net_io_counters(pernic=True)
    if NET_LAST is None:
        NET_LAST = cur
        return None, None
    up = down = 0
    for name, stats in cur.items():
        if _should_skip(name):
            continue
        last = NET_LAST.get(name)
        if not last:
            continue
        up += stats.bytes_sent - last.bytes_sent
        down += stats.bytes_recv - last.bytes_recv
    NET_LAST = cur
    # даже если трафика нет, возвращаем 0, а не None
    return up / INTERVAL, down / INTERVAL

def gather_metrics(full: bool = False) -> dict:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    net_up, net_down = gather_net_usage()
    cpu_temp = None
    tmp = get_cpu_temp()
    if tmp and tmp.split()[0].replace('.', '', 1).isdigit():
        cpu_temp = float(tmp.split()[0])
    uptime = int(time.time() - psutil.boot_time())
    gpu_data = gather_gpu_metrics() or {}
    data = {
        "cpu": cpu,
        "ram": mem.percent,
        "ram_used": mem.used,
        "ram_total": mem.total,
        "swap": swap.percent,
        "swap_used": swap.used,
        "swap_total": swap.total,
        "cpu_temp": cpu_temp,
        "uptime": uptime,
        **gpu_data,
        "net_up": net_up,
        "net_down": net_down,
    }
    if full:
        data["disks"] = gather_disks_metrics()
        data["top_procs"] = gather_top_processes(instant=True)
    else:
        data["disks"] = []
        data["top_procs"] = []
    return data


# ---------- async speedtest helper ----------
speedtest_running = False      # флаг «тест уже идёт»
diag_running = False
stability_running = False

def _speedtest_job():
    global speedtest_running
    try:
        push_text("⏳ Тестируем скорость…")
        dl, ul, ping = submit(run_speedtest).result()
        if dl is not None:
            push_text(
                f"💨 Speedtest:\n"
                f"↓ {dl:.1f} Mbit/s  ↑ {ul:.1f} Mbit/s  Ping {ping:.0f} ms"
            )
        else:
            push_text("⚠️  Speedtest не удался.")
    except Exception as exc:
        log.error("speedtest job error: %s", exc)
    finally:
        speedtest_running = False


def ws_send(obj: dict) -> None:
    if WS_CONN and WS_LOOP:
        fut = asyncio.run_coroutine_threadsafe(
            WS_CONN.send(json.dumps(obj)), WS_LOOP
        )
        try:
            fut.result()
            return
        except Exception as e:
            log.error("WS send error: %s, queued", e)
    else:
        log.warning("WS connection not ready, queued")
    with WS_PENDING_LOCK:
        WS_PENDING.append(obj)

def push_diag(txt: str, ok: bool = True):
    """Send diagnostics result to the server."""
    ws_send({"diag": txt, "diag_ok": ok})

def _diag_job():
    global diag_running
    try:
        push_text("⏳ Собираем диагностику…")
        out = submit(run_diagnostics).result()
        if out:
            push_diag(out, ok=True)
        else:
            push_diag("", ok=False)
    except Exception as exc:
        log.error("diagnostics job error: %s", exc)
    finally:
        diag_running = False


def _stability_job(interval_ms: int, duration_s: int) -> None:
    """Проверка стабильности соединения UDP-эхо."""
    global stability_running
    try:
        addr = (SERVER_IP, UDP_PORT)
        log.info(
            "stability test to %s:%s interval=%sms duration=%ss",
            SERVER_IP,
            UDP_PORT,
            interval_ms,
            duration_s,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.connect(addr)
        local_ip, local_port = sock.getsockname()
        log.info(
            "stability UDP socket %s:%s -> %s:%s",
            local_ip,
            local_port,
            SERVER_IP,
            UDP_PORT,
        )
        start_ts = time.time()
        rtts: list[float | None] = []
        pkt_idx = 0
        interval = interval_ms / 1000
        deadline = time.monotonic() + duration_s
        next_send = time.monotonic()
        # основное окно отправки
        while time.monotonic() < deadline:
            now = time.monotonic()
            # при задержках наверстываем пропущенные пакеты
            while now >= next_send:
                ts = time.time()
                payload = f"{pkt_idx}:{ts}".encode()
                try:
                    sock.send(payload)
                    rtts.append(None)
                    log.debug("stability packet %s sent", pkt_idx)
                except Exception as exc:
                    rtts.append(None)
                    log.warning("stability packet %s send failed: %s", pkt_idx, exc)
                pkt_idx += 1
                next_send += interval

            # собираем все полученные ответы
            while True:
                try:
                    data = sock.recv(1024)
                except BlockingIOError:
                    break
                except Exception as exc:
                    log.debug("stability recv failed: %s", exc)
                    break
                recv_time = time.time()
                idx_str, ts_str = data.decode().split(":", 1)
                idx = int(idx_str)
                rtt = (recv_time - float(ts_str)) * 1000
                if idx < len(rtts) and rtts[idx] is None:
                    rtts[idx] = rtt
                    log.debug("stability packet %s rtt %.2fms", idx, rtt)

            sleep_for = min(0.001, max(0.0, next_send - time.monotonic()))
            if sleep_for:
                time.sleep(sleep_for)

        # добираем поздние ответы ещё секунду
        end_wait = time.monotonic() + 1.0
        while time.monotonic() < end_wait:
            try:
                data = sock.recv(1024)
            except BlockingIOError:
                time.sleep(0.001)
                continue
            except Exception:
                break
            recv_time = time.time()
            idx_str, ts_str = data.decode().split(":", 1)
            idx = int(idx_str)
            rtt = (recv_time - float(ts_str)) * 1000
            if idx < len(rtts) and rtts[idx] is None:
                rtts[idx] = rtt
                log.debug("stability packet %s late rtt %.2fms", idx, rtt)

        sock.close()
        chunk = 10000
        total = (len(rtts) + chunk - 1) // chunk
        log.debug("stability collected %s samples, sending %s chunks", len(rtts), total)
        for i in range(total):
            part = rtts[i * chunk : (i + 1) * chunk]
            log.debug(
                "stability sending chunk %s/%s with %s samples", i + 1, total, len(part)
            )
            ws_send(
                {
                    "stability": {
                        "start_ts": start_ts,
                        "interval_ms": interval_ms,
                        "rtts": part,
                        "done": i + 1 == total,
                    }
                }
            )
    except Exception as exc:
        log.error("stability job error: %s", exc)
        ws_send({"stability": {"error": str(exc)}})
    finally:
        stability_running = False
# ────── network layer: TLS TOFU + fingerprint pinning ────────────
import ssl, socket, json, hashlib, pathlib, logging
from urllib.parse import urlparse
import sys

log      = logging.getLogger(__name__)
FP_FILE  = pathlib.Path.home() / ".bot_fingerprint.json"

def _fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()

def _load_fp() -> str | None:
    return json.loads(FP_FILE.read_text())["fp"] if FP_FILE.exists() else None

def _save_fp(fp: str) -> None:
    FP_FILE.write_text(json.dumps({"fp": fp}))

def _fetch_cert_der(parsed) -> bytes:
    host, port = parsed.hostname, parsed.port or 443
    ctx = ssl._create_unverified_context()
    with socket.create_connection((host, port), timeout=5) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as s:
            return s.getpeercert(binary_form=True)

def _mismatch_exit(pinned: str, new_fp: str) -> None:
    msg = (
        "\n❌ Ошибка TLS!\n"
        f"• сохранён: {pinned}\n"
        f"• получен: {new_fp}\n"
        f"\nℹ️  Удалите файл {FP_FILE} и запустите агент заново, "
        "чтобы доверить новому сертификату."
    )
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("TLS ошибка", msg)
    except Exception:
        pass
    print(msg, file=sys.stderr)
    sys.exit(1)

def _ensure_fp(url: str) -> str:
    pinned = _load_fp()
    cert_der = _fetch_cert_der(urlparse(url))
    current_fp = _fingerprint(cert_der)
    if pinned is None:
        _save_fp(current_fp)
        log.info("\ud83c\udf89  Cert saved, fp=%s\u2026", current_fp[:16])
    elif pinned != current_fp:
        _mismatch_exit(pinned, current_fp)
    return current_fp


def push_text(txt: str):
    ws_send({"text": txt})


def push_metrics(data: dict, oneshot: bool = False):
    payload = dict(data)
    if oneshot:
        payload["oneshot"] = True
    ws_send(payload)



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


async def _send_metrics_loop(ws: websockets.WebSocketClientProtocol) -> None:
    """Периодическая отправка метрик."""
    psutil.cpu_percent(interval=None)
    _refresh_proc_cache()
    init_gpu_metrics()
    while True:
        try:
            metrics = gather_metrics()
            await ws.send(json.dumps(metrics))
        except Exception as exc:
            log.error("WS send error: %s", exc)
            break
        await asyncio.sleep(INTERVAL)


async def _recv_loop(ws: websockets.WebSocketClientProtocol) -> None:
    """Получение команд от сервера."""
    global stability_running
    while True:
        try:
            resp = json.loads(await ws.recv())
        except Exception as exc:
            log.error("WS recv error: %s", exc)
            break
        for c in resp.get("commands", []):
            if c == "reboot":
                log.info("cmd reboot"); push_text("⚡️ Rebooting…"); do_reboot()
            elif c == "shutdown":
                log.info("cmd shutdown"); push_text("💤 Shutting down…"); do_shutdown()
            elif c == "speedtest":
                if not speedtest_running:
                    log.info("cmd speedtest (async)")
                    threading.Thread(target=_speedtest_job, daemon=True).start()
                else:
                    push_text("🚧 Speedtest уже выполняется, дождитесь окончания.")
            elif c == "diag":
                if not diag_running:
                    log.info("cmd diagnostics (async)")
                    threading.Thread(target=_diag_job, daemon=True).start()
                else:
                    push_text("🚧 Диагностика уже выполняется, дождитесь окончания.")
            elif c.startswith("stability"):
                if not stability_running:
                    try:
                        _, ms, dur = c.split()
                        ms_i = int(ms)
                        dur_i = int(dur)
                    except Exception:
                        push_text("⚠️ Неверные параметры теста.")
                        continue
                    log.info("cmd stability %sms %ss", ms_i, dur_i)
                    stability_running = True
                    threading.Thread(
                        target=_stability_job,
                        args=(ms_i, dur_i),
                        daemon=True,
                    ).start()
                else:
                    push_text("🚧 Тест стабильности уже выполняется, дождитесь окончания.")
            elif c == "status":
                await ws.send(json.dumps({**gather_metrics(full=True), "oneshot": True}))


async def ws_main() -> None:
    """Main loop using WebSockets."""
    global WS_LOOP, WS_CONN, WS_MAIN_TASK
    uri = f"{'wss' if SCHEME == 'https' else 'ws'}://{SERVER_IP}:{PORT}/ws/{SECRET}"
    ssl_ctx = None
    if SCHEME == 'https':
        # disable certificate validation unless explicitly forced
        if isinstance(VERIFY_SSL, str):
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.load_verify_locations(VERIFY_SSL)
        elif VERIFY_ENV == "force":
            ssl_ctx = ssl.create_default_context()
        else:
            ssl_ctx = ssl._create_unverified_context()
    _ensure_fp(SERVER)
    WS_LOOP = asyncio.get_running_loop()
    WS_MAIN_TASK = asyncio.current_task()
    try:
        while True:
            try:
                async with websockets.connect(uri, ssl=ssl_ctx) as ws:
                    log.info("Agent WS connected → %s", uri)
                    WS_CONN = ws
                    with WS_PENDING_LOCK:
                        queued = WS_PENDING.copy()
                        WS_PENDING.clear()
                    for obj in queued:
                        try:
                            await ws.send(json.dumps(obj))
                        except Exception as exc:
                            log.error("WS queued send failed: %s", exc)
                            with WS_PENDING_LOCK:
                                WS_PENDING.append(obj)
                            break
                    tasks = [
                        asyncio.create_task(_send_metrics_loop(ws)),
                        asyncio.create_task(_recv_loop(ws)),
                    ]
                    try:
                        _, pending_tasks = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_EXCEPTION
                        )
                        for t in pending_tasks:
                            t.cancel()
                    finally:
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("WS connection error: %s", exc)
            finally:
                WS_CONN = None
            log.info("WS reconnecting in %ss…", RECONNECT_DELAY)
            await asyncio.sleep(RECONNECT_DELAY)
    except asyncio.CancelledError:
        log.info("WS main cancelled, shutting down")
        return
    finally:
        WS_CONN = None
        WS_LOOP = None
        WS_MAIN_TASK = None

# ────────────────────────── main loop ─────────────────────────────────────

def main() -> None:
    global TRAY_ICON
    TRAY_ICON = start_tray_icon()
    if os.name == "nt":
        threading.Thread(target=_hide_console_on_minimize, daemon=True).start()
    asyncio.run(ws_main())


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
