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
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "gpu": util,
            "vram_used": mem.used / 2 ** 20,
            "vram_total": mem.total / 2 ** 20,
            "vram": mem.used / mem.total * 100 if mem.total else None,
            "gpu_temp": float(temp),
        }
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
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
            return {
                "gpu": util,
                "vram_used": used,
                "vram_total": total,
                "vram": used / total * 100 if total else None,
                "gpu_temp": temp,
            }
        except Exception:
            pass

    try:
        import GPUtil

        gpu = GPUtil.getGPUs()[0]
        util = gpu.load * 100
        used = gpu.memoryUsed
        total = gpu.memoryTotal
        temp = gpu.temperature
        return {
            "gpu": util,
            "vram_used": used,
            "vram_total": total,
            "vram": used / total * 100 if total else None,
            "gpu_temp": temp,
        }
    except Exception:
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
    try:
        import amdsmi
        amdsmi.amdsmi_init()
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
            amdsmi.amdsmi_shut_down()
            return {
                "gpu": util,
                "vram_used": used,
                "vram_total": total,
                "vram": used / total * 100 if total else None,
                "gpu_temp": temp,
            }
    except Exception:
        pass

    if shutil.which("amd-smi"):
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
            return {
                "gpu": util,
                "vram_used": used,
                "vram_total": total,
                "vram": used / total * 100 if total else None,
                "gpu_temp": temp,
            }
        except Exception:
            pass

    if platform.system() == "Windows":
        data = _windows_wmi_amd_metrics()
        if data:
            return data

        try:
            import adlxpy

            helper = adlxpy.ADLXHelper()
            if helper.initialize():
                system = helper.get_system()
                gpu = system.get_gpus().at(0)
                perf = system.get_performance_monitoring_services()
                metrics = perf.get_gpu_metrics(gpu)

                util = metrics.gpu_utilization()
                vram = metrics.vram_usage()
                used = vram.vram_used() / 2 ** 20
                total = vram.vram_total() / 2 ** 20
                temp = metrics.gpu_temperatures().edge_current()

                helper.terminate()
                return {
                    "gpu": util,
                    "vram_used": used,
                    "vram_total": total,
                    "vram": used / total * 100 if total else None,
                    "gpu_temp": temp,
                }
        except Exception:
            pass

        try:
            from pyadl import ADLManager

            devs = ADLManager.getInstance().getDevices()
            if devs:
                dev = devs[0]
                util = dev.getCurrentUsage()
                temp = dev.getCurrentTemperature()
                return {"gpu": util, "gpu_temp": temp}
        except Exception:
            pass

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


def gather_gpu_metrics() -> dict | None:
    vendor = detect_gpu_vendor()
    if vendor == "nvidia":
        return _nvidia_gpu_metrics()
    if vendor == "amd":
        return _amd_gpu_metrics()

    # неизвестно – пробуем всё подряд
    for fn in (_nvidia_gpu_metrics, _amd_gpu_metrics):
        data = fn()
        if data:
            return data
    return None

# ──────────────────────── network usage ─────────────────────────-
NET_LAST = None

def gather_net_usage():
    global NET_LAST
    cur = psutil.net_io_counters()
    if NET_LAST is None:
        NET_LAST = cur
        return None, None
    up = cur.bytes_sent - NET_LAST.bytes_sent
    down = cur.bytes_recv - NET_LAST.bytes_recv
    NET_LAST = cur
    return up / INTERVAL, down / INTERVAL

def gather_metrics() -> dict:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    net_up, net_down = gather_net_usage()
    cpu_temp = None
    tmp = get_cpu_temp()
    if tmp and tmp.split()[0].replace('.', '', 1).isdigit():
        cpu_temp = float(tmp.split()[0])
    uptime = int(time.time() - psutil.boot_time())
    gpu_data = gather_gpu_metrics() or {}
    disks = gather_disks_metrics()
    return {
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
        "disks": disks,
        "net_up": net_up,
        "net_down": net_down,
    }


def run_speedtest() -> tuple[float | None, float | None, float | None]:
    try:
        if speedtest:
            st = speedtest.Speedtest(secure=True)
            st.get_best_server()
            dl = st.download() / 1e6
            ul = st.upload() / 1e6
            return dl, ul, st.results.ping

        import shutil, subprocess, json

        if shutil.which("speedtest"):
            out = subprocess.check_output(
                ["speedtest", "--format=json"], text=True, timeout=120
            )
            data = json.loads(out)
            dl = data["download"]["bandwidth"] * 8 / 1e6
            ul = data["upload"]["bandwidth"] * 8 / 1e6
            ping = data["ping"]["latency"]
            return dl, ul, ping
    except Exception as exc:
        log.error("speedtest failed: %s", exc)
    return None, None, None
# ---------- async speedtest helper ----------
speedtest_running = False      # флаг «тест уже идёт»

def _speedtest_job():
    global speedtest_running
    push_text("⏳ Тестируем скорость…")
    dl, ul, ping = run_speedtest()
    if dl is not None:
        push_text(
            f"💨 Speedtest:\n"
            f"↓ {dl:.1f} Mbit/s  ↑ {ul:.1f} Mbit/s  Ping {ping:.0f} ms"
        )
    else:
        push_text("⚠️  Speedtest не удался.")
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

def push_text(txt: str):
    try:
        r = _request("POST", f"{SERVER}/api/push/{SECRET}", json={"text": txt})
        r.raise_for_status()
    except Exception as e:
        log.error("push error: %s", e)


def push_metrics(data: dict):
    try:
        r = _request("POST", f"{SERVER}/api/push/{SECRET}", json=data)
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
    metrics = gather_metrics()
    push_metrics(metrics)
    for c in pull_cmds():
        if c == "reboot":
            log.info("cmd reboot"); push_text("⚡️ Rebooting…"); do_reboot()
        elif c == "shutdown":
            log.info("cmd shutdown"); push_text("💤 Shutting down…"); do_shutdown()
        elif c == "speedtest":
            # запускаем тест в отдельном потоке, чтобы не блокировать основной цикл
            if not speedtest_running:
                log.info("cmd speedtest (async)")
                speedtest_running = True
                threading.Thread(target=_speedtest_job, daemon=True).start()
            else:
                push_text("🚧 Speedtest уже выполняется, дождитесь окончания.")
    time.sleep(INTERVAL)