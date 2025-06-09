from __future__ import annotations

"""Фоновый воркер клиента для тяжёлых задач."""

from concurrent.futures import ProcessPoolExecutor, Future
from pathlib import Path
import platform
import subprocess
import shutil
import tempfile
import locale

__all__ = [
    "submit",
    "run_speedtest",
    "run_diagnostics",
]

_executor = ProcessPoolExecutor()

def submit(func, *args, **kwargs) -> Future:
    """Отправить функцию в пул процессов."""
    return _executor.submit(func, *args, **kwargs)


def _subprocess_flags() -> int:
    if platform.system() == "Windows":
        return subprocess.CREATE_NO_WINDOW
    return 0


def run_speedtest() -> tuple[float | None, float | None, float | None]:
    try:
        try:
            import speedtest  # type: ignore
        except Exception:
            speedtest = None
        if speedtest:
            try:
                speedtest.printer = lambda *a, **k: None
            except Exception:
                pass
            st = speedtest.Speedtest(secure=True)
            st.get_best_server()
            dl = st.download() / 1e6
            ul = st.upload() / 1e6
            return dl, ul, st.results.ping

        import json
        for prog in ("speedtest", "speedtest-cli", "speedtest.exe", "speedtest-cli.exe"):
            path = shutil.which(prog)
            if not path:
                continue
            out = subprocess.check_output(
                [path, "--format=json"],
                text=True,
                timeout=120,
                creationflags=_subprocess_flags(),
            )
            data = json.loads(out)
            dl = data["download"]["bandwidth"] * 8 / 1e6
            ul = data["upload"]["bandwidth"] * 8 / 1e6
            ping = data["ping"]["latency"]
            return dl, ul, ping
    except Exception:
        return None, None, None
    return None, None, None


def run_diagnostics() -> str | None:
    try:
        if platform.system() == "Windows":
            dxdiag = shutil.which("dxdiag") or shutil.which("dxdiag.exe")
            if dxdiag:
                tmp = Path(tempfile.gettempdir()) / "dxdiag.txt"
                cmd = [dxdiag, "/dontskip", "/whql:off", "/t", str(tmp)]
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=120,
                    creationflags=_subprocess_flags(),
                )
                try:
                    return tmp.read_text(encoding="utf-16")
                except UnicodeError:
                    enc = locale.getpreferredencoding(False)
                    try:
                        return tmp.read_text(encoding=enc, errors="ignore")
                    except UnicodeError:
                        return tmp.read_text(encoding="utf-8", errors="ignore")
            sysinfo = shutil.which("systeminfo") or shutil.which("systeminfo.exe")
            if sysinfo:
                out = subprocess.check_output(
                    [sysinfo],
                    text=True,
                    timeout=120,
                    errors="ignore",
                    creationflags=_subprocess_flags(),
                )
                return out

        if shutil.which("inxi"):
            out = subprocess.check_output(
                ["inxi", "-F"],
                text=True,
                timeout=120,
                creationflags=_subprocess_flags(),
            )
            return out
        if shutil.which("lshw"):
            out = subprocess.check_output(
                ["lshw", "-short"],
                text=True,
                timeout=120,
                creationflags=_subprocess_flags(),
            )
            return out
    except Exception:
        return None
    return None
