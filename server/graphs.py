"""Функции для построения графиков."""
from __future__ import annotations

import io
import time
from statistics import median
from datetime import datetime, timedelta
from typing import List
import os
import re
from concurrent.futures import ProcessPoolExecutor, Future
import multiprocessing as mp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import gc

from .db import fetch_metrics, fetch_metrics_full

# matplotlib без X-сервера

matplotlib.use("Agg")

# количество процессов берётся из переменной окружения
# после завершения задачи исполнитель завершается, чтобы освободить память

def _new_executor() -> ProcessPoolExecutor:
    workers = os.getenv("GRAPH_WORKERS", "1")
    try:
        workers_num = int(workers)
    except ValueError:
        workers_num = 1
    workers_num = max(workers_num, 1)
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=workers_num, mp_context=ctx)


def submit(func, *args, **kwargs) -> Future:
    """Запустить функцию в отдельном процессе и закрыть его после выполнения."""
    executor = _new_executor()
    future = executor.submit(func, *args, **kwargs)
    future.add_done_callback(lambda _: executor.shutdown(wait=False))
    return future


def _find_gaps(ts, factor: float = 2.0):
    if len(ts) < 2:
        return [(0, len(ts) - 1)], [], 0

    intervals = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    med = median(intervals) if intervals else 0
    if med <= 0:
        med = max(intervals) if intervals else 60
    thr = med * factor

    segments, gaps = [], []
    start = 0
    for i, dt in enumerate(intervals, start=1):
        if dt > thr:
            segments.append((start, i - 1))
            gaps.append((ts[i - 1], ts[i]))
            start = i
    segments.append((start, len(ts) - 1))
    return segments, gaps, thr


TIME_RE = re.compile(r"^(\d+)([smhd])$", re.I)


def parse_timespan(tokens: List[str]) -> int:
    """Разобрать продолжительность вроде ['2d', '3h', '15m'] в секунды."""
    total = 0
    for t in tokens:
        m = TIME_RE.match(t)
        if not m:
            raise ValueError(f"invalid time token: {t}")
        val = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "s":
            total += val
        elif unit == "m":
            total += val * 60
        elif unit == "h":
            total += val * 3600
        elif unit == "d":
            total += val * 86400
    return total


def _plot_segments(ax, ts, ys, segments, *args, **kwargs):
    first = True
    col = None
    for s, e in segments:
        if first:
            line, = ax.plot(ts[s:e + 1], ys[s:e + 1], *args, **kwargs)
            col = line.get_color()
            first = False
        else:
            kw = dict(kwargs)
            kw.pop("label", None)
            kw["color"] = col
            ax.plot(ts[s:e + 1], ys[s:e + 1], *args, **kw)


def _make_figure(seconds: int):
    """Размер фигуры пропорционален запрашиваемому промежутку времени."""
    base_width = 12
    width = base_width
    hours = seconds / 3600
    if hours >= 6:
        width += base_width
    if hours >= 12:
        width += base_width
    if hours >= 24:
        days = seconds // 86_400
        width += base_width * days
    dpi = 500
    fig, ax = plt.subplots(figsize=(width, 6), dpi=dpi)

    base = 9
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 4,
        "axes.labelsize": base + 2,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
    })
    return fig, ax


def _apply_time_locator(ax, seconds: int) -> None:
    """Настроить частоту меток времени в зависимости от интервала."""
    hours = seconds / 3600
    if hours < 12:
        return
    if hours <= 48:
        interval = 1
    elif hours <= 7 * 24:
        interval = 2
    else:
        interval = 3
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %H:%M"))


UNIT_NAMES_BITS = ["bit", "Kbit", "Mbit", "Gbit", "Tbit", "Pbit"]


def best_unit(max_val_bytes: float) -> tuple[float, str]:
    """Подобрать единицу измерения скорости."""
    max_bits = max_val_bytes * 8
    scale_bits = 1.0
    idx = 0
    while idx < len(UNIT_NAMES_BITS) - 1 and max_bits >= 1000:
        max_bits /= 1000
        scale_bits *= 1000
        idx += 1
    unit = UNIT_NAMES_BITS[idx] + "/s"
    return scale_bits / 8, unit


def plot_metric(secret: str, metric: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]

    idx_map = {
        "cpu": (1, "CPU %", "%", (0, 100)),
        "ram": (2, "RAM %", "%", (0, 100)),
        "gpu": (3, "GPU %", "%", (0, 100)),
        "vram": (4, "VRAM %", "%", (0, 100)),
        "net_up": (5, "Net Up", "bit/s", None),
        "net_down": (6, "Net Down", "bit/s", None),
    }
    col_idx, label, ylab, ylim = idx_map[metric]
    ys = [np.nan if rows[i][col_idx] is None else rows[i][col_idx] for i in range(len(rows))]

    if metric.startswith("net_"):
        max_val = max([v for v in ys if not np.isnan(v)] or [0])
        scale, unit = best_unit(max_val)
        if scale != 1:
            ys = [v / scale if not np.isnan(v) else np.nan for v in ys]
        ylab = unit
        if max_val:
            ylim = (0, (max_val / scale) * 1.1)

    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    _plot_segments(ax, ts, ys, segments, linewidth=1.5)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_title(f"{label} за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=0.3)
    _apply_time_locator(ax, seconds)
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf


def plot_net(secret: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]
    up = [np.nan if r[5] is None else r[5] for r in rows]
    down = [np.nan if r[6] is None else r[6] for r in rows]
    max_val = max([v for v in up + down if not np.isnan(v)] or [0])
    scale, unit = best_unit(max_val)
    if scale != 1:
        up = [v / scale if not np.isnan(v) else np.nan for v in up]
        down = [v / scale if not np.isnan(v) else np.nan for v in down]
    ylim_top = None
    if max_val:
        ylim_top = (max_val / scale) * 1.1
    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    _plot_segments(ax, ts, up, segments, label="Up", linewidth=1.2)
    _plot_segments(ax, ts, down, segments, label="Down", linewidth=1.2)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_title(f"Net за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel(unit)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    ax.grid(True, linestyle="--", linewidth=0.3)
    _apply_time_locator(ax, seconds)
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf


def plot_all_metrics(secret: str, seconds: int):
    rows = fetch_metrics(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r[0]) for r in rows]
    segments, gaps, _ = _find_gaps(ts)

    cpu = [r[1] for r in rows]
    ram = [r[2] for r in rows]
    gpu = [np.nan if r[3] is None else r[3] for r in rows]
    vram = [np.nan if r[4] is None else r[4] for r in rows]

    plt.style.use("dark_background")

    fig, ax = _make_figure(seconds)

    for ys, lab in ((cpu, "CPU %"), (ram, "RAM %")):
        _plot_segments(ax, ts, ys, segments, label=lab, linewidth=1.2)
    if not all(np.isnan(g) for g in gpu):
        _plot_segments(ax, ts, gpu, segments, label="GPU %", linewidth=1.2)
    if not all(np.isnan(v) for v in vram):
        _plot_segments(ax, ts, vram, segments, label="VRAM %", linewidth=1.2)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_ylim(0, 100)
    ax.set_title(f"Все метрики за {timedelta(seconds=seconds)}")
    ax.set_xlabel("Время")
    ax.set_ylabel("%")
    ax.grid(True, linestyle="--", linewidth=0.3)
    _apply_time_locator(ax, seconds)
    ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf


MEM_UNITS = {
    "b": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 ** 2,
    "mib": 1024 ** 2,
    "gb": 1024 ** 3,
    "gib": 1024 ** 3,
}

NET_UNITS = {
    "bit": 1,
    "kbit": 1000,
    "mbit": 1000 ** 2,
    "gbit": 1000 ** 3,
}


def plot_custom(secret: str, metrics: list[str], seconds: int, ylim_top: float | None, unit: str | None):
    rows = fetch_metrics_full(secret, int(time.time()) - seconds)
    if not rows:
        return None

    ts = [datetime.fromtimestamp(r["ts"]) for r in rows]
    segments, gaps, _ = _find_gaps(ts)

    plt.style.use("dark_background")
    fig, ax = _make_figure(seconds)

    unit_category = None
    ylab = unit

    data_sets: list[tuple[list[float], str]] = []

    for m in metrics:
        m = m.lower()
        if m == "net":
            metrics.extend(["net_up", "net_down"])
            continue

    for m in metrics:
        m = m.lower()
        label = m.upper()
        ys = []
        cat = ""
        default_unit = "%"

        if m == "cpu":
            ys = [r["cpu"] for r in rows]
            cat = "percent"
        elif m == "gpu":
            ys = [r["gpu"] for r in rows]
            cat = "percent"
        elif m == "ram":
            if unit and unit.lower() not in {"%", "percent"}:
                factor = MEM_UNITS.get(unit.lower(), 1024 ** 2)
                ys = [r["ram_used"] / factor if r["ram_used"] is not None else np.nan for r in rows]
                cat = "bytes"
                default_unit = unit
            else:
                ys = [r["ram"] for r in rows]
                cat = "percent"
        elif m == "vram":
            if unit and unit.lower() not in {"%", "percent"}:
                factor = MEM_UNITS.get(unit.lower(), 1024 ** 2)
                ys = [r["vram_used"] / factor if r["vram_used"] is not None else np.nan for r in rows]
                cat = "bytes"
                default_unit = unit
            else:
                ys = [r["vram"] for r in rows]
                cat = "percent"
        elif m == "net_up":
            cat = "net"
            default_unit = "bit/s"
            if unit and unit.lower() in NET_UNITS:
                scale = NET_UNITS[unit.lower()]
                ys = [r["net_up"] * 8 / scale if r["net_up"] is not None else np.nan for r in rows]
            else:
                max_val = max([r["net_up"] or 0 for r in rows])
                scale, auto_unit = best_unit(max_val)
                ys = [r["net_up"] / scale if r["net_up"] is not None else np.nan for r in rows]
                if not unit:
                    ylab = auto_unit
            label = "Up"
        elif m == "net_down":
            cat = "net"
            default_unit = "bit/s"
            if unit and unit.lower() in NET_UNITS:
                scale = NET_UNITS[unit.lower()]
                ys = [r["net_down"] * 8 / scale if r["net_down"] is not None else np.nan for r in rows]
            else:
                max_val = max([r["net_down"] or 0 for r in rows])
                scale, auto_unit = best_unit(max_val)
                ys = [r["net_down"] / scale if r["net_down"] is not None else np.nan for r in rows]
                if not unit:
                    ylab = auto_unit
            label = "Down"
        else:
            continue

        if unit_category is None:
            unit_category = cat
            if not ylab:
                ylab = default_unit
        elif unit_category != cat:
            raise ValueError("Единицы метрик не совпадают")

        data_sets.append((ys, label))

    for ys, lab in data_sets:
        _plot_segments(ax, ts, ys, segments, label=lab, linewidth=1.2)

    for g0, g1 in gaps:
        ax.axvspan(g0, g1, facecolor="none", hatch="//", edgecolor="white", alpha=0.3, linewidth=0)

    ax.set_xlabel("Время")
    ax.set_ylabel(ylab or "%")
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    ax.set_title(f"{'/'.join([l for _, l in data_sets])} за {timedelta(seconds=seconds)}")
    ax.grid(True, linestyle="--", linewidth=0.3)
    _apply_time_locator(ax, seconds)
    if len(data_sets) > 1:
        ax.legend(loc="upper left", fontsize="small")
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, dpi=fig.dpi, format="png")
    plt.close(fig)
    gc.collect()
    buf.seek(0)
    return buf
