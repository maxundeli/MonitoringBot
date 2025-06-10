# MonitoringBot

A lightweight client–server system for remote monitoring of basic host metrics.  
This project was developed as part of a university coursework assignment and is now maintained at **<https://github.com/maxundeli/MonitoringBot>**.

## Overview
MonitoringBot consists of:

| Component | Description |
|-----------|-------------|
| **server/** | FastAPI application that receives, stores and visualises metrics. Integrates with a Telegram bot for user interaction. Run with `python -m server`. |
| **client/** | Cross‑platform agent that collects host metrics (CPU, RAM, GPU, VRAM, disk, uptime) and periodically pushes them to the server. Run with `python -m client`. |

The server stores incoming data in a MySQL database and can generate on‑demand plots delivered via Telegram.

## Key Features
* **Metric collection** – CPU and GPU load, memory usage, per‑disk utilisation and system uptime.
* **Secure communication** – WebSocket over TLS with automatically generated self‑signed certificates (fallback to WS for testing).
* **Telegram integration** – Inline commands for status queries and remote actions (reboot / shutdown).
* **Data persistence** – Uses MySQL for reliable storage and multi‑instance deployments.
* **WebSocket transport** – persistent bidirectional connection lets the server trigger tasks like speed tests instantly and receive full status without delay.
* **Visualization** – Matplotlib charts returned directly in chat.

## Installation

### Prerequisites
* Python 3.11 or later
* A Telegram bot token created with **@BotFather**
* (Optional) `virtualenv` or similar tool

### Clone the repository
```bash
git clone https://github.com/maxundeli/MonitoringBot
cd MonitoringBot
```

### Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Prepare the MySQL database
Запустите локальный сервер MySQL. Приложение использует базу `monitoring` c учётными данными `root`/`""` и создаёт её автоматически при первом запуске. Также будут перенесены данные из `metrics.sqlite` и `db.json`, если такие файлы существуют.
## Quick Start

### 1. Launch the server
Укажите токен бота и запустите FastAPI приложение:
```bash
export BOT_TOKEN="<telegram‑bot‑token>"
python -m server
```
The first launch generates `cert.pem` and `key.pem` for TLS.

### 2. Register an agent
In the Telegram chat, obtain an agent secret:
```
/newkey My‑PC
```
This returns a unique token for the host.

### 3. Start the client on the monitored host
```bash
export AGENT_SECRET="<token from /newkey>"
export AGENT_SERVER_IP="<server IPv4 address>"
python -m client
```

### 4. Interact through Telegram
* `/status` – current CPU/RAM/disk usage
* `CPU` / `RAM` buttons – historical charts
* `Reboot` / `Shutdown` – remote control actions
* `/setalert <key|name> <metric> <threshold>` – configure alerts
* `/delalert <key|name> <metric>` – remove alert
* `/plot <key|name> <metrics> <interval> [limit] [unit]` – custom chart

## Configuration

| Variable | Component | Default | Purpose |
|----------|-----------|---------|---------|
| `BOT_TOKEN` | server | — | Telegram bot token (required) |
| `PORT` | server | 8000 | WebSocket listening port |
| `GRAPH_WORKERS` | server | 1 | Processes for each chart (terminated after use) |
| `AGENT_SECRET` | client | — | Secret linking agent to server |
| `AGENT_SERVER_IP` | client | prompt | Server IPv4 address |
| `AGENT_PORT` | client | 8000 | Server port |
| `AGENT_INTERVAL` | client | 5 | Seconds between metric pushes |
| `AGENT_RECONNECT_DELAY` | client | 5 | Seconds before reconnecting |
| `AGENT_VERIFY_SSL` | client | 1 | `0` = disable verification |
| `AGENT_ICON_FILE` | client | `client/icon.png` if exists | Tray icon image path |

Values are persisted to a local `.env` file after the first run.

### TLS fingerprint
On the first connection the client saves the server's TLS certificate
fingerprint to `~/.bot_fingerprint.json`. If the certificate later changes
(e.g. the server was reinstalled), the agent will stop with a dialog
window advising you to delete this file and restart. This prevents silently
accepting unexpected certificates.

### Tray icon
When the client starts, an icon appears in the system tray. Right-click this icon to exit the program. To use your own image, set `AGENT_ICON_FILE` to the path of a PNG before launching:
```bash
export AGENT_ICON_FILE="/path/my_icon.png"
python -m client
```
If this variable is not set, the client will try to load `client/icon.png` from the program directory.
On Windows double clicking the icon toggles the console window. If the program
was built using `--noconsole`, the console will be allocated when you double
click.

## Project Structure
```
MonitoringBot/
├── client/
│   ├── __main__.py  # client entry point
│   └── worker.py    # heavy tasks
├── server/
│   ├── __main__.py  # main server
│   ├── db.py        # DB helper
│   └── graphs.py    # graphs building
├── requirements.txt
└── README.md
```



---

*This README provides a concise reference for deploying and using MonitoringBot in coursework demonstrations or small‑scale monitoring scenarios.*
