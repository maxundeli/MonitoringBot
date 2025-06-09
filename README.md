# MonitoringBot

A lightweight client–server system for remote monitoring of basic host metrics.  
This project was developed as part of a university coursework assignment and is now maintained at **<https://github.com/maxundeli/MonitoringBot>**.

## Overview
MonitoringBot consists of:

| Component | Description |
|-----------|-------------|
| **server/** | FastAPI application that receives, stores and visualises metrics. Integrates with a Telegram bot for user interaction. Run with `python -m server`. |
| **client/** | Cross‑platform agent that collects host metrics (CPU, RAM, GPU, VRAM, disk, uptime) and periodically pushes them to the server. Run with `python -m client`. |

The server stores incoming data in SQLite and can generate on‑demand plots delivered via Telegram.

## Key Features
* **Metric collection** – CPU and GPU load, memory usage, per‑disk utilisation and system uptime.
* **Secure communication** – HTTPS with automatically generated self‑signed certificates (fallback to HTTP for testing).
* **Telegram integration** – Inline commands for status queries and remote actions (reboot / shutdown).
* **Data persistence** – Lightweight storage using SQLite; suitable for single‑instance deployments.
* **WebSocket transport** – two-way communication between agent and server without HTTP polling.
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

## Quick Start

### 1. Launch the server
Set the required environment variables and start the FastAPI application:
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
| `PORT` | server | 8000 | HTTP(S) listening port |
| `AGENT_SECRET` | client | — | Secret linking agent to server |
| `AGENT_SERVER_IP` | client | prompt | Server IPv4 address |
| `AGENT_PORT` | client | 8000 | Server port |
| `AGENT_INTERVAL` | client | 5 | Seconds between metric pushes |
| `AGENT_VERIFY_SSL` | client | 1 | `0` = disable verification |

Values are persisted to a local `.env` file after the first run.

### TLS fingerprint
On the first connection the client saves the server's TLS certificate
fingerprint to `~/.bot_fingerprint.json`. If the certificate later changes
(e.g. the server was reinstalled), the agent will stop with a dialog
window advising you to delete this file and restart. This prevents silently
accepting unexpected certificates.

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
