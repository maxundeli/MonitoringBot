# MonitoringBot

MonitoringBot is a simple client-server solution for collecting and visualizing host metrics through a Telegram bot. It started as a university coursework project and is maintained at <https://github.com/maxundeli/MonitoringBot>.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
  - [Running the Client](#running-the-client)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Features
- **Metric collection**: CPU and GPU load, memory usage, per-disk utilization, and system uptime.
- **Secure communication**: HTTPS with auto-generated self-signed certificates; HTTP available for testing.
- **Telegram integration**: Inline commands for status queries and remote actions (reboot/shutdown).
- **Data persistence**: Lightweight SQLite database.
- **Visualisation**: Metrics can be plotted and delivered via chat.

## Getting Started

### Prerequisites
- Python 3.11 or later
- Telegram bot token from **@BotFather**
- (optional) `virtualenv` or similar

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/loshara131313/MonitoringBot
cd MonitoringBot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Server
Set the required environment variable and start the FastAPI application:
```bash
export BOT_TOKEN="<telegram-bot-token>"
python server.py
```
Certificates (`cert.pem` and `key.pem`) are generated on the first run.

### Running the Client
Register a new agent in the Telegram chat:
```bash
/newkey My-PC
```
Use the returned token to launch the client on the monitored host:
```bash
export AGENT_SECRET="<token from /newkey>"
export AGENT_SERVER_IP="<server IPv4 address>"
python client.py
```

## Configuration
| Variable           | Component | Default | Description                     |
|--------------------|-----------|---------|---------------------------------|
| `BOT_TOKEN`        | server    | —       | Telegram bot token (required)   |
| `PORT`             | server    | 8000    | HTTP(S) listening port          |
| `AGENT_SECRET`     | client    | —       | Secret linking agent to server  |
| `AGENT_SERVER_IP`  | client    | prompt  | Server IPv4 address             |
| `AGENT_PORT`       | client    | 8000    | Server port                     |
| `AGENT_INTERVAL`   | client    | 5       | Seconds between metric pushes   |
| `AGENT_VERIFY_SSL` | client    | 1       | `0` disables certificate check  |

Configuration values persist to a local `.env` file after the first run.

## Project Structure
```
MonitoringBot/
├── client.py
├── server.py
├── requirements.txt
└── README.md
```

---
*This repository contains a coursework project and serves as a small-scale monitoring example.*
