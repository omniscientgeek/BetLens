# Bet Stamp - Project Guide

## MCP Project Manager

This project is managed via the **MCP Project Manager** (project ID: **10**, name: "Bet Stamp"). Use MCP tools to start, stop, and restart services instead of manual commands.

> **Important:** Always `select_project(projectId: 10)` before using entity-scoped operations (list, get, logs).

| Entity | Type | ID | Port | Command | Directory |
|--------|------|----|------|---------|-----------|
| BetStamp Website | Website | 11 | 8190 | `npm start` | `./website` |
| BetStamp API | WebService | 25 | 8191 | `pip install -r requirements.txt && python app.py` | `./webservice` |

**Common MCP operations:**
- `start_project(projectId: 10)` - Start all services
- `stop_project(projectId: 10)` - Stop all services
- `restart_project(projectId: 10)` - Restart all services
- `start_process` / `stop_process` / `restart_process` - Manage individual entities
- `get_logs` - View process output logs
- `get_project_status(projectId: 10)` - Check running status of all entities

## Project Overview

A two-service application consisting of a React website and a Python webservice API.

## Local Development

| Service | Command | Working Directory |
|---------|---------|-------------------|
| Website (React) | `npm start` | `./website` |
| API (Python) | `pip install -r requirements.txt && python app.py` | `./webservice` |

## Tech Stack

- **Website**: React 19, Create React App (react-scripts 5)
- **API**: Python (Flask/FastAPI)

## Common Commands

### Website (`website/`)
- `npm start` - Start dev server
- `npm run build` - Production build
- `npm test` - Run tests

### Webservice (`webservice/`)
- `pip install -r requirements.txt` - Install dependencies
- `python app.py` - Start the API server
