# BetStamp

A two-service application consisting of a React website and a Python (FastAPI) webservice API, with an integrated betting intelligence MCP server.

## Prerequisites

- **Node.js** (v18+ recommended) & **npm**
- **Python** 3.10+
- **pip**

## Project Structure

```
BetStamp/
├── website/        # React 19 frontend (Create React App)
├── webservice/     # Python FastAPI backend
├── mcp-server/     # BetStamp Intelligence MCP server
└── saved_results/  # Bet Lens saved analysis results
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd BetStamp
```

### 2. Website (React)

```bash
cd website
npm install
npm start
```

The dev server starts on **port 8190**. API requests are proxied to the backend automatically.

### 3. API (Python / FastAPI)

```bash
cd webservice
pip install -r requirements.txt
python app.py
```

The API server starts on **port 8191**.

### 4. MCP Server (optional)

```bash
python mcp-server/mcp_server.py
```

Exposes 40+ betting intelligence tools for use with Claude. Registered in `.mcp.json` as `betstamp-intelligence`.

## Common Commands

| Command | Directory | Description |
|---------|-----------|-------------|
| `npm start` | `website/` | Start React dev server |
| `npm run build` | `website/` | Production build |
| `npm test` | `website/` | Run tests |
| `pip install -r requirements.txt` | `webservice/` | Install Python dependencies |
| `python app.py` | `webservice/` | Start the API server |

## Tech Stack

- **Frontend:** React 19, React Router 7, Socket.IO Client, Create React App (react-scripts 5)
- **Backend:** Python, FastAPI, Uvicorn, Socket.IO, Anthropic SDK, OpenAI SDK
- **MCP Server:** Python, MCP SDK
