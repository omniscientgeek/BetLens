# BetStamp AI Odds Agent

An AI-powered odds analysis agent that ingests sportsbook data and delivers actionable betting intelligence through a 5-phase pipeline: **Detect**, **Analyze**, **Audit**, **Brief**, **Audit**.

## Architecture

```
website/          React 19 SPA — briefing display, chat, tool-call viewer
webservice/       Python FastAPI + Socket.IO — pipeline orchestrator, AI service
mcp-server/       MCP server — 40 betting intelligence tools (odds, vig, arb, EV, etc.)
data/             Sample odds JSON (10 NBA games x 8 sportsbooks)
saved_results/    Completed pipeline run outputs
```

### Pipeline Flow

1. **Detect** — Enrich every odds record with implied probabilities, vig, fair odds, EV, stale-line flags, and arbitrage curves (pure math, no AI).
2. **Analyze** — Claude calls 5-10+ MCP tools (odds comparison, vig analysis, outlier detection, entropy, etc.) to cross-check the data. Mental arithmetic is forbidden; the agent must use MCP arithmetic tools.
3. **Audit Analysis** — Three concurrent verification agents (Reasoning, Factual, Betting) validate the analysis. If any agent fails, a self-healing loop rewrites and re-audits up to 3 times.
4. **Brief** — AI generates a structured daily market briefing (markdown) covering anomalies, value bets, sportsbook rankings, and more.
5. **Audit Brief** — Same 3-agent verification on the brief, with self-healing.

### Key Design Decisions

- **MCP tool use, not context-window dumping** — The agent queries data through structured tool calls, not by injecting the entire dataset into the prompt.
- **No hallucinated math** — Arithmetic MCP tools enforce real calculations. The system prompt explicitly forbids mental math.
- **Self-healing audits** — If verification fails, the pipeline automatically fixes and re-verifies rather than shipping bad output.
- **Multi-provider failover** — Claude SDK (primary), Anthropic API, OpenAI GPT-4o, and custom endpoints, tried in priority order.

## Local Development

### Prerequisites

- **Node.js 20+** and **Python 3.11+**
- An Anthropic API key (set `ANTHROPIC_API_KEY` env var) or Claude CLI authenticated

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/BetStamp.git
cd BetStamp

# 2. Start the API
cd webservice
pip install -r requirements.txt
npm install                       # Claude Agent SDK (Node.js wrapper)
python app.py                     # Starts on http://localhost:8191

# 3. Start the website (in a new terminal)
cd website
npm install
npm start                         # Starts on http://localhost:3000
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (if using Claude API directly) | Anthropic API key |
| `OPENAI_API_KEY` | No | OpenAI key (for GPT-4o failover) |
| `PORT` | No | API server port (default: 8191) |

## Deployment (Railway)

This project deploys as **two Railway services** from one monorepo.

### Setup

1. Create a new project on [railway.app](https://railway.app)

2. **Add the API service:**
   - New Service > GitHub Repo > select this repo
   - Settings > Root Directory: `/` (Dockerfile at `webservice/Dockerfile`)
   - Settings > Dockerfile Path: `webservice/Dockerfile`
   - Variables: `ANTHROPIC_API_KEY=sk-ant-...`
   - Deploy

3. **Add the Website service:**
   - New Service > GitHub Repo > select this repo
   - Settings > Root Directory: `/` (Dockerfile at `website/Dockerfile`)
   - Settings > Dockerfile Path: `website/Dockerfile`
   - Variables:
     - `REACT_APP_API_BASE=https://<api-service>.up.railway.app/api`
     - `REACT_APP_SOCKET_URL=https://<api-service>.up.railway.app`
   - Deploy

4. Generate public domains for both services in Railway Settings > Networking.

### Railway Environment Variables

| Service | Variable | Value |
|---------|----------|-------|
| API | `ANTHROPIC_API_KEY` | Your Anthropic API key |
| API | `PORT` | (auto-injected by Railway) |
| Website | `REACT_APP_API_BASE` | `https://<api-domain>/api` |
| Website | `REACT_APP_SOCKET_URL` | `https://<api-domain>` |
| Website | `PORT` | (auto-injected by Railway) |

## Tech Stack

- **Frontend:** React 19, React Router 7, Socket.IO Client, Create React App
- **Backend:** Python 3, FastAPI, Uvicorn, Python-SocketIO, Anthropic SDK, OpenAI SDK
- **AI:** Claude (via Claude Agent SDK + Anthropic API) with OpenAI failover
- **MCP Server:** 40 betting intelligence tools (odds math, anomaly detection, arbitrage, entropy, Shin model, GAMLSS, KNN, etc.)

## Project Structure

```
BetStamp/
├── README.md                  # This file
├── DEVLOG.md                  # Development log (required deliverable)
├── CLAUDE.md                  # Project guide for Claude Code
├── .gitignore
├── data/
│   └── sample.json            # 10 NBA games x 8 sportsbooks (with seeded anomalies)
├── mcp-server/
│   ├── mcp_server.py          # 40+ MCP tools (3300+ lines)
│   └── .mcp.json              # MCP server registration
├── webservice/
│   ├── Dockerfile             # Railway deployment (Python + Node.js)
│   ├── app.py                 # FastAPI + Socket.IO orchestrator
│   ├── ai_service.py          # Multi-provider AI with system prompts
│   ├── detect.py              # Detection phase (odds enrichment)
│   ├── verification_agents.py # 3-agent audit system
│   ├── odds_math.py           # Pure math utilities
│   ├── claude-sdk-wrapper.mjs # Node.js bridge for Claude Agent SDK
│   ├── requirements.txt       # Python dependencies
│   └── package.json           # Node dependencies (Claude Agent SDK)
├── website/
│   ├── Dockerfile             # Railway deployment (React + nginx)
│   ├── nginx.conf             # SPA routing config
│   ├── src/
│   │   ├── App.js             # Main app (pipeline state, WebSocket)
│   │   ├── api.js             # API config (env-var aware)
│   │   ├── BriefPanel.js      # Briefing display + verification badges
│   │   ├── ChatPanel.js       # Follow-up chat interface
│   │   ├── AnalyzeConversation.js  # Tool call viewer
│   │   ├── PastRuns.js        # Historical runs browser
│   │   └── AISettings.js      # Provider configuration UI
│   └── package.json
└── saved_results/             # Pipeline run outputs
```
