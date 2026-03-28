# BetStamp

A two-service application consisting of a React website and a Python (FastAPI) webservice API, with an integrated betting intelligence MCP server. The core product — **Bet Lens** — runs a multi-phase AI pipeline that analyzes sports betting odds, generates actionable briefings, and self-heals through automated audit/fix loops.

## Prerequisites

- **Node.js** (v18+ recommended) & **npm**
- **Python** 3.10+
- **pip**

## Project Structure

```
BetStamp/
├── website/          # React 19 frontend (Create React App)
│   └── src/
│       ├── App.js              # Main router, pipeline orchestration, WebSocket mgmt
│       ├── ActiveRunDetail.js   # Live pipeline progress display
│       ├── BriefPanel.js        # Brief markdown + verification badges
│       ├── ChatPanel.js         # Chat interface for AI interaction
│       ├── AnalyzeConversation.js # Analyze phase conversation view
│       ├── PastRuns.js          # Historical runs browser
│       ├── AISettings.js        # Provider configuration UI
│       └── AIAgents.js          # Verification agent results/badges
├── webservice/       # Python FastAPI backend
│   ├── app.py                  # FastAPI app, WebSocket server, pipeline orchestration
│   ├── ai_service.py           # AI provider abstraction, phase functions, fix prompts
│   ├── verification_agents.py  # 3-agent concurrent audit system
│   ├── detect.py               # Detection phase: enrichment & calculations
│   ├── mcp_client.py           # MCP client for betstamp-intelligence server
│   ├── odds_math.py            # Pure betting math functions
│   ├── ai_config.json          # AI provider configuration
│   └── claude-sdk-wrapper.mjs  # Node.js bridge to Claude Agent SDK
├── mcp-server/       # BetStamp Intelligence MCP server (40+ tools)
│   └── mcp_server.py
├── saved_results/    # Bet Lens saved analysis results
└── data/             # Raw odds data files (JSON)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│              (Socket.IO Client, Port 8190)               │
│                                                          │
│  BetLens ── ActiveRunDetail ── BriefPanel ── PastRuns   │
└──────────────────────┬──────────────────────────────────┘
                       │ WebSocket (Socket.IO)
                       │ Events: phase_complete, analyze_chunk,
                       │   brief_chunk, verification_agent_update,
                       │   fix_started, fix_complete, ...
                       ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend (Port 8191)                │
│                                                          │
│  Pipeline Orchestrator (app.py)                          │
│   ├── detect.py          ─── odds enrichment & math     │
│   ├── ai_service.py      ─── AI provider abstraction    │
│   ├── verification_agents.py ── 3-agent audit system    │
│   └── mcp_client.py      ─── MCP tool access            │
│                                                          │
│  AI Providers: Anthropic │ OpenAI │ Claude Agent SDK     │
│  (priority-based failover)                               │
└──────────────────────┬──────────────────────────────────┘
                       │ stdio transport
                       ▼
┌─────────────────────────────────────────────────────────┐
│           BetStamp Intelligence MCP Server               │
│                (mcp_server.py)                           │
│                                                          │
│  40+ tools: odds comparison, EV detection, vig analysis, │
│  arbitrage, Poisson modeling, Kelly sizing, anomaly      │
│  detection, sportsbook clustering, information flow ...  │
└─────────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

- **Streaming**: Analyze and brief phases stream tokens in real-time via WebSocket, providing live progress feedback.
- **Stateful Pipelines**: Pipeline state is cached in-memory with a 30-minute TTL. If a client disconnects and reconnects, all completed phase events are replayed automatically.
- **Multi-Provider Failover**: AI requests fall back through a priority-ordered provider list (configured in `ai_config.json`).
- **MCP-Driven Verification**: Audit agents use the same 40+ MCP tools to verify claims, ensuring verdicts are grounded in actual data.

---

## Bet Lens Processing Pipeline

When a user selects an odds data file, Bet Lens runs a **5-phase pipeline**. Each phase builds on the previous, and audit phases trigger a self-healing fix loop when they detect errors.

```
  ┌──────────┐     ┌──────────┐     ┌───────────────┐     ┌──────────┐     ┌─────────────┐
  │ 1. Detect │────▶│2. Analyze│────▶│3. Audit       │────▶│ 4. Brief │────▶│5. Audit     │
  │ (code)    │     │ (AI)     │     │   Analyze     │     │ (AI)     │     │   Brief     │
  └──────────┘     └──────────┘     │ (3 AI agents) │     └──────────┘     │(3 AI agents)│
                                     └───────┬───────┘                      └──────┬──────┘
                                             │ fail?                               │ fail?
                                             ▼                                     ▼
                                     ┌───────────────┐                      ┌─────────────┐
                                     │  Fix Analyze   │◀─┐                  │  Fix Brief   │◀─┐
                                     │  (AI rewrite)  │  │                  │ (AI rewrite) │  │
                                     └───────┬───────┘  │                  └──────┬──────┘  │
                                             │          │                         │          │
                                             ▼          │                         ▼          │
                                     ┌───────────────┐  │                  ┌─────────────┐  │
                                     │  Re-audit      │──┘                  │  Re-audit    │──┘
                                     │ (up to 2x)     │                    │ (up to 2x)   │
                                     └───────────────┘                      └─────────────┘
```

### Phase 1: Detect (Code-based, ~2-5s)

**File:** `webservice/detect.py`

Pure computational phase — no AI involved. Enriches every raw odds record with:

- **Implied probabilities** per side (spread, moneyline, total)
- **Vig (vigorish)** and vig percentage per book/market
- **No-vig fair probabilities** and fair American odds
- **Expected value (EV)** vs consensus fair odds (dollars + percentage)
- **Staleness detection** based on `last_updated` timestamps

Cross-sportsbook summaries computed:
- Consensus fair odds per game/market
- EV opportunities ranked by edge
- Stale line flags
- Arbitrage profit curves across all book-pair combinations
- Synthetic "perfect book" (best available odds per side)

### Phase 2: Analyze (AI-powered, streamed, ~3-10min)

**File:** `webservice/ai_service.py` → `run_analyze_phase()`

Uses the configured AI provider (with optional MCP tool access) to generate a detailed cross-sportsbook analysis. Tokens are streamed to the frontend in real-time via WebSocket `analyze_chunk` events.

**Output:** Full conversation record (system prompt, user prompt, assistant response, tool calls) plus AI metadata (provider, model, token usage, elapsed time).

### Phase 3: Audit Analyze (3 concurrent AI agents, ~2-5min)

**File:** `webservice/verification_agents.py`

Three specialized verification agents run **concurrently** via `asyncio.gather()`, each with access to the full set of 40+ MCP tools:

| Agent | Responsibility |
|-------|----------------|
| **Reasoning** | Validates logical consistency — no contradictions, rankings match data, conclusions follow from premises |
| **Factual** | Cross-checks every factual claim against source data using MCP tool calls |
| **Betting** | Verifies betting-specific accuracy — odds values, line numbers, EV calculations, Kelly sizing, arb math |

**Critical rule — Zero Mental Math:** All agents are explicitly forbidden from doing arithmetic in their heads. Every calculation must go through `arithmetic_*` MCP tools, ensuring full traceability and eliminating rounding/logic errors.

Each agent returns:
- **Verdict:** `pass` | `warn` | `fail`
- **Confidence:** 0.0–1.0
- **Checks:** total count and failed count
- **Issues:** list of `{severity, claim, finding}` objects
- **Summary:** 1-2 sentence overview

The **overall verdict** is the worst verdict across all three agents.

### Phase 4: Brief (AI-powered, streamed, ~2-5min)

**File:** `webservice/ai_service.py` → `run_brief_phase()`

Takes the detect + analyze results and generates a concise, actionable **markdown briefing**. Streamed to the frontend via `brief_chunk` events.

A preamble-stripping mechanism buffers initial chunks and only begins sending content after the first `## ` heading, ensuring clean markdown output.

### Phase 5: Audit Brief (3 concurrent AI agents, ~2-5min)

Same 3-agent audit system as Phase 3, but applied to the brief text instead of the analysis.

---

## Self-Healing Fix Loop

When an audit phase returns a `fail` verdict, the pipeline automatically enters a **fix loop** (up to 2 attempts):

1. **Fix Phase** — Calls `run_fix_phase()` with:
   - The original text (analysis or brief)
   - The full audit result with all agent issues
   - A phase-specific fix system prompt

2. **AI Rewrite** — The AI rewrites the content, guided by strict rules:
   - Fix **every** flagged issue (errors and warnings)
   - Use MCP tools to retrieve correct data — no fabrication
   - No mental math — use `arithmetic_*` tools for all calculations
   - Preserve the original structure and tone
   - Remove any claims that can't be verified

3. **Re-audit** — The fixed text is run through the same 3-agent audit. If the verdict is:
   - `pass` or `warn` → Accept and continue to the next pipeline phase
   - `fail` → Repeat the fix (up to `MAX_FIX_ATTEMPTS = 2`)

4. **Fix History** — Every fix attempt is tracked in `verification["fix_history"]` with the audit result, fix conversation, and resulting text. The final attempt count is stored in `verification["fix_attempts"]`.

---

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
- **MCP Server:** Python, MCP SDK (40+ betting intelligence tools)
- **Betting Math:** Implied probability, vigorish, no-vig fair odds, Kelly Criterion, Poisson modeling, Bayesian inference, Shin model, arbitrage, EV calculation

## Saved Results

Pipeline results are auto-saved to `saved_results/betlens_results_{YYYY-MM-DD_HHMMSS}.json` containing the source file, all pipeline phase outputs, audit verdicts, fix history, and the original input data. See `CLAUDE.md` for the full schema.
