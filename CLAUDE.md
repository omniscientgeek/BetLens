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
- **API**: Python (FastAPI + uvicorn, asyncio)

## Common Commands

### Website (`website/`)
- `npm start` - Start dev server
- `npm run build` - Production build
- `npm test` - Run tests

### Webservice (`webservice/`)
- `pip install -r requirements.txt` - Install dependencies
- `python app.py` - Start the API server

## BetStamp Intelligence MCP Server

A standalone MCP server (`mcp-server/mcp_server.py`) that exposes betting intelligence tools for Claude. Registered in `.mcp.json` as `betstamp-intelligence`.

**Run manually:** `python mcp-server/mcp_server.py`

### Available MCP Tools (24 total)

| Tool | Purpose |
|------|---------|
| `list_data_files` | List available betting data files |
| `list_events` | List games/events with team info and book counts |
| `get_odds_comparison` | Side-by-side odds across all books for a game |
| `get_best_odds` | Best available odds for a specific bet (game + market + side) |
| `get_worst_odds` | Worst available odds — books to avoid |
| `get_vig_analysis` | Vig rankings across books and games (lower = fairer) |
| `get_hold_percentage` | Hold percentage by sportsbook — overall margin breakdown by market type |
| `find_arbitrage_opportunities` | Scan for guaranteed-profit arb situations |
| `find_expected_value_bets` | Find +EV bets where a book misprices vs consensus |
| `detect_stale_lines` | Lines that haven't updated recently vs peers |
| `infer_odds_movement` | Compare stale vs fresh odds to infer sharp money direction (e.g., spread moved -4.5→-5.5 = sharps on favorite) |
| `detect_line_outliers` | Odds AND lines that deviate significantly from consensus (spread/total line + odds outliers) |
| `get_fair_odds` | Consensus no-vig "true" probability baseline per game/market — compare books against this |
| `get_market_summary` | Full digest — start here for an overview (includes middles, fair odds, outliers) |
| `calculate_odds` | Quick calculator: American odds → probability, decimal, payout |
| `get_best_bets_today` | Top-N ranked bets by composite value score (EV + outlier + vig + freshness) |
| `find_middle_opportunities` | Find middles where spread/total gaps let you win both sides |
| `get_book_rankings` | Multi-metric sportsbook report card (vig, odds quality, freshness) |
| `get_implied_scores` | Combine spread + total to estimate implied final score per team (e.g., spread -5.5 + total 220 → Home ~107.25, Away ~112.75) |
| `get_power_rankings` | Market-implied team strength ratings & relative rankings from moneyline odds |
| `get_sharpness_scores` | Score each sportsbook's sharpness vs Pinnacle (benchmark sharp book) — correlation, line diffs, classification |
| `get_market_correlations` | Cross-market consistency analysis: spread vs ML vs total correlations, per-book consistency scores, mispricing flags |
| `get_market_entropy` | Shannon entropy of implied probabilities across books — higher entropy = more disagreement = potentially exploitable |
| `get_daily_digest` | Structured daily briefing: must-bet, avoid, interesting, book grades, power rankings |

### MCP Resources

| URI | Purpose |
|-----|---------|
| `betstamp://data/{filename}` | Raw odds data from any file |
| `betstamp://glossary` | Betting terms glossary for context |
