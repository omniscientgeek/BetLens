"""
BetStamp MCP Server — Betting Intelligence Tools for Claude

Exposes structured betting data (odds, value, vig, comparisons, anomalies)
via MCP so Claude can reason about best/worst bets, arbitrage, and market
inefficiencies.

Usage:
    python mcp_server.py          # stdio transport (default for Claude)
    python mcp_server.py --sse    # SSE transport for web clients
"""

import sys
import os
import json
from datetime import datetime, timezone
from typing import Optional

# Allow imports from sibling webservice/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))

from mcp.server.fastmcp import FastMCP
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, fair_odds_to_american

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

mcp = FastMCP(
    "BetStamp Intelligence",
    instructions="Betting intelligence MCP — odds comparison, value detection, vig analysis, and anomaly spotting. Use get_market_summary as a starting point, then drill into specific tools.",
)


# ---------------------------------------------------------------------------
# Cache — avoids redundant file I/O, parsing, and enrichment
# ---------------------------------------------------------------------------

class _OddsCache:
    """In-memory cache for loaded odds, enriched records, and analysis results.

    Keyed by resolved filepath.  Automatically invalidates when the file's
    mtime changes (i.e. the data file is refreshed on disk).
    """

    def __init__(self):
        self._raw: dict[str, list[dict]] = {}          # filepath -> raw odds
        self._mtime: dict[str, float] = {}              # filepath -> os.path.getmtime
        self._enriched: dict[str, list[dict]] = {}      # filepath -> enriched odds
        self._by_game: dict[str, dict] = {}             # filepath -> grouped by game_id
        self._analysis: dict[str, dict] = {}            # (filepath, tool_key) -> result

    def _resolve(self, filename: Optional[str] = None) -> str:
        """Return the absolute filepath for a given filename (or the default)."""
        if filename:
            return os.path.join(DATA_DIR, filename)
        files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
        if not files:
            return ""
        return os.path.join(DATA_DIR, files[0])

    def _is_valid(self, filepath: str) -> bool:
        """Check if cached data is still fresh (file hasn't changed)."""
        if filepath not in self._mtime:
            return False
        try:
            return os.path.getmtime(filepath) == self._mtime[filepath]
        except OSError:
            return False

    def _invalidate(self, filepath: str):
        """Clear all caches for a filepath."""
        self._raw.pop(filepath, None)
        self._mtime.pop(filepath, None)
        self._enriched.pop(filepath, None)
        self._by_game.pop(filepath, None)
        # Clear analysis entries for this filepath
        keys_to_drop = [k for k in self._analysis if k[0] == filepath]
        for k in keys_to_drop:
            del self._analysis[k]

    def load_odds(self, filename: Optional[str] = None) -> list[dict]:
        """Load odds, returning cached version if the file hasn't changed."""
        filepath = self._resolve(filename)
        if not filepath:
            return []
        if not self._is_valid(filepath):
            self._invalidate(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._raw[filepath] = data.get("odds", [])
            self._mtime[filepath] = os.path.getmtime(filepath)
        return self._raw[filepath]

    def load_enriched(self, filename: Optional[str] = None) -> list[dict]:
        """Return enriched odds (with vig/prob/fair odds), cached."""
        filepath = self._resolve(filename)
        if not filepath:
            return []
        # Ensure raw is loaded first (triggers invalidation if needed)
        raw = self.load_odds(filename)
        if filepath not in self._enriched:
            self._enriched[filepath] = [_enrich_record(r) for r in raw]
        return self._enriched[filepath]

    def load_by_game(self, filename: Optional[str] = None) -> dict[str, list[dict]]:
        """Return odds grouped by game_id, cached."""
        filepath = self._resolve(filename)
        if not filepath:
            return {}
        raw = self.load_odds(filename)
        if filepath not in self._by_game:
            self._by_game[filepath] = _group_by_game(raw)
        return self._by_game[filepath]

    def get_analysis(self, filename: Optional[str], key: str) -> Optional[dict]:
        """Retrieve a cached analysis result (e.g. vig, arb, ev)."""
        filepath = self._resolve(filename)
        if not filepath or not self._is_valid(filepath):
            return None
        return self._analysis.get((filepath, key))

    def set_analysis(self, filename: Optional[str], key: str, result: dict):
        """Store an analysis result in cache."""
        filepath = self._resolve(filename)
        if filepath:
            self._analysis[(filepath, key)] = result


_cache = _OddsCache()


def _load_odds(filename: Optional[str] = None) -> list[dict]:
    """Load odds records from a data file (cached)."""
    return _cache.load_odds(filename)


def _enrich_record(record: dict) -> dict:
    """Enrich a single odds record with implied probabilities, vig, and fair odds."""
    markets = record.get("markets", {})
    enriched_markets = {}

    for market_name, market in markets.items():
        if market_name == "spread":
            vig_info = calculate_vig(market["home_odds"], market["away_odds"])
            fair = no_vig_probabilities(market["home_odds"], market["away_odds"])
            enriched_markets["spread"] = {
                **market,
                "home_implied_prob": vig_info["implied_a"],
                "away_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
            }
        elif market_name == "moneyline":
            vig_info = calculate_vig(market["home_odds"], market["away_odds"])
            fair = no_vig_probabilities(market["home_odds"], market["away_odds"])
            enriched_markets["moneyline"] = {
                **market,
                "home_implied_prob": vig_info["implied_a"],
                "away_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
            }
        elif market_name == "total":
            vig_info = calculate_vig(market["over_odds"], market["under_odds"])
            fair = no_vig_probabilities(market["over_odds"], market["under_odds"])
            enriched_markets["total"] = {
                **market,
                "over_implied_prob": vig_info["implied_a"],
                "under_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "over_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "under_fair_odds": fair_odds_to_american(fair["fair_b"]),
            }

    return {**{k: v for k, v in record.items() if k != "markets"}, "markets": enriched_markets}


def _group_by_game(odds: list[dict]) -> dict[str, list[dict]]:
    """Group odds records by game_id."""
    games = {}
    for record in odds:
        gid = record.get("game_id", "unknown")
        games.setdefault(gid, []).append(record)
    return games


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Phase 1: Core Odds & Comparison
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def list_data_files() -> str:
    """List all available betting data files that can be loaded for analysis.

    Returns a JSON list of filenames in the data/ directory.
    """
    if not os.path.isdir(DATA_DIR):
        return json.dumps({"files": [], "error": "Data directory not found"})

    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    return json.dumps({"files": files, "count": len(files)})


@mcp.tool()
def list_events(filename: Optional[str] = None, sport: Optional[str] = None) -> str:
    """List all unique games/events in the dataset with basic info.

    Args:
        filename: Data file to load (optional, defaults to first available)
        sport: Filter by sport (e.g., "NBA"). Optional.

    Returns a JSON list of events with game_id, teams, sport, commence_time,
    and the number of sportsbooks offering odds.
    """
    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)

    events = []
    for game_id, records in games.items():
        first = records[0]
        if sport and first.get("sport", "").upper() != sport.upper():
            continue
        events.append({
            "game_id": game_id,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "commence_time": first.get("commence_time"),
            "sportsbook_count": len(records),
            "sportsbooks": [r["sportsbook"] for r in records],
        })

    events.sort(key=lambda e: e.get("commence_time", ""))
    return json.dumps({"events": events, "count": len(events)}, indent=2)


@mcp.tool()
def get_odds_comparison(game_id: str, market_type: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Compare odds across all sportsbooks for a specific game — side by side.

    This is the primary tool for understanding where the best and worst lines are.

    Args:
        game_id: The game identifier (e.g., "nba_20260320_lal_bos")
        market_type: Filter to a specific market: "spread", "moneyline", or "total". Optional (shows all).
        filename: Data file to load. Optional.

    Returns enriched odds from every sportsbook for the game, including
    implied probabilities, vig, and fair odds for each market.
    """
    all_enriched = _cache.load_enriched(filename)
    enriched = [r for r in all_enriched if r.get("game_id") == game_id]

    if not enriched:
        return json.dumps({"error": f"No records found for game_id: {game_id}"})

    # If market_type filter, create shallow copies to avoid mutating cached data
    if market_type:
        enriched = [
            {**r, "markets": {k: v for k, v in r["markets"].items() if k == market_type}}
            for r in enriched
        ]

    first = enriched[0]
    return json.dumps({
        "game_id": game_id,
        "sport": first.get("sport"),
        "home_team": first.get("home_team"),
        "away_team": first.get("away_team"),
        "commence_time": first.get("commence_time"),
        "sportsbook_count": len(enriched),
        "odds_by_book": enriched,
    }, indent=2)


@mcp.tool()
def get_best_odds(game_id: str, market_type: str, side: str, filename: Optional[str] = None) -> str:
    """Find the single best available odds for a specific bet across all sportsbooks.

    Use this to answer "where should I place this bet?" questions.

    Args:
        game_id: The game identifier (e.g., "nba_20260320_lal_bos")
        market_type: "spread", "moneyline", or "total"
        side: Which side of the bet:
              - spread/moneyline: "home" or "away"
              - total: "over" or "under"
        filename: Data file to load. Optional.

    Returns the best odds, which sportsbook offers them, and how they
    compare to the consensus/average.
    """
    odds = _load_odds(filename)
    game_records = [r for r in odds if r.get("game_id") == game_id]

    if not game_records:
        return json.dumps({"error": f"No records found for game_id: {game_id}"})

    # Determine the odds key based on market_type + side
    if market_type in ("spread", "moneyline"):
        odds_key = f"{side}_odds"
    elif market_type == "total":
        odds_key = f"{side}_odds"
    else:
        return json.dumps({"error": f"Unknown market_type: {market_type}"})

    candidates = []
    for r in game_records:
        market = r.get("markets", {}).get(market_type, {})
        if odds_key in market:
            candidates.append({
                "sportsbook": r["sportsbook"],
                "odds": market[odds_key],
                "last_updated": r.get("last_updated"),
            })

    if not candidates:
        return json.dumps({"error": f"No {market_type} {side} odds found for {game_id}"})

    # Best odds = highest American odds (more positive / less negative = better payout)
    candidates.sort(key=lambda c: c["odds"], reverse=True)
    best = candidates[0]
    worst = candidates[-1]

    # Compute average and edge
    avg_odds = sum(c["odds"] for c in candidates) / len(candidates)
    best_prob = implied_probability(best["odds"])
    avg_prob = implied_probability(avg_odds)

    first = game_records[0]
    extra_info = {}
    if market_type == "spread":
        extra_info["line"] = first["markets"].get("spread", {}).get(f"{side}_line",
                             first["markets"].get("spread", {}).get("home_line"))
    elif market_type == "total":
        extra_info["line"] = first["markets"].get("total", {}).get("line")

    return json.dumps({
        "game_id": game_id,
        "market_type": market_type,
        "side": side,
        **extra_info,
        "best": best,
        "worst": worst,
        "average_odds": round(avg_odds, 1),
        "edge_vs_average": f"{round((avg_prob - best_prob) * 100, 2)}%",
        "all_books": candidates,
        "context": f"Best {side} {market_type} odds at {best['sportsbook']} ({best['odds']}) vs worst at {worst['sportsbook']} ({worst['odds']}). Edge vs avg: {round((avg_prob - best_prob) * 100, 2)}%",
    }, indent=2)


@mcp.tool()
def get_worst_odds(game_id: str, market_type: str, side: str, filename: Optional[str] = None) -> str:
    """Find the worst available odds for a specific bet — useful for identifying books to avoid.

    Args:
        game_id: The game identifier
        market_type: "spread", "moneyline", or "total"
        side: "home", "away", "over", or "under"
        filename: Data file to load. Optional.

    Returns the worst odds and compares to the best available.
    """
    # Reuse get_best_odds logic — just flip the perspective
    odds = _load_odds(filename)
    game_records = [r for r in odds if r.get("game_id") == game_id]

    if not game_records:
        return json.dumps({"error": f"No records found for game_id: {game_id}"})

    if market_type in ("spread", "moneyline"):
        odds_key = f"{side}_odds"
    elif market_type == "total":
        odds_key = f"{side}_odds"
    else:
        return json.dumps({"error": f"Unknown market_type: {market_type}"})

    candidates = []
    for r in game_records:
        market = r.get("markets", {}).get(market_type, {})
        if odds_key in market:
            candidates.append({
                "sportsbook": r["sportsbook"],
                "odds": market[odds_key],
                "last_updated": r.get("last_updated"),
            })

    if not candidates:
        return json.dumps({"error": f"No {market_type} {side} odds found for {game_id}"})

    candidates.sort(key=lambda c: c["odds"])
    worst = candidates[0]
    best = candidates[-1]

    return json.dumps({
        "game_id": game_id,
        "market_type": market_type,
        "side": side,
        "worst": worst,
        "best": best,
        "spread_between_best_worst": best["odds"] - worst["odds"],
        "all_books_worst_to_best": candidates,
        "context": f"AVOID {worst['sportsbook']} for {side} {market_type} ({worst['odds']}). Best alternative: {best['sportsbook']} ({best['odds']}). Difference: {best['odds'] - worst['odds']} pts.",
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Phase 2: Value & Edge Detection
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_vig_analysis(game_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Analyze the vig (bookmaker margin) across all sportsbooks and markets.

    Lower vig = better for bettors. Use this to rank sportsbooks by fairness.

    Args:
        game_id: Filter to a specific game. Optional (analyzes all games).
        filename: Data file to load. Optional.

    Returns per-game, per-market vig rankings sorted tightest to widest,
    plus an overall sportsbook vig ranking.
    """
    odds = _load_odds(filename)
    if game_id:
        odds = [r for r in odds if r.get("game_id") == game_id]

    # Check cache first
    cache_key = f"vig:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    # Per-game, per-market vig
    vig_by_game = {}
    book_vig_totals = {}  # sportsbook -> list of vigs for overall ranking

    for gid, records in games.items():
        vig_by_game[gid] = {}
        for r in records:
            book = r["sportsbook"]
            for mkt_name, mkt in r.get("markets", {}).items():
                vig_val = mkt.get("vig", 0)
                vig_by_game[gid].setdefault(mkt_name, []).append({
                    "sportsbook": book,
                    "vig": vig_val,
                    "vig_pct": mkt.get("vig_pct", ""),
                })
                book_vig_totals.setdefault(book, []).append(vig_val)

        # Sort each market by vig ascending
        for mkt_name in vig_by_game[gid]:
            vig_by_game[gid][mkt_name].sort(key=lambda x: x["vig"])

    # Overall sportsbook ranking by average vig
    book_rankings = []
    for book, vigs in book_vig_totals.items():
        avg = sum(vigs) / len(vigs)
        book_rankings.append({
            "sportsbook": book,
            "average_vig": round(avg, 6),
            "average_vig_pct": f"{round(avg * 100, 2)}%",
            "markets_sampled": len(vigs),
        })
    book_rankings.sort(key=lambda x: x["average_vig"])

    result = {
        "vig_by_game": vig_by_game,
        "sportsbook_rankings": book_rankings,
        "best_book": book_rankings[0] if book_rankings else None,
        "worst_book": book_rankings[-1] if book_rankings else None,
        "context": f"Tightest margins: {book_rankings[0]['sportsbook']} ({book_rankings[0]['average_vig_pct']} avg vig). Widest: {book_rankings[-1]['sportsbook']} ({book_rankings[-1]['average_vig_pct']})." if book_rankings else "No data",
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def find_arbitrage_opportunities(filename: Optional[str] = None, min_profit_pct: float = 0.0) -> str:
    """Scan all games for arbitrage (arb) opportunities across sportsbooks.

    An arbitrage exists when you can bet both sides across different books
    and guarantee a profit regardless of outcome. This happens when the
    combined implied probability < 100%.

    Args:
        filename: Data file to load. Optional.
        min_profit_pct: Minimum profit % to report (default 0 = show all arbs). E.g., 1.0 = only arbs with 1%+ profit.

    Returns all arbitrage opportunities found, sorted by profit potential.
    """
    cache_key = f"arb:{min_profit_pct}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    arbs = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                side_a, side_b = "home", "away"
                key_a, key_b = f"{side_a}_odds", f"{side_b}_odds"
            else:
                side_a, side_b = "over", "under"
                key_a, key_b = "over_odds", "under_odds"

            # Collect best odds for each side
            best_a = {"odds": -99999, "book": ""}
            best_b = {"odds": -99999, "book": ""}

            for r in records:
                market = r.get("markets", {}).get(market_type, {})
                if key_a in market and market[key_a] > best_a["odds"]:
                    best_a = {"odds": market[key_a], "book": r["sportsbook"]}
                if key_b in market and market[key_b] > best_b["odds"]:
                    best_b = {"odds": market[key_b], "book": r["sportsbook"]}

            if best_a["book"] and best_b["book"]:
                prob_a = implied_probability(best_a["odds"])
                prob_b = implied_probability(best_b["odds"])
                total_implied = prob_a + prob_b

                if total_implied < 1.0:
                    profit_pct = round((1.0 - total_implied) * 100, 3)
                    if profit_pct >= min_profit_pct:
                        arbs.append({
                            "game_id": game_id,
                            "sport": first.get("sport"),
                            "home_team": first.get("home_team"),
                            "away_team": first.get("away_team"),
                            "market_type": market_type,
                            "side_a": {
                                "side": side_a,
                                "sportsbook": best_a["book"],
                                "odds": best_a["odds"],
                                "implied_prob": round(prob_a, 6),
                            },
                            "side_b": {
                                "side": side_b,
                                "sportsbook": best_b["book"],
                                "odds": best_b["odds"],
                                "implied_prob": round(prob_b, 6),
                            },
                            "combined_implied": round(total_implied, 6),
                            "profit_pct": profit_pct,
                            "context": f"ARB: Bet {side_a} at {best_a['book']} ({best_a['odds']}) + {side_b} at {best_b['book']} ({best_b['odds']}) = {profit_pct}% guaranteed profit",
                        })

    arbs.sort(key=lambda a: a["profit_pct"], reverse=True)
    result = {
        "arbitrage_opportunities": arbs,
        "count": len(arbs),
        "context": f"Found {len(arbs)} arbitrage opportunities." + (f" Best: {arbs[0]['context']}" if arbs else " None found — markets are efficient."),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def find_expected_value_bets(filename: Optional[str] = None, min_ev_pct: float = 0.0) -> str:
    """Find +EV (positive expected value) bets by comparing each book's odds to the consensus fair odds.

    The "consensus" is derived by averaging implied probabilities across all books
    and removing vig. A bet is +EV if the book's odds imply a lower probability
    than the consensus fair probability.

    Args:
        filename: Data file to load. Optional.
        min_ev_pct: Minimum EV % to report (default 0). E.g., 2.0 = only bets with 2%+ edge.

    Returns all +EV bets sorted by expected value.
    """
    cache_key = f"ev:{min_ev_pct}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    ev_bets = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds"), ("away", "away_odds")]
            else:
                sides = [("over", "over_odds"), ("under", "under_odds")]

            for side, odds_key in sides:
                all_odds = []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key in market:
                        all_odds.append({
                            "sportsbook": r["sportsbook"],
                            "odds": market[odds_key],
                            "last_updated": r.get("last_updated"),
                        })

                if len(all_odds) < 2:
                    continue

                # Compute consensus fair probability (average implied prob, normalized)
                probs = [implied_probability(o["odds"]) for o in all_odds]
                avg_prob = sum(probs) / len(probs)

                # Check each book for +EV
                for entry in all_odds:
                    book_prob = implied_probability(entry["odds"])
                    ev_edge = avg_prob - book_prob  # positive = +EV (book underestimates)

                    ev_pct = round(ev_edge * 100, 3)
                    if ev_pct > min_ev_pct:
                        ev_bets.append({
                            "game_id": game_id,
                            "sport": first.get("sport"),
                            "home_team": first.get("home_team"),
                            "away_team": first.get("away_team"),
                            "market_type": market_type,
                            "side": side,
                            "sportsbook": entry["sportsbook"],
                            "odds": entry["odds"],
                            "book_implied_prob": round(book_prob, 6),
                            "consensus_fair_prob": round(avg_prob, 6),
                            "ev_edge_pct": ev_pct,
                            "last_updated": entry.get("last_updated"),
                            "context": f"+EV: {side} {market_type} at {entry['sportsbook']} ({entry['odds']}). Book says {round(book_prob*100,1)}%, consensus says {round(avg_prob*100,1)}%. Edge: {ev_pct}%",
                        })

    ev_bets.sort(key=lambda b: b["ev_edge_pct"], reverse=True)
    result = {
        "ev_bets": ev_bets,
        "count": len(ev_bets),
        "context": f"Found {len(ev_bets)} +EV bets." + (f" Top: {ev_bets[0]['context']}" if ev_bets else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Phase 3: Anomaly Detection & Staleness
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def detect_stale_lines(filename: Optional[str] = None, stale_threshold_minutes: int = 30) -> str:
    """Detect sportsbook lines that haven't been updated recently (potentially stale).

    Stale lines may represent value opportunities if other books have moved.

    Args:
        filename: Data file to load. Optional.
        stale_threshold_minutes: How old a line must be (vs the newest for that game) to be considered stale. Default 30 minutes.

    Returns records where last_updated is significantly older than peers.
    """
    cache_key = f"stale:{stale_threshold_minutes}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    stale = []

    for game_id, records in games.items():
        # Parse timestamps
        timed = []
        for r in records:
            ts_str = r.get("last_updated", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = None
            timed.append({"record": r, "ts": ts})

        valid = [t for t in timed if t["ts"]]
        if len(valid) < 2:
            continue

        newest = max(t["ts"] for t in valid)

        for t in valid:
            age_minutes = (newest - t["ts"]).total_seconds() / 60
            if age_minutes >= stale_threshold_minutes:
                r = t["record"]
                stale.append({
                    "game_id": game_id,
                    "sportsbook": r["sportsbook"],
                    "last_updated": r.get("last_updated"),
                    "newest_update": newest.isoformat(),
                    "staleness_minutes": round(age_minutes, 1),
                    "sport": r.get("sport"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "markets": r.get("markets", {}),
                    "context": f"STALE: {r['sportsbook']} line for {r.get('away_team')} @ {r.get('home_team')} is {round(age_minutes)} min behind. May represent value if line has moved elsewhere.",
                })

    stale.sort(key=lambda s: s["staleness_minutes"], reverse=True)
    result = {
        "stale_lines": stale,
        "count": len(stale),
        "threshold_minutes": stale_threshold_minutes,
        "context": f"Found {len(stale)} stale lines (>{stale_threshold_minutes} min old)." + (f" Most stale: {stale[0]['context']}" if stale else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def detect_line_outliers(filename: Optional[str] = None, threshold_odds: int = 15) -> str:
    """Detect odds that are significant outliers compared to the consensus across books.

    An outlier is a line that deviates substantially from the average, which
    could indicate a pricing error, a sharp move, or stale data.

    Args:
        filename: Data file to load. Optional.
        threshold_odds: Minimum deviation from average (in American odds points) to flag. Default 15.

    Returns all outlier lines with deviation details.
    """
    cache_key = f"outliers:{threshold_odds}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    outliers = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                keys = ["home_odds", "away_odds"]
            else:
                keys = ["over_odds", "under_odds"]

            for odds_key in keys:
                values = []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key in market:
                        values.append({"sportsbook": r["sportsbook"], "odds": market[odds_key], "last_updated": r.get("last_updated")})

                if len(values) < 3:
                    continue

                avg = sum(v["odds"] for v in values) / len(values)

                for v in values:
                    deviation = abs(v["odds"] - avg)
                    if deviation >= threshold_odds:
                        side = odds_key.replace("_odds", "")
                        outliers.append({
                            "game_id": game_id,
                            "sport": first.get("sport"),
                            "home_team": first.get("home_team"),
                            "away_team": first.get("away_team"),
                            "market_type": market_type,
                            "side": side,
                            "sportsbook": v["sportsbook"],
                            "odds": v["odds"],
                            "consensus_avg": round(avg, 1),
                            "deviation": round(deviation, 1),
                            "last_updated": v.get("last_updated"),
                            "direction": "better_for_bettor" if v["odds"] > avg else "worse_for_bettor",
                            "context": f"OUTLIER: {v['sportsbook']} {side} {market_type} at {v['odds']} (avg {round(avg,1)}, dev {round(deviation,1)}). {'Potential value!' if v['odds'] > avg else 'Avoid — worse than market.'}",
                        })

    outliers.sort(key=lambda o: o["deviation"], reverse=True)
    result = {
        "outliers": outliers,
        "count": len(outliers),
        "threshold": threshold_odds,
        "context": f"Found {len(outliers)} outlier lines (>{threshold_odds} pts from consensus)." + (f" Biggest: {outliers[0]['context']}" if outliers else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Aggregation / Summary
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_market_summary(filename: Optional[str] = None) -> str:
    """Get a comprehensive market summary — a structured digest of the entire dataset.

    This is the best "start here" tool. It gives Claude a high-level overview:
    - Event count and sports covered
    - Sportsbook rankings by vig
    - Top +EV bets
    - Arbitrage opportunities
    - Stale lines
    - Biggest outliers

    Args:
        filename: Data file to load. Optional.

    Returns a structured JSON summary suitable for Claude to reason about.
    """
    cache_key = "market_summary"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    if not odds:
        return json.dumps({"error": "No odds data found"})

    games = _cache.load_by_game(filename)

    # Basics
    sports = list(set(r.get("sport", "Unknown") for r in odds))
    books = list(set(r.get("sportsbook", "Unknown") for r in odds))

    # Get results from other tools (parse the JSON they return)
    vig_data = json.loads(get_vig_analysis(filename))
    arb_data = json.loads(find_arbitrage_opportunities(filename))
    ev_data = json.loads(find_expected_value_bets(filename, min_ev_pct=1.0))
    stale_data = json.loads(detect_stale_lines(filename))
    outlier_data = json.loads(detect_line_outliers(filename))

    result = {
        "summary": {
            "total_records": len(odds),
            "unique_games": len(games),
            "sports": sports,
            "sportsbooks": books,
            "sportsbook_count": len(books),
        },
        "sportsbook_rankings": vig_data.get("sportsbook_rankings", []),
        "best_book": vig_data.get("best_book"),
        "worst_book": vig_data.get("worst_book"),
        "arbitrage_opportunities": {
            "count": arb_data.get("count", 0),
            "top_3": arb_data.get("arbitrage_opportunities", [])[:3],
        },
        "top_ev_bets": {
            "count": ev_data.get("count", 0),
            "top_5": ev_data.get("ev_bets", [])[:5],
        },
        "stale_lines": {
            "count": stale_data.get("count", 0),
            "top_3": stale_data.get("stale_lines", [])[:3],
        },
        "outliers": {
            "count": outlier_data.get("count", 0),
            "top_3": outlier_data.get("outliers", [])[:3],
        },
        "context": "Full market summary with sportsbook rankings, arb opportunities, +EV bets, stale lines, and outliers. Drill into specific tools for more detail.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def calculate_odds(american_odds: int) -> str:
    """Convert American odds to implied probability, decimal odds, and potential payout.

    Useful as a quick calculator for understanding any odds number.

    Args:
        american_odds: American odds (e.g., -110, +150, -200)
    """
    prob = implied_probability(american_odds)

    # Decimal odds
    if american_odds > 0:
        decimal_odds = (american_odds / 100) + 1
    elif american_odds < 0:
        decimal_odds = (100 / abs(american_odds)) + 1
    else:
        decimal_odds = 2.0

    # Payout on $100 bet
    if american_odds > 0:
        profit_on_100 = american_odds
    else:
        profit_on_100 = round(100 / abs(american_odds) * 100, 2)

    return json.dumps({
        "american_odds": american_odds,
        "implied_probability": round(prob, 6),
        "implied_probability_pct": f"{round(prob * 100, 2)}%",
        "decimal_odds": round(decimal_odds, 4),
        "profit_on_100_bet": profit_on_100,
        "total_return_on_100_bet": round(profit_on_100 + 100, 2),
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Phase 4: New Recommended Tools
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_best_bets_today(filename: Optional[str] = None, count: int = 10) -> str:
    """Get the top-N best bets right now, ranked by a composite value score.

    Combines +EV edge, low vig, outlier value, and line freshness into a single
    ranked list of actionable recommendations. This is the best tool for
    answering "what should I bet on?"

    Args:
        filename: Data file to load. Optional.
        count: Number of top bets to return. Default 10.

    Returns a ranked list of the best bets with reasoning for each.
    """
    cache_key = f"best_bets:{count}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)

    scored_bets = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds"), ("away", "away_odds")]
            else:
                sides = [("over", "over_odds"), ("under", "under_odds")]

            for side, odds_key in sides:
                all_entries = []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key in market:
                        all_entries.append({
                            "sportsbook": r["sportsbook"],
                            "odds": market[odds_key],
                            "last_updated": r.get("last_updated", ""),
                            "market": market,
                        })

                if len(all_entries) < 2:
                    continue

                # Consensus probability
                probs = [implied_probability(e["odds"]) for e in all_entries]
                avg_prob = sum(probs) / len(probs)

                # Find the best odds entry (highest American odds)
                best_entry = max(all_entries, key=lambda e: e["odds"])
                best_prob = implied_probability(best_entry["odds"])

                # EV edge
                ev_edge = (avg_prob - best_prob) * 100
                if ev_edge <= 0:
                    continue

                # Vig at the best book
                market = best_entry["market"]
                vig = market.get("vig", 0)
                if vig == 0:
                    # Calculate it
                    enriched = _enrich_record({"markets": {market_type: market}, "game_id": game_id, "sportsbook": best_entry["sportsbook"]})
                    vig = enriched["markets"].get(market_type, {}).get("vig", 0.05)

                # Outlier bonus: how far is the best line from average
                avg_odds = sum(e["odds"] for e in all_entries) / len(all_entries)
                outlier_bonus = max(0, (best_entry["odds"] - avg_odds) / 10)

                # Freshness penalty: if the best line is stale, penalize
                freshness_penalty = 0
                try:
                    timestamps = []
                    for e in all_entries:
                        ts_str = e.get("last_updated", "")
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            timestamps.append((e["sportsbook"], ts))
                    if timestamps:
                        newest = max(t[1] for t in timestamps)
                        best_ts = next((t[1] for t in timestamps if t[0] == best_entry["sportsbook"]), newest)
                        staleness_min = (newest - best_ts).total_seconds() / 60
                        if staleness_min > 60:
                            freshness_penalty = min(staleness_min / 100, 3)
                except (ValueError, AttributeError):
                    pass

                # Composite score: EV edge (primary) + outlier bonus - vig penalty - freshness penalty
                composite_score = ev_edge + outlier_bonus - (vig * 20) - freshness_penalty

                extra_info = {}
                if market_type == "spread":
                    extra_info["line"] = first["markets"].get("spread", {}).get("home_line")
                elif market_type == "total":
                    extra_info["line"] = first["markets"].get("total", {}).get("line")

                scored_bets.append({
                    "game_id": game_id,
                    "sport": first.get("sport"),
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market_type": market_type,
                    "side": side,
                    **extra_info,
                    "sportsbook": best_entry["sportsbook"],
                    "odds": best_entry["odds"],
                    "ev_edge_pct": round(ev_edge, 3),
                    "vig_at_book": round(vig, 6),
                    "composite_score": round(composite_score, 3),
                    "reasons": [],
                })

                # Build reasoning
                bet = scored_bets[-1]
                bet["reasons"].append(f"+EV edge: {round(ev_edge, 2)}% vs consensus")
                if outlier_bonus > 0.5:
                    bet["reasons"].append(f"Outlier value: {round(outlier_bonus, 1)} pts better than avg")
                if vig < 0.04:
                    bet["reasons"].append(f"Low vig: {round(vig*100, 2)}%")
                if freshness_penalty > 0:
                    bet["reasons"].append(f"Caution: line may be stale ({round(freshness_penalty * 100 / 3, 0)} min old)")

    scored_bets.sort(key=lambda b: b["composite_score"], reverse=True)
    top = scored_bets[:count]

    result = {
        "best_bets": top,
        "count": len(top),
        "total_candidates": len(scored_bets),
        "context": f"Top {len(top)} bets by composite score (EV + outlier value - vig - staleness)." + (f" #1: {top[0]['side']} {top[0]['market_type']} at {top[0]['sportsbook']} ({top[0]['odds']}) — {', '.join(top[0]['reasons'])}" if top else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def find_middle_opportunities(filename: Optional[str] = None) -> str:
    """Find middling opportunities where spread/total lines differ enough across books
    that you could bet both sides and potentially win BOTH bets.

    A middle exists when Book A has Team -3.5 and Book B has Team +4.5 — if the
    game lands on 4, you win both bets.

    Args:
        filename: Data file to load. Optional.

    Returns all middle opportunities with the window size and which books to use.
    """
    cache_key = "middles"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    middles = []

    for game_id, records in games.items():
        first = records[0]

        # Check spreads — middle exists when home_line varies across books
        spread_books = []
        for r in records:
            spread = r.get("markets", {}).get("spread", {})
            if "home_line" in spread:
                spread_books.append({
                    "sportsbook": r["sportsbook"],
                    "home_line": spread["home_line"],
                    "home_odds": spread["home_odds"],
                    "away_line": spread.get("away_line", -spread["home_line"]),
                    "away_odds": spread["away_odds"],
                })

        if len(spread_books) >= 2:
            # Find the most negative home line (biggest favorite spread) and most positive away line
            most_neg_home = min(spread_books, key=lambda b: b["home_line"])
            most_pos_away = max(spread_books, key=lambda b: b["away_line"])

            window = most_pos_away["away_line"] + most_neg_home["home_line"]
            if window > 0 and most_neg_home["sportsbook"] != most_pos_away["sportsbook"]:
                middles.append({
                    "game_id": game_id,
                    "sport": first.get("sport"),
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market_type": "spread",
                    "leg_a": {
                        "side": "home",
                        "sportsbook": most_neg_home["sportsbook"],
                        "line": most_neg_home["home_line"],
                        "odds": most_neg_home["home_odds"],
                    },
                    "leg_b": {
                        "side": "away",
                        "sportsbook": most_pos_away["sportsbook"],
                        "line": most_pos_away["away_line"],
                        "odds": most_pos_away["away_odds"],
                    },
                    "middle_window": window,
                    "context": f"MIDDLE: Bet {first.get('home_team')} {most_neg_home['home_line']} at {most_neg_home['sportsbook']} + {first.get('away_team')} +{most_pos_away['away_line']} at {most_pos_away['sportsbook']}. Window: {window} pts.",
                })

        # Check totals — middle exists when line varies
        total_books = []
        for r in records:
            total = r.get("markets", {}).get("total", {})
            if "line" in total:
                total_books.append({
                    "sportsbook": r["sportsbook"],
                    "line": total["line"],
                    "over_odds": total["over_odds"],
                    "under_odds": total["under_odds"],
                })

        if len(total_books) >= 2:
            highest_line = max(total_books, key=lambda b: b["line"])
            lowest_line = min(total_books, key=lambda b: b["line"])

            window = highest_line["line"] - lowest_line["line"]
            if window > 0 and highest_line["sportsbook"] != lowest_line["sportsbook"]:
                middles.append({
                    "game_id": game_id,
                    "sport": first.get("sport"),
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market_type": "total",
                    "leg_a": {
                        "side": "under",
                        "sportsbook": highest_line["sportsbook"],
                        "line": highest_line["line"],
                        "odds": highest_line["under_odds"],
                    },
                    "leg_b": {
                        "side": "over",
                        "sportsbook": lowest_line["sportsbook"],
                        "line": lowest_line["line"],
                        "odds": lowest_line["over_odds"],
                    },
                    "middle_window": window,
                    "context": f"MIDDLE: Under {highest_line['line']} at {highest_line['sportsbook']} + Over {lowest_line['line']} at {lowest_line['sportsbook']}. Window: {window} pts.",
                })

    middles.sort(key=lambda m: m["middle_window"], reverse=True)
    result = {
        "middle_opportunities": middles,
        "count": len(middles),
        "context": f"Found {len(middles)} middle opportunities." + (f" Best: {middles[0]['context']}" if middles else " Lines are too tight for middles."),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_book_rankings(filename: Optional[str] = None) -> str:
    """Rank sportsbooks across multiple quality metrics — a comprehensive book report card.

    Metrics scored:
    - Average vig (lower = better)
    - How often the book has the best odds for any bet
    - How often the book has the worst odds
    - Average line staleness
    - Outlier frequency (positive and negative)

    Args:
        filename: Data file to load. Optional.

    Returns a ranked list of sportsbooks with scores and grades.
    """
    cache_key = "book_rankings"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)

    book_stats = {}  # sportsbook -> stats

    enriched_by_game = _group_by_game(_cache.load_enriched(filename))

    for game_id, enriched in enriched_by_game.items():
        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                keys = ["home_odds", "away_odds"]
            else:
                keys = ["over_odds", "under_odds"]

            for odds_key in keys:
                values = []
                for r in enriched:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key in market:
                        book = r["sportsbook"]
                        if book not in book_stats:
                            book_stats[book] = {
                                "vigs": [],
                                "best_count": 0,
                                "worst_count": 0,
                                "total_markets": 0,
                                "staleness_minutes": [],
                            }
                        values.append({"book": book, "odds": market[odds_key], "vig": market.get("vig", 0)})

                if len(values) < 2:
                    continue

                best_book = max(values, key=lambda v: v["odds"])["book"]
                worst_book = min(values, key=lambda v: v["odds"])["book"]

                for v in values:
                    book_stats[v["book"]]["total_markets"] += 1
                    book_stats[v["book"]]["vigs"].append(v["vig"])

                book_stats[best_book]["best_count"] += 1
                book_stats[worst_book]["worst_count"] += 1

    # Calculate staleness per book
    for game_id, records in games.items():
        timestamps = []
        for r in records:
            ts_str = r.get("last_updated", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                timestamps.append((r["sportsbook"], ts))
            except (ValueError, AttributeError):
                pass

        if timestamps:
            newest = max(t[1] for t in timestamps)
            for book, ts in timestamps:
                if book in book_stats:
                    staleness = (newest - ts).total_seconds() / 60
                    book_stats[book]["staleness_minutes"].append(staleness)

    # Build rankings
    rankings = []
    for book, stats in book_stats.items():
        avg_vig = sum(stats["vigs"]) / len(stats["vigs"]) if stats["vigs"] else 0.05
        avg_stale = sum(stats["staleness_minutes"]) / len(stats["staleness_minutes"]) if stats["staleness_minutes"] else 0
        total = stats["total_markets"]
        best_pct = (stats["best_count"] / total * 100) if total else 0
        worst_pct = (stats["worst_count"] / total * 100) if total else 0

        # Composite score: lower = better
        # Weighted: vig (35%), worst odds frequency (25%), staleness (25%), best odds bonus (-15%)
        score = (avg_vig * 100 * 0.35) + (worst_pct * 0.25) + (min(avg_stale, 500) / 500 * 25) - (best_pct * 0.15)

        # Grade — calibrated for real sportsbook data (vig ~3-6%, staleness varies)
        if score < 2.5:
            grade = "A"
        elif score < 4.0:
            grade = "B"
        elif score < 5.5:
            grade = "C"
        elif score < 7.0:
            grade = "D"
        else:
            grade = "F"

        rankings.append({
            "sportsbook": book,
            "grade": grade,
            "composite_score": round(score, 3),
            "avg_vig_pct": f"{round(avg_vig * 100, 2)}%",
            "best_odds_pct": f"{round(best_pct, 1)}%",
            "worst_odds_pct": f"{round(worst_pct, 1)}%",
            "avg_staleness_min": round(avg_stale, 1),
            "markets_sampled": total,
        })

    rankings.sort(key=lambda r: r["composite_score"])

    result = {
        "rankings": rankings,
        "best_book": rankings[0] if rankings else None,
        "worst_book": rankings[-1] if rankings else None,
        "context": f"Book rankings by composite score (vig, odds quality, freshness). Best: {rankings[0]['sportsbook']} ({rankings[0]['grade']}). Worst: {rankings[-1]['sportsbook']} ({rankings[-1]['grade']})." if rankings else "No data",
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_daily_digest(filename: Optional[str] = None) -> str:
    """Generate a structured daily digest optimized for Claude to reason about.

    Organized into clear sections: must-bet opportunities, lines to avoid,
    interesting situations, and sportsbook grades. Each item includes
    actionable context.

    Args:
        filename: Data file to load. Optional.

    Returns a structured digest with sections and action items.
    """
    cache_key = "daily_digest"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    # Gather data from other tools (these will hit cache after first call)
    summary_data = json.loads(get_market_summary(filename))
    best_bets_data = json.loads(get_best_bets_today(filename, count=5))
    middles_data = json.loads(find_middle_opportunities(filename))
    rankings_data = json.loads(get_book_rankings(filename))

    # Categorize
    must_bet = []
    for bet in best_bets_data.get("best_bets", []):
        if bet.get("composite_score", 0) > 2.0:
            must_bet.append({
                "action": f"Bet {bet['side']} {bet['market_type']} at {bet['sportsbook']}",
                "game": f"{bet.get('away_team', '?')} @ {bet.get('home_team', '?')}",
                "odds": bet["odds"],
                "ev_edge": f"{bet['ev_edge_pct']}%",
                "reasons": bet.get("reasons", []),
            })

    avoid = []
    for stale in summary_data.get("stale_lines", {}).get("top_3", []):
        avoid.append({
            "action": f"Avoid {stale['sportsbook']} for {stale.get('away_team', '?')} @ {stale.get('home_team', '?')}",
            "reason": f"Line is {stale.get('staleness_minutes', 0)} minutes stale",
        })
    for outlier in summary_data.get("outliers", {}).get("top_3", []):
        if outlier.get("direction") == "worse_for_bettor":
            avoid.append({
                "action": f"Avoid {outlier['sportsbook']} {outlier['side']} {outlier['market_type']}",
                "game": f"{outlier.get('away_team', '?')} @ {outlier.get('home_team', '?')}",
                "reason": f"Outlier: {outlier['odds']} vs consensus {outlier['consensus_avg']}",
            })

    interesting = []
    for arb in summary_data.get("arbitrage_opportunities", {}).get("top_3", []):
        interesting.append({
            "type": "arbitrage",
            "description": arb.get("context", ""),
            "profit_pct": arb.get("profit_pct"),
        })
    for mid in middles_data.get("middle_opportunities", [])[:3]:
        interesting.append({
            "type": "middle",
            "description": mid.get("context", ""),
            "window": mid.get("middle_window"),
        })

    book_grades = {}
    for r in rankings_data.get("rankings", []):
        book_grades[r["sportsbook"]] = {
            "grade": r["grade"],
            "avg_vig": r["avg_vig_pct"],
            "best_odds_freq": r["best_odds_pct"],
        }

    result = {
        "digest": {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "games_covered": summary_data.get("summary", {}).get("unique_games", 0),
            "books_covered": summary_data.get("summary", {}).get("sportsbook_count", 0),
        },
        "must_bet": must_bet,
        "avoid": avoid,
        "interesting": interesting,
        "book_grades": book_grades,
        "context": f"Daily digest: {len(must_bet)} must-bet opps, {len(avoid)} lines to avoid, {len(interesting)} interesting situations across {summary_data.get('summary', {}).get('unique_games', 0)} games.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# RESOURCES — Context Data
# ═══════════════════════════════════════════════════════════════════════════


@mcp.resource("betstamp://data/{filename}")
def get_raw_data(filename: str) -> str:
    """Access raw odds data from a file."""
    odds = _load_odds(filename)
    return json.dumps({"odds": odds, "count": len(odds)}, indent=2)


@mcp.resource("betstamp://glossary")
def get_glossary() -> str:
    """Betting terms glossary for Claude to reference when reasoning about data."""
    return json.dumps({
        "terms": {
            "vig": "Vigorish (juice) — the bookmaker's margin built into odds. Lower vig = fairer odds for bettors. Typical range: 2-8%.",
            "implied_probability": "The probability of an outcome as implied by the odds. Includes the vig, so raw implied probs sum to >100%.",
            "fair_odds": "Odds with the vig mathematically removed. Represents the 'true' probability the market assigns.",
            "spread": "Point spread — a handicap applied to the favorite. Betting the spread means the team must win by more than the spread.",
            "moneyline": "A straight-up bet on which team wins. No spread involved.",
            "total": "Over/under — a bet on whether the combined score exceeds or falls below a set number.",
            "+EV": "Positive Expected Value — a bet where the true probability of winning exceeds what the odds imply. Long-term profitable.",
            "arbitrage": "Betting both sides across different sportsbooks to guarantee profit regardless of outcome. Exists when combined implied probability < 100%.",
            "stale_line": "A line that hasn't been updated recently while other books have moved. May represent a value opportunity.",
            "closing_line_value": "Whether your bet was placed at better odds than the final closing line. The gold standard metric of sharp betting.",
            "steam_move": "A sudden, sharp line movement typically driven by professional/sharp bettors.",
            "reverse_line_movement": "When the line moves in the opposite direction of where the public is betting — often a sign of sharp action.",
            "hold": "The sportsbook's total margin across all bets on a market. Similar to vig but applied to the overall market.",
        }
    }, indent=2)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
