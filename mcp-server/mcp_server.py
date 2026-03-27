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
import random
from datetime import datetime, timezone
from math import sqrt, log, inf
from statistics import pstdev, mean
from typing import Optional

# Allow imports from sibling webservice/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))

from mcp.server.fastmcp import FastMCP
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, fair_odds_to_american, kelly_criterion

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
        self._consensus: dict[str, dict] = {}           # filepath -> {game_id: {market: consensus}}
        self._sharp_vs_crowd: dict[str, dict] = {}     # filepath -> {game_id: {market: sharp_vs_crowd}}
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
        self._consensus.pop(filepath, None)
        self._sharp_vs_crowd.pop(filepath, None)
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
        """Return enriched odds (with vig/prob/fair odds + consensus), cached."""
        filepath = self._resolve(filename)
        if not filepath:
            return []
        # Ensure raw is loaded first (triggers invalidation if needed)
        raw = self.load_odds(filename)
        if filepath not in self._enriched:
            enriched = [_enrich_record(r) for r in raw]
            # Attach consensus (market average) data to each record
            consensus = self.load_consensus(filename)
            sharp_vs_crowd = self.load_sharp_vs_crowd(filename)
            for record in enriched:
                game_id = record.get("game_id", "unknown")
                game_consensus = consensus.get(game_id, {})
                game_svc = sharp_vs_crowd.get(game_id, {})
                for market_name, market in record.get("markets", {}).items():
                    if market_name in game_consensus:
                        market["consensus"] = game_consensus[market_name]
                    if market_name in game_svc:
                        market["sharp_vs_crowd"] = game_svc[market_name]
            self._enriched[filepath] = enriched
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

    def load_consensus(self, filename: Optional[str] = None) -> dict[str, dict]:
        """Return consensus (market average) data per game per market, cached.

        Structure: {game_id: {market_type: {field: avg_value, ...}}}
        For each game & market, averages the line and odds across all sportsbooks.
        """
        filepath = self._resolve(filename)
        if not filepath:
            return {}
        by_game = self.load_by_game(filename)
        if filepath not in self._consensus:
            self._consensus[filepath] = _compute_consensus(by_game)
        return self._consensus[filepath]

    def load_sharp_vs_crowd(self, filename: Optional[str] = None) -> dict[str, dict]:
        """Return sharp (Pinnacle) vs crowd (all books) comparison per game per market, cached.

        Structure: {game_id: {market_type: {crowd_*, sharp_*, divergence_*, ...}}}
        """
        filepath = self._resolve(filename)
        if not filepath:
            return {}
        by_game = self.load_by_game(filename)
        if filepath not in self._sharp_vs_crowd:
            self._sharp_vs_crowd[filepath] = _compute_sharp_vs_crowd(by_game)
        return self._sharp_vs_crowd[filepath]

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


# ---------------------------------------------------------------------------
# Pinnacle fair-probability helper
# ---------------------------------------------------------------------------

SHARP_BOOK = "Pinnacle"  # Used as "true" probability source for Kelly sizing


def _get_pinnacle_fair_probs(games: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """Extract Pinnacle's no-vig fair probabilities for every game & market.

    Falls back to consensus (average across all books) when Pinnacle data is
    unavailable for a game/market.

    Returns
    -------
    {game_id: {market_type: {"side_a_prob": float, "side_b_prob": float, "source": str}}}
    where side_a/side_b depend on market type:
      - spread/moneyline: side_a = home, side_b = away
      - total: side_a = over, side_b = under
    """
    result: dict[str, dict[str, dict]] = {}

    for game_id, records in games.items():
        result[game_id] = {}

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                odds_key_a, odds_key_b = "home_odds", "away_odds"
            else:
                odds_key_a, odds_key_b = "over_odds", "under_odds"

            # Try Pinnacle first
            pinnacle_record = None
            all_probs_a, all_probs_b = [], []

            for r in records:
                market = r.get("markets", {}).get(market_type, {})
                if odds_key_a not in market or odds_key_b not in market:
                    continue

                prob_a = implied_probability(market[odds_key_a])
                prob_b = implied_probability(market[odds_key_b])
                all_probs_a.append(prob_a)
                all_probs_b.append(prob_b)

                if r.get("sportsbook", "").lower() == SHARP_BOOK.lower():
                    pinnacle_record = market

            if pinnacle_record is not None:
                # Use Pinnacle no-vig
                fair = no_vig_probabilities(
                    pinnacle_record[odds_key_a],
                    pinnacle_record[odds_key_b],
                )
                result[game_id][market_type] = {
                    "side_a_prob": fair["fair_a"],
                    "side_b_prob": fair["fair_b"],
                    "source": SHARP_BOOK,
                }
            elif all_probs_a:
                # Fallback: consensus average, normalized
                avg_a = sum(all_probs_a) / len(all_probs_a)
                avg_b = sum(all_probs_b) / len(all_probs_b)
                total = avg_a + avg_b
                result[game_id][market_type] = {
                    "side_a_prob": round(avg_a / total, 6) if total else 0.5,
                    "side_b_prob": round(avg_b / total, 6) if total else 0.5,
                    "source": "consensus",
                }

    return result


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


def _compute_consensus(by_game: dict[str, list[dict]]) -> dict[str, dict]:
    """Compute consensus (market average) line and odds across all sportsbooks per game & market.

    Returns: {game_id: {market_type: {field: avg, ...}, ...}, ...}
    """
    consensus: dict[str, dict] = {}

    for game_id, records in by_game.items():
        game_consensus: dict[str, dict] = {}

        # --- Spread ---
        spread_data = [
            r["markets"]["spread"]
            for r in records
            if "spread" in r.get("markets", {})
        ]
        if spread_data:
            home_lines = [s["home_line"] for s in spread_data if "home_line" in s]
            home_odds  = [s["home_odds"] for s in spread_data if "home_odds" in s]
            away_odds  = [s["away_odds"] for s in spread_data if "away_odds" in s]
            game_consensus["spread"] = {
                "avg_home_line": round(sum(home_lines) / len(home_lines), 2) if home_lines else None,
                "avg_away_line": round(-sum(home_lines) / len(home_lines), 2) if home_lines else None,
                "avg_home_odds": round(sum(home_odds) / len(home_odds), 1) if home_odds else None,
                "avg_away_odds": round(sum(away_odds) / len(away_odds), 1) if away_odds else None,
                "std_home_line": round(pstdev(home_lines), 3) if len(home_lines) > 1 else 0.0,
                "std_home_odds": round(pstdev(home_odds), 2) if len(home_odds) > 1 else 0.0,
                "std_away_odds": round(pstdev(away_odds), 2) if len(away_odds) > 1 else 0.0,
                "book_count": len(spread_data),
            }

        # --- Moneyline ---
        ml_data = [
            r["markets"]["moneyline"]
            for r in records
            if "moneyline" in r.get("markets", {})
        ]
        if ml_data:
            home_odds = [m["home_odds"] for m in ml_data if "home_odds" in m]
            away_odds = [m["away_odds"] for m in ml_data if "away_odds" in m]
            game_consensus["moneyline"] = {
                "avg_home_odds": round(sum(home_odds) / len(home_odds), 1) if home_odds else None,
                "avg_away_odds": round(sum(away_odds) / len(away_odds), 1) if away_odds else None,
                "std_home_odds": round(pstdev(home_odds), 2) if len(home_odds) > 1 else 0.0,
                "std_away_odds": round(pstdev(away_odds), 2) if len(away_odds) > 1 else 0.0,
                "book_count": len(ml_data),
            }

        # --- Total ---
        total_data = [
            r["markets"]["total"]
            for r in records
            if "total" in r.get("markets", {})
        ]
        if total_data:
            lines      = [t["line"] for t in total_data if "line" in t]
            over_odds  = [t["over_odds"] for t in total_data if "over_odds" in t]
            under_odds = [t["under_odds"] for t in total_data if "under_odds" in t]
            game_consensus["total"] = {
                "avg_line": round(sum(lines) / len(lines), 2) if lines else None,
                "avg_over_odds": round(sum(over_odds) / len(over_odds), 1) if over_odds else None,
                "avg_under_odds": round(sum(under_odds) / len(under_odds), 1) if under_odds else None,
                "std_line": round(pstdev(lines), 3) if len(lines) > 1 else 0.0,
                "std_over_odds": round(pstdev(over_odds), 2) if len(over_odds) > 1 else 0.0,
                "std_under_odds": round(pstdev(under_odds), 2) if len(under_odds) > 1 else 0.0,
                "book_count": len(total_data),
            }

        consensus[game_id] = game_consensus

    return consensus


# ---------------------------------------------------------------------------
# Sharp vs. Crowd — Pinnacle (sharp book) vs all-book average (crowd wisdom)
# ---------------------------------------------------------------------------


def _compute_sharp_vs_crowd(by_game: dict[str, list[dict]]) -> dict[str, dict]:
    """Compare Pinnacle's no-vig implied probabilities against the crowd average.

    For each game and market, computes:
      - crowd_fair_*   : average no-vig prob across ALL books (wisdom of crowds)
      - sharp_fair_*   : Pinnacle's no-vig prob (sharp wisdom)
      - divergence_pct : absolute difference in probability points (* 100)
      - mispriced_side : which side the crowd is mispricing (if divergence is notable)

    Returns: {game_id: {market_type: {crowd, sharp, divergence, ...}}}
    """
    result: dict[str, dict] = {}

    for game_id, records in by_game.items():
        game_result: dict[str, dict] = {}

        sharp_records = [r for r in records if r.get("sportsbook", "").lower() == SHARP_BOOK.lower()]
        crowd_records = records  # all books including Pinnacle for the crowd average

        for market_type in ("spread", "moneyline", "total"):
            # Gather crowd data (all books)
            crowd_markets = [
                r["markets"][market_type]
                for r in crowd_records
                if market_type in r.get("markets", {})
            ]
            # Gather sharp data (Pinnacle only)
            sharp_markets = [
                r["markets"][market_type]
                for r in sharp_records
                if market_type in r.get("markets", {})
            ]

            if not crowd_markets or not sharp_markets:
                continue

            sharp_m = sharp_markets[0]  # only one Pinnacle record per game

            if market_type in ("spread", "moneyline"):
                # --- Crowd average no-vig probs ---
                crowd_home_probs = []
                crowd_away_probs = []
                for m in crowd_markets:
                    home_odds = m.get("home_odds")
                    away_odds = m.get("away_odds")
                    if home_odds is not None and away_odds is not None:
                        fair = no_vig_probabilities(home_odds, away_odds)
                        crowd_home_probs.append(fair["fair_a"])
                        crowd_away_probs.append(fair["fair_b"])

                if not crowd_home_probs:
                    continue

                crowd_home = sum(crowd_home_probs) / len(crowd_home_probs)
                crowd_away = sum(crowd_away_probs) / len(crowd_away_probs)

                # --- Sharp (Pinnacle) no-vig probs ---
                sharp_fair = no_vig_probabilities(sharp_m["home_odds"], sharp_m["away_odds"])
                sharp_home = sharp_fair["fair_a"]
                sharp_away = sharp_fair["fair_b"]

                # --- Divergence ---
                div_home = abs(sharp_home - crowd_home)
                div_away = abs(sharp_away - crowd_away)
                max_div = max(div_home, div_away)

                # Determine which side the crowd is mispricing
                mispriced_side = None
                mispriced_direction = None
                if max_div >= 0.01:  # >= 1 pp divergence threshold
                    if div_home >= div_away:
                        mispriced_side = "home"
                        mispriced_direction = "crowd overvalues home" if crowd_home > sharp_home else "crowd undervalues home"
                    else:
                        mispriced_side = "away"
                        mispriced_direction = "crowd overvalues away" if crowd_away > sharp_away else "crowd undervalues away"

                entry: dict = {
                    "crowd_home_fair_prob": round(crowd_home, 6),
                    "crowd_away_fair_prob": round(crowd_away, 6),
                    "sharp_home_fair_prob": round(sharp_home, 6),
                    "sharp_away_fair_prob": round(sharp_away, 6),
                    "divergence_home_pct": round(div_home * 100, 2),
                    "divergence_away_pct": round(div_away * 100, 2),
                    "max_divergence_pct": round(max_div * 100, 2),
                    "crowd_books": len(crowd_home_probs),
                    "sharp_book": SHARP_BOOK,
                }
                if market_type == "spread":
                    crowd_lines = [m.get("home_line", 0) for m in crowd_markets if "home_line" in m]
                    entry["crowd_consensus_line"] = round(sum(crowd_lines) / len(crowd_lines), 1) if crowd_lines else None
                    entry["sharp_line"] = sharp_m.get("home_line")
                if mispriced_side:
                    entry["mispriced_side"] = mispriced_side
                    entry["mispriced_direction"] = mispriced_direction

                game_result[market_type] = entry

            else:  # total
                crowd_over_probs = []
                crowd_under_probs = []
                for m in crowd_markets:
                    over_odds = m.get("over_odds")
                    under_odds = m.get("under_odds")
                    if over_odds is not None and under_odds is not None:
                        fair = no_vig_probabilities(over_odds, under_odds)
                        crowd_over_probs.append(fair["fair_a"])
                        crowd_under_probs.append(fair["fair_b"])

                if not crowd_over_probs:
                    continue

                crowd_over = sum(crowd_over_probs) / len(crowd_over_probs)
                crowd_under = sum(crowd_under_probs) / len(crowd_under_probs)

                sharp_fair = no_vig_probabilities(sharp_m["over_odds"], sharp_m["under_odds"])
                sharp_over = sharp_fair["fair_a"]
                sharp_under = sharp_fair["fair_b"]

                div_over = abs(sharp_over - crowd_over)
                div_under = abs(sharp_under - crowd_under)
                max_div = max(div_over, div_under)

                mispriced_side = None
                mispriced_direction = None
                if max_div >= 0.01:
                    if div_over >= div_under:
                        mispriced_side = "over"
                        mispriced_direction = "crowd overvalues over" if crowd_over > sharp_over else "crowd undervalues over"
                    else:
                        mispriced_side = "under"
                        mispriced_direction = "crowd overvalues under" if crowd_under > sharp_under else "crowd undervalues under"

                crowd_lines = [m.get("line", 0) for m in crowd_markets if "line" in m]
                entry = {
                    "crowd_over_fair_prob": round(crowd_over, 6),
                    "crowd_under_fair_prob": round(crowd_under, 6),
                    "sharp_over_fair_prob": round(sharp_over, 6),
                    "sharp_under_fair_prob": round(sharp_under, 6),
                    "divergence_over_pct": round(div_over * 100, 2),
                    "divergence_under_pct": round(div_under * 100, 2),
                    "max_divergence_pct": round(max_div * 100, 2),
                    "crowd_consensus_line": round(sum(crowd_lines) / len(crowd_lines), 1) if crowd_lines else None,
                    "sharp_line": sharp_m.get("line"),
                    "crowd_books": len(crowd_over_probs),
                    "sharp_book": SHARP_BOOK,
                }
                if mispriced_side:
                    entry["mispriced_side"] = mispriced_side
                    entry["mispriced_direction"] = mispriced_direction

                game_result["total"] = entry

        result[game_id] = game_result

    return result


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
def get_hold_percentage(filename: Optional[str] = None) -> str:
    """Calculate hold percentage by sportsbook - the overall margin a book keeps.

    Hold percentage measures how much a sportsbook retains from every dollar
    wagered. Lower hold = fairer pricing for bettors. Broken down by market
    type (spread, moneyline, total) and aggregated into an overall hold.

    Hold pct per market = (sum of implied probabilities for both sides - 1) x 100.
    Sportsbook hold = average hold across all markets offered by that book.

    Args:
        filename: Data file to load. Optional.

    Returns per-sportsbook hold percentages ranked from lowest (best) to highest.
    """
    cache_key = "hold_percentage"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No odds data found"})

    # Collect vig (= hold) per sportsbook per market type
    book_holds: dict[str, dict[str, list[float]]] = {}

    for record in enriched:
        book = record["sportsbook"]
        if book not in book_holds:
            book_holds[book] = {"spread": [], "moneyline": [], "total": [], "_all": []}

        for market_name, market in record.get("markets", {}).items():
            hold_val = market.get("vig", 0)
            if market_name in book_holds[book]:
                book_holds[book][market_name].append(hold_val)
            book_holds[book]["_all"].append(hold_val)

    # Build per-sportsbook hold report
    hold_report = []
    for book, markets in book_holds.items():
        all_holds = markets["_all"]
        if not all_holds:
            continue

        overall_hold = sum(all_holds) / len(all_holds)

        entry = {
            "sportsbook": book,
            "overall_hold_pct": f"{round(overall_hold * 100, 2)}%",
            "overall_hold_raw": round(overall_hold, 6),
            "markets_sampled": len(all_holds),
            "by_market": {},
        }

        for mkt in ("spread", "moneyline", "total"):
            vals = markets.get(mkt, [])
            if vals:
                avg = sum(vals) / len(vals)
                entry["by_market"][mkt] = {
                    "hold_pct": f"{round(avg * 100, 2)}%",
                    "hold_raw": round(avg, 6),
                    "sample_count": len(vals),
                    "min_hold_pct": f"{round(min(vals) * 100, 2)}%",
                    "max_hold_pct": f"{round(max(vals) * 100, 2)}%",
                }

        hold_report.append(entry)

    hold_report.sort(key=lambda x: x["overall_hold_raw"])

    all_holds_flat = [h["overall_hold_raw"] for h in hold_report]
    avg_market_hold = sum(all_holds_flat) / len(all_holds_flat) if all_holds_flat else 0

    result = {
        "hold_by_sportsbook": hold_report,
        "lowest_hold": hold_report[0] if hold_report else None,
        "highest_hold": hold_report[-1] if hold_report else None,
        "market_average_hold_pct": f"{round(avg_market_hold * 100, 2)}%",
        "book_count": len(hold_report),
        "context": (
            f"Hold % ranks sportsbooks by how much margin they keep. "
            f"Lowest hold: {hold_report[0]['sportsbook']} ({hold_report[0]['overall_hold_pct']}). "
            f"Highest hold: {hold_report[-1]['sportsbook']} ({hold_report[-1]['overall_hold_pct']}). "
            f"Market average: {round(avg_market_hold * 100, 2)}%."
        ) if hold_report else "No data",
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
    """Find +EV (positive expected value) bets with Kelly Criterion bet sizing.

    Uses Pinnacle's no-vig odds as the "true" probability reference (falls back
    to consensus when Pinnacle data is unavailable). A bet is +EV if the book's
    odds imply a lower probability than the fair probability.

    Each +EV bet includes Kelly Criterion sizing (full, half, and quarter Kelly)
    showing not just *what* to bet, but *how much* of your bankroll to wager.

    Args:
        filename: Data file to load. Optional.
        min_ev_pct: Minimum EV % to report (default 0). E.g., 2.0 = only bets with 2%+ edge.

    Returns all +EV bets sorted by expected value, with Kelly bet sizing.
    """
    cache_key = f"ev:{min_ev_pct}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    pinnacle_probs = _get_pinnacle_fair_probs(games)
    ev_bets = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds", "side_a_prob"), ("away", "away_odds", "side_b_prob")]
            else:
                sides = [("over", "over_odds", "side_a_prob"), ("under", "under_odds", "side_b_prob")]

            for side, odds_key, prob_key in sides:
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

                # Use Pinnacle fair prob if available, else consensus
                game_fair = pinnacle_probs.get(game_id, {}).get(market_type)
                if game_fair:
                    fair_prob = game_fair[prob_key]
                    prob_source = game_fair["source"]
                else:
                    probs = [implied_probability(o["odds"]) for o in all_odds]
                    fair_prob = sum(probs) / len(probs)
                    prob_source = "consensus"

                # Check each book for +EV (skip Pinnacle -- it is our reference)
                for entry in all_odds:
                    if entry["sportsbook"].lower() == SHARP_BOOK.lower():
                        continue

                    book_prob = implied_probability(entry["odds"])
                    ev_edge = fair_prob - book_prob

                    ev_pct = round(ev_edge * 100, 3)
                    if ev_pct > min_ev_pct:
                        # Kelly sizing at three fractions
                        kelly_full = kelly_criterion(entry["odds"], fair_prob, fraction=1.0)
                        kelly_half = kelly_criterion(entry["odds"], fair_prob, fraction=0.5)
                        kelly_quarter = kelly_criterion(entry["odds"], fair_prob, fraction=0.25)

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
                            "fair_prob": round(fair_prob, 6),
                            "fair_prob_source": prob_source,
                            "ev_edge_pct": ev_pct,
                            "kelly": {
                                "full_kelly": kelly_full["recommended_pct"],
                                "half_kelly": kelly_half["recommended_pct"],
                                "quarter_kelly": kelly_quarter["recommended_pct"],
                                "full_kelly_example": kelly_full["bankroll_example"],
                                "half_kelly_example": kelly_half["bankroll_example"],
                                "quarter_kelly_example": kelly_quarter["bankroll_example"],
                            },
                            "last_updated": entry.get("last_updated"),
                            "context": f"+EV: {side} {market_type} at {entry['sportsbook']} ({entry['odds']}). Fair ({prob_source}): {round(fair_prob*100,1)}%, book: {round(book_prob*100,1)}%. Edge: {ev_pct}%. Kelly (quarter): {kelly_quarter['recommended_pct']}",
                        })

    ev_bets.sort(key=lambda b: b["ev_edge_pct"], reverse=True)
    result = {
        "ev_bets": ev_bets,
        "count": len(ev_bets),
        "fair_prob_source": "Pinnacle no-vig (preferred) with consensus fallback",
        "kelly_note": "Quarter Kelly (25%) is recommended for most bettors to manage variance. Full Kelly maximizes long-term growth but has high volatility.",
        "context": f"Found {len(ev_bets)} +EV bets with Kelly sizing." + (f" Top: {ev_bets[0]['context']}" if ev_bets else ""),
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
def infer_odds_movement(filename: Optional[str] = None, stale_threshold_minutes: int = 30) -> str:
    """Infer where sharp money moved by comparing stale odds against fresh odds for the same game.

    When a sportsbook's line hasn't been updated but other books have moved,
    the *direction* of that movement reveals where sharp/professional bettors
    are placing money.  For example, if stale books still show -4.5 but fresh
    books have moved to -5.5, sharp money likely went on the favorite.

    This tool pairs each stale sportsbook entry with the freshest entry for
    the same game and reports line/odds deltas across spread, moneyline, and
    total markets.

    Args:
        filename: Data file to load. Optional.
        stale_threshold_minutes: How old a line must be (vs the freshest for
            that game) to be compared. Default 30 minutes.

    Returns inferred movement records sorted by staleness, with sharp-money
    direction labels.
    """
    cache_key = f"odds_movement:{stale_threshold_minutes}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    movements: list[dict] = []

    for game_id, records in games.items():
        # Parse timestamps for each record
        timed: list[dict] = []
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

        newest_ts = max(t["ts"] for t in valid)
        # The freshest record(s) serve as the "current market" reference
        fresh = [t for t in valid if (newest_ts - t["ts"]).total_seconds() / 60 < 5]
        stale = [t for t in valid if (newest_ts - t["ts"]).total_seconds() / 60 >= stale_threshold_minutes]

        if not fresh or not stale:
            continue

        # Build a composite "fresh consensus" from the freshest records
        fresh_records = [t["record"] for t in fresh]

        def _avg(vals: list) -> float | None:
            nums = [v for v in vals if v is not None]
            return round(sum(nums) / len(nums), 2) if nums else None

        fresh_markets: dict[str, dict] = {}
        for mtype in ("spread", "moneyline", "total"):
            market_entries = [r["markets"][mtype] for r in fresh_records if mtype in r.get("markets", {})]
            if not market_entries:
                continue
            if mtype == "spread":
                fresh_markets["spread"] = {
                    "home_line": _avg([m.get("home_line") for m in market_entries]),
                    "home_odds": _avg([m.get("home_odds") for m in market_entries]),
                    "away_odds": _avg([m.get("away_odds") for m in market_entries]),
                }
            elif mtype == "moneyline":
                fresh_markets["moneyline"] = {
                    "home_odds": _avg([m.get("home_odds") for m in market_entries]),
                    "away_odds": _avg([m.get("away_odds") for m in market_entries]),
                }
            elif mtype == "total":
                fresh_markets["total"] = {
                    "line": _avg([m.get("line") for m in market_entries]),
                    "over_odds": _avg([m.get("over_odds") for m in market_entries]),
                    "under_odds": _avg([m.get("under_odds") for m in market_entries]),
                }

        # Compare each stale record against the fresh consensus
        for t in stale:
            r = t["record"]
            age_minutes = (newest_ts - t["ts"]).total_seconds() / 60
            stale_markets = r.get("markets", {})
            market_deltas: dict[str, dict] = {}

            for mtype, fresh_m in fresh_markets.items():
                stale_m = stale_markets.get(mtype)
                if not stale_m:
                    continue

                delta: dict = {}
                if mtype == "spread":
                    stale_line = stale_m.get("home_line")
                    fresh_line = fresh_m.get("home_line")
                    if stale_line is not None and fresh_line is not None:
                        line_move = round(fresh_line - stale_line, 1)
                        delta["stale_home_line"] = stale_line
                        delta["fresh_home_line"] = fresh_line
                        delta["line_move"] = line_move
                        if line_move < 0:
                            delta["sharp_direction"] = "home_favorite_strengthened"
                            delta["interpretation"] = f"Line moved from {stale_line} to {fresh_line} — sharps backing the home team (favorite got more points)"
                        elif line_move > 0:
                            delta["sharp_direction"] = "away_team_getting_action"
                            delta["interpretation"] = f"Line moved from {stale_line} to {fresh_line} — sharps backing the away team"
                        else:
                            delta["sharp_direction"] = "no_movement"
                            delta["interpretation"] = "Spread unchanged"
                    # Odds deltas
                    for side in ("home_odds", "away_odds"):
                        s_val, f_val = stale_m.get(side), fresh_m.get(side)
                        if s_val is not None and f_val is not None:
                            delta[f"stale_{side}"] = s_val
                            delta[f"fresh_{side}"] = f_val
                            delta[f"{side}_move"] = round(f_val - s_val, 1)

                elif mtype == "moneyline":
                    for side in ("home_odds", "away_odds"):
                        s_val, f_val = stale_m.get(side), fresh_m.get(side)
                        if s_val is not None and f_val is not None:
                            delta[f"stale_{side}"] = s_val
                            delta[f"fresh_{side}"] = f_val
                            delta[f"{side}_move"] = round(f_val - s_val, 1)
                    # Infer direction from home moneyline shift
                    home_move = delta.get("home_odds_move")
                    if home_move is not None:
                        if home_move < -5:
                            delta["sharp_direction"] = "sharp_on_home"
                            delta["interpretation"] = f"Home ML shortened ({delta['stale_home_odds']}→{delta['fresh_home_odds']}) — sharps backing home"
                        elif home_move > 5:
                            delta["sharp_direction"] = "sharp_on_away"
                            delta["interpretation"] = f"Home ML lengthened ({delta['stale_home_odds']}→{delta['fresh_home_odds']}) — sharps backing away"
                        else:
                            delta["sharp_direction"] = "no_significant_movement"
                            delta["interpretation"] = "Moneyline stable"

                elif mtype == "total":
                    stale_line = stale_m.get("line")
                    fresh_line = fresh_m.get("line")
                    if stale_line is not None and fresh_line is not None:
                        line_move = round(fresh_line - stale_line, 1)
                        delta["stale_line"] = stale_line
                        delta["fresh_line"] = fresh_line
                        delta["line_move"] = line_move
                        if line_move > 0:
                            delta["sharp_direction"] = "sharp_on_over"
                            delta["interpretation"] = f"Total moved up from {stale_line} to {fresh_line} — sharps betting the over"
                        elif line_move < 0:
                            delta["sharp_direction"] = "sharp_on_under"
                            delta["interpretation"] = f"Total moved down from {stale_line} to {fresh_line} — sharps betting the under"
                        else:
                            delta["sharp_direction"] = "no_movement"
                            delta["interpretation"] = "Total unchanged"
                    for side in ("over_odds", "under_odds"):
                        s_val, f_val = stale_m.get(side), fresh_m.get(side)
                        if s_val is not None and f_val is not None:
                            delta[f"stale_{side}"] = s_val
                            delta[f"fresh_{side}"] = f_val
                            delta[f"{side}_move"] = round(f_val - s_val, 1)

                if delta:
                    market_deltas[mtype] = delta

            if market_deltas:
                # Summarize the dominant sharp direction across markets
                directions = [d.get("sharp_direction", "") for d in market_deltas.values()
                              if d.get("sharp_direction") and "no_" not in d.get("sharp_direction", "")]
                dominant = directions[0] if len(set(directions)) == 1 and directions else (
                    "mixed_signals" if directions else "no_significant_movement"
                )

                movements.append({
                    "game_id": game_id,
                    "sport": r.get("sport"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "stale_sportsbook": r["sportsbook"],
                    "stale_last_updated": r.get("last_updated"),
                    "fresh_books": [fr["sportsbook"] for fr in fresh_records],
                    "fresh_last_updated": newest_ts.isoformat(),
                    "staleness_minutes": round(age_minutes, 1),
                    "market_movements": market_deltas,
                    "dominant_sharp_direction": dominant,
                    "context": (
                        f"{r['sportsbook']} is {round(age_minutes)} min stale for "
                        f"{r.get('away_team', '?')} @ {r.get('home_team', '?')}. "
                        + "; ".join(
                            d.get("interpretation", "")
                            for d in market_deltas.values()
                            if d.get("interpretation") and "unchanged" not in d.get("interpretation", "").lower()
                                and "stable" not in d.get("interpretation", "").lower()
                        )
                    ),
                })

    movements.sort(key=lambda m: m["staleness_minutes"], reverse=True)

    # Summary stats
    sharp_counts: dict[str, int] = {}
    for m in movements:
        d = m["dominant_sharp_direction"]
        sharp_counts[d] = sharp_counts.get(d, 0) + 1

    result = {
        "odds_movements": movements,
        "count": len(movements),
        "sharp_direction_summary": sharp_counts,
        "threshold_minutes": stale_threshold_minutes,
        "methodology": (
            "Compares each stale sportsbook's lines/odds against the freshest "
            "entries for the same game. The direction of the delta (e.g., spread "
            "moved from -4.5 to -5.5) indicates where sharp money has gone. "
            "Fresh = updated within 5 min of the newest timestamp for that game."
        ),
        "context": (
            f"Found {len(movements)} stale-vs-fresh comparisons revealing line movement. "
            + (f"Dominant signal: {max(sharp_counts, key=sharp_counts.get)} ({sharp_counts[max(sharp_counts, key=sharp_counts.get)]} games)."
               if sharp_counts else "No significant movements detected.")
        ),
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

    # Also check for LINE outliers (spread line / total line deviations)
    for game_id, records in games.items():
        first = records[0]

        # Spread line outliers
        spread_entries = []
        for r in records:
            spread = r.get("markets", {}).get("spread", {})
            if "home_line" in spread:
                spread_entries.append({"sportsbook": r["sportsbook"], "home_line": spread["home_line"], "last_updated": r.get("last_updated")})
        if len(spread_entries) >= 3:
            lines = [e["home_line"] for e in spread_entries]
            avg_line = sum(lines) / len(lines)
            for e in spread_entries:
                line_dev = abs(e["home_line"] - avg_line)
                if line_dev >= 1.0:
                    outliers.append({
                        "game_id": game_id,
                        "sport": first.get("sport"),
                        "home_team": first.get("home_team"),
                        "away_team": first.get("away_team"),
                        "market_type": "spread",
                        "type": "line_outlier",
                        "sportsbook": e["sportsbook"],
                        "line": e["home_line"],
                        "consensus_line": round(avg_line, 1),
                        "deviation": round(line_dev, 1),
                        "last_updated": e.get("last_updated"),
                        "context": f"LINE OUTLIER: {e['sportsbook']} spread {e['home_line']} vs consensus {round(avg_line, 1)} (dev {round(line_dev, 1)} pts)",
                    })

        # Total line outliers
        total_entries = []
        for r in records:
            total = r.get("markets", {}).get("total", {})
            if "line" in total:
                total_entries.append({"sportsbook": r["sportsbook"], "line": total["line"], "last_updated": r.get("last_updated")})
        if len(total_entries) >= 3:
            lines = [e["line"] for e in total_entries]
            avg_line = sum(lines) / len(lines)
            for e in total_entries:
                line_dev = abs(e["line"] - avg_line)
                if line_dev >= 1.0:
                    outliers.append({
                        "game_id": game_id,
                        "sport": first.get("sport"),
                        "home_team": first.get("home_team"),
                        "away_team": first.get("away_team"),
                        "market_type": "total",
                        "type": "line_outlier",
                        "sportsbook": e["sportsbook"],
                        "line": e["line"],
                        "consensus_line": round(avg_line, 1),
                        "deviation": round(line_dev, 1),
                        "last_updated": e.get("last_updated"),
                        "context": f"LINE OUTLIER: {e['sportsbook']} total {e['line']} vs consensus {round(avg_line, 1)} (dev {round(line_dev, 1)} pts)",
                    })

    outliers.sort(key=lambda o: o["deviation"], reverse=True)
    result = {
        "outliers": outliers,
        "count": len(outliers),
        "threshold": threshold_odds,
        "odds_outliers": len([o for o in outliers if o.get("type") != "line_outlier"]),
        "line_outliers": len([o for o in outliers if o.get("type") == "line_outlier"]),
        "context": f"Found {len(outliers)} outlier lines (>{threshold_odds} pts from consensus for odds, >=1 pt for lines)." + (f" Biggest: {outliers[0]['context']}" if outliers else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Aggregation / Summary
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_fair_odds(game_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Get the consensus no-vig "fair" odds for each game and market.

    Removes the vig from each sportsbook's odds, then averages across all
    books to produce a consensus true probability baseline (crowd wisdom).
    Also compares against Pinnacle's no-vig odds (sharp wisdom) to highlight
    where recreational books may be mispricing.

    Args:
        game_id: Filter to a specific game. Optional (shows all games).
        filename: Data file to load. Optional.

    Returns consensus fair probabilities, fair American odds, and sharp vs crowd
    divergence data per game per market.
    """
    cache_key = f"fair_odds:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    # Load sharp vs crowd data for all games
    sharp_vs_crowd = _cache.load_sharp_vs_crowd(filename)

    fair_odds_list = []
    for gid, records in games.items():
        first = records[0]
        game_entry = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "markets": {},
        }

        game_svc = sharp_vs_crowd.get(gid, {})

        for market_type in ("spread", "moneyline", "total"):
            market_records = [r["markets"][market_type] for r in records if market_type in r.get("markets", {})]
            if not market_records:
                continue

            if market_type in ("spread", "moneyline"):
                # Collect fair probs from each book's enriched data
                home_fair_probs = [m.get("home_implied_prob", 0) for m in market_records]
                away_fair_probs = [m.get("away_implied_prob", 0) for m in market_records]
                home_vigs = [m.get("vig", 0) for m in market_records]

                # Average raw implied probs, then normalize to remove vig
                avg_home = sum(home_fair_probs) / len(home_fair_probs)
                avg_away = sum(away_fair_probs) / len(away_fair_probs)
                total_prob = avg_home + avg_away
                fair_home = avg_home / total_prob if total_prob else 0.5
                fair_away = avg_away / total_prob if total_prob else 0.5

                market_entry = {
                    "home_fair_prob": round(fair_home, 6),
                    "away_fair_prob": round(fair_away, 6),
                    "home_fair_prob_pct": f"{round(fair_home * 100, 2)}%",
                    "away_fair_prob_pct": f"{round(fair_away * 100, 2)}%",
                    "home_fair_odds": fair_odds_to_american(fair_home),
                    "away_fair_odds": fair_odds_to_american(fair_away),
                    "avg_vig": round(sum(home_vigs) / len(home_vigs), 6) if home_vigs else 0,
                    "books_sampled": len(market_records),
                }
                if market_type == "spread":
                    lines = [m.get("home_line", 0) for m in market_records]
                    market_entry["consensus_line"] = round(sum(lines) / len(lines), 1)

                # Attach sharp vs crowd divergence data
                if market_type in game_svc:
                    market_entry["sharp_vs_crowd"] = game_svc[market_type]

                game_entry["markets"][market_type] = market_entry
            else:
                over_probs = [m.get("over_implied_prob", 0) for m in market_records]
                under_probs = [m.get("under_implied_prob", 0) for m in market_records]
                total_vigs = [m.get("vig", 0) for m in market_records]

                avg_over = sum(over_probs) / len(over_probs)
                avg_under = sum(under_probs) / len(under_probs)
                total_prob = avg_over + avg_under
                fair_over = avg_over / total_prob if total_prob else 0.5
                fair_under = avg_under / total_prob if total_prob else 0.5

                lines = [m.get("line", 0) for m in market_records]
                total_entry = {
                    "consensus_line": round(sum(lines) / len(lines), 1),
                    "over_fair_prob": round(fair_over, 6),
                    "under_fair_prob": round(fair_under, 6),
                    "over_fair_prob_pct": f"{round(fair_over * 100, 2)}%",
                    "under_fair_prob_pct": f"{round(fair_under * 100, 2)}%",
                    "over_fair_odds": fair_odds_to_american(fair_over),
                    "under_fair_odds": fair_odds_to_american(fair_under),
                    "avg_vig": round(sum(total_vigs) / len(total_vigs), 6) if total_vigs else 0,
                    "books_sampled": len(market_records),
                }

                # Attach sharp vs crowd divergence data
                if "total" in game_svc:
                    total_entry["sharp_vs_crowd"] = game_svc["total"]

                game_entry["markets"]["total"] = total_entry

        fair_odds_list.append(game_entry)

    # Summarize mispricing alerts across all games
    mispricing_alerts = []
    for game in fair_odds_list:
        for mkt_name, mkt_data in game.get("markets", {}).items():
            svc = mkt_data.get("sharp_vs_crowd", {})
            if svc.get("mispriced_side"):
                mispricing_alerts.append({
                    "game_id": game["game_id"],
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "market": mkt_name,
                    "mispriced_side": svc["mispriced_side"],
                    "direction": svc["mispriced_direction"],
                    "divergence_pct": svc["max_divergence_pct"],
                })
    mispricing_alerts.sort(key=lambda x: x["divergence_pct"], reverse=True)

    result = {
        "games": fair_odds_list,
        "count": len(fair_odds_list),
        "mispricing_alerts": mispricing_alerts,
        "mispricing_count": len(mispricing_alerts),
        "context": (
            f"Consensus fair (no-vig) odds for {len(fair_odds_list)} games. "
            f"Includes sharp (Pinnacle) vs crowd (all 8 books) divergence. "
            f"{len(mispricing_alerts)} market(s) show significant divergence (>=1pp) — "
            f"these highlight where recreational books may be mispricing vs the sharp line."
        ),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_market_summary(filename: Optional[str] = None) -> str:
    """Get a comprehensive market summary — a structured digest of the entire dataset.

    This is the best "start here" tool. It gives Claude a high-level overview:
    - Event count and sports covered
    - Sportsbook rankings by vig
    - Top +EV bets
    - Arbitrage opportunities
    - Stale lines & sharp money movement inferred from staleness
    - Biggest outliers
    - Power rankings (market-implied team strength)

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

    middles_data = json.loads(find_middle_opportunities(filename))
    fair_odds_data = json.loads(get_fair_odds(filename))
    hold_data = json.loads(get_hold_percentage(filename))
    power_data = json.loads(get_power_rankings(filename))
    movement_data = json.loads(infer_odds_movement(filename))
    sharpness_data = json.loads(get_sharpness_scores(filename))
    correlation_data = json.loads(get_market_correlations(filename))
    synthetic_data = json.loads(get_synthetic_hold_free_market(filename=filename))
    entropy_data = json.loads(get_market_entropy(filename=filename))
    cross_mkt_data = json.loads(find_cross_market_arbitrage(filename))

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
        "middle_opportunities": {
            "count": middles_data.get("count", 0),
            "top_3": middles_data.get("middle_opportunities", [])[:3],
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
            "odds_outliers": outlier_data.get("odds_outliers", 0),
            "line_outliers": outlier_data.get("line_outliers", 0),
            "top_3": outlier_data.get("outliers", [])[:3],
        },
        "sharp_money_movement": {
            "count": movement_data.get("count", 0),
            "direction_summary": movement_data.get("sharp_direction_summary", {}),
            "top_3": movement_data.get("odds_movements", [])[:3],
        },
        "fair_odds_consensus": fair_odds_data.get("games", [])[:3],
        "hold_percentage": {
            "market_average": hold_data.get("market_average_hold_pct", "N/A"),
            "lowest_hold": hold_data.get("lowest_hold"),
            "highest_hold": hold_data.get("highest_hold"),
            "by_sportsbook": hold_data.get("hold_by_sportsbook", []),
        },
        "power_rankings": {
            "total_teams": power_data.get("total_teams", 0),
            "top_5": [
                {"rank": r["rank"], "team": r["team"], "strength": r["strength_pct"], "tier": r["tier"]}
                for r in power_data.get("power_rankings", [])[:5]
            ],
            "bottom_5": [
                {"rank": r["rank"], "team": r["team"], "strength": r["strength_pct"], "tier": r["tier"]}
                for r in power_data.get("power_rankings", [])[-5:]
            ],
        },
        "sharpness_scores": {
            "benchmark": sharpness_data.get("benchmark_book", "Pinnacle"),
            "sharpest": sharpness_data.get("sharpest_book"),
            "softest": sharpness_data.get("softest_book"),
            "rankings": [
                {"sportsbook": r["sportsbook"], "score": r["sharpness_score"], "classification": r["classification"]}
                for r in sharpness_data.get("rankings", [])
            ],
        },
        "market_correlations": {
            "spread_vs_ml": correlation_data.get("correlations", {}).get("spread_vs_moneyline", {}),
            "spread_vs_total": correlation_data.get("correlations", {}).get("spread_size_vs_total", {}),
            "inconsistency_count": correlation_data.get("inconsistency_count", 0),
            "top_inconsistencies": correlation_data.get("inconsistencies", [])[:3],
            "most_consistent_book": correlation_data.get("most_consistent_book"),
            "least_consistent_book": correlation_data.get("least_consistent_book"),
        },
        "market_entropy": {
            "overall_avg_entropy": entropy_data.get("overall_avg_entropy", 0),
            "sport_breakdown": entropy_data.get("sport_breakdown", []),
            "most_exploitable_games": entropy_data.get("most_exploitable_games", [])[:3],
            "total_games_analyzed": entropy_data.get("total_games_analyzed", 0),
        },
        "synthetic_hold_free_market": {
            "avg_synthetic_hold": synthetic_data.get("aggregate", {}).get("avg_synthetic_hold_pct", "N/A"),
            "avg_sharp_edge": synthetic_data.get("aggregate", {}).get("avg_sharp_edge_pct", "N/A"),
            "arb_markets": synthetic_data.get("aggregate", {}).get("arb_market_count", 0),
            "best_edge": synthetic_data.get("aggregate", {}).get("best_edge_pct", "N/A"),
            "by_market_type": synthetic_data.get("aggregate", {}).get("by_market_type", {}),
            "top_3_games": synthetic_data.get("games", [])[:3],
        },
        "cross_market_arbitrage": {
            "count": cross_mkt_data.get("count", 0),
            "breakdown": cross_mkt_data.get("breakdown", {}),
            "top_3": cross_mkt_data.get("cross_market_opportunities", [])[:3],
        },
        "context": "Full market summary with sportsbook rankings, hold percentage, arb opportunities (same-market + cross-market), middles, +EV bets, stale lines, sharp money movement, outliers (odds + line), fair odds, power rankings, sharpness scores, market correlations, entropy-based market efficiency, and synthetic hold-free market. Drill into specific tools for more detail.",
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


# 

@mcp.tool()
def get_kelly_sizing(game_id: Optional[str] = None, filename: Optional[str] = None, kelly_fraction: float = 0.25, bankroll: float = 1000.0) -> str:
    """Calculate Kelly Criterion bet sizing for every +EV opportunity.

    Uses Pinnacle's no-vig odds as the "true" probability baseline (falls back
    to consensus when Pinnacle data is unavailable).  For each +EV bet at other
    books, computes the optimal Kelly fraction of bankroll to wager.

    This answers "how much should I bet?" -- not just "what should I bet?"

    Args:
        game_id: Filter to a specific game. Optional (shows all games).
        filename: Data file to load. Optional.
        kelly_fraction: Kelly fraction to apply (default 0.25 = quarter Kelly, safest).
                        Common values: 1.0 (full), 0.5 (half), 0.25 (quarter).
        bankroll: Your bankroll in dollars for sizing examples. Default $1,000.

    Returns all +EV opportunities with Kelly bet sizes, sorted by recommended wager.
    """
    cache_key = f"kelly:{game_id or 'all'}:{kelly_fraction}:{bankroll}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    if game_id:
        games = {k: v for k, v in games.items() if k == game_id}

    pinnacle_probs = _get_pinnacle_fair_probs(games)
    kelly_bets = []

    for gid, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds", "side_a_prob"), ("away", "away_odds", "side_b_prob")]
            else:
                sides = [("over", "over_odds", "side_a_prob"), ("under", "under_odds", "side_b_prob")]

            for side, odds_key, prob_key in sides:
                game_fair = pinnacle_probs.get(gid, {}).get(market_type)
                if not game_fair:
                    continue
                fair_prob = game_fair[prob_key]
                prob_source = game_fair["source"]

                for r in records:
                    # Skip the sharp book itself
                    if r.get("sportsbook", "").lower() == SHARP_BOOK.lower():
                        continue

                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key not in market:
                        continue

                    book_odds = market[odds_key]
                    book_prob = implied_probability(book_odds)
                    ev_edge = fair_prob - book_prob

                    if ev_edge <= 0:
                        continue

                    # Kelly sizing
                    kelly = kelly_criterion(book_odds, fair_prob, fraction=kelly_fraction)
                    if not kelly["is_positive"]:
                        continue

                    wager = round(kelly["recommended_fraction"] * bankroll, 2)

                    kelly_bets.append({
                        "game_id": gid,
                        "sport": first.get("sport"),
                        "home_team": first.get("home_team"),
                        "away_team": first.get("away_team"),
                        "market_type": market_type,
                        "side": side,
                        "sportsbook": r["sportsbook"],
                        "odds": book_odds,
                        "book_implied_prob": round(book_prob, 6),
                        "fair_prob": round(fair_prob, 6),
                        "fair_prob_source": prob_source,
                        "ev_edge_pct": round(ev_edge * 100, 3),
                        "kelly_pct": kelly["recommended_pct"],
                        "wager": f"${wager}",
                        "wager_raw": wager,
                        "full_kelly_pct": kelly["full_kelly_pct"],
                        "last_updated": r.get("last_updated"),
                        "context": f"Bet ${wager} ({kelly['recommended_pct']} of bankroll) on {side} {market_type} at {r['sportsbook']} ({book_odds}). Edge: {round(ev_edge*100,2)}% vs {prob_source}.",
                    })

    kelly_bets.sort(key=lambda b: b["wager_raw"], reverse=True)
    total_wagered = round(sum(b["wager_raw"] for b in kelly_bets), 2)

    result = {
        "kelly_bets": kelly_bets,
        "count": len(kelly_bets),
        "settings": {
            "kelly_fraction": kelly_fraction,
            "bankroll": f"${bankroll}",
            "fair_prob_source": "Pinnacle no-vig (preferred) with consensus fallback",
        },
        "total_suggested_wager": f"${total_wagered}",
        "bankroll_pct_deployed": f"{round(total_wagered / bankroll * 100, 1)}%" if bankroll > 0 else "N/A",
        "kelly_note": f"Using {kelly_fraction}x Kelly ({round(kelly_fraction*100)}% of full Kelly). Quarter Kelly (0.25) is recommended for most bettors to manage variance.",
        "context": f"Found {len(kelly_bets)} Kelly-sized bets totaling ${total_wagered} ({round(total_wagered/bankroll*100,1)}% of ${bankroll} bankroll)." + (f" Top: {kelly_bets[0]['context']}" if kelly_bets else ""),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_market_entropy(game_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Measure market efficiency via Shannon entropy of implied probabilities across books.

    For each game and market, collects the implied probability each sportsbook
    assigns, normalises them into a distribution, and computes Shannon entropy.

    Interpretation:
    - **Higher entropy** → more disagreement between books → potentially exploitable.
    - **Lower entropy**  → strong consensus → efficient, hard-to-beat market.
    - Maximum possible entropy = log2(N) where N = number of books.

    The tool also returns an **efficiency ratio** (actual / max entropy, 0–1) so
    values are comparable across games with different book counts, and a league /
    overall summary so you can quickly spot the most inefficient games.

    Args:
        game_id: Filter to a specific game. Optional (analyzes all games).
        filename: Data file to load. Optional.

    Returns per-game, per-market entropy metrics plus a ranked list of the
    most inefficient (highest-entropy) games.
    """
    odds = _load_odds(filename)
    if game_id:
        odds = [r for r in odds if r.get("game_id") == game_id]

    cache_key = f"entropy:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    def _shannon_entropy(probs: list[float]) -> float:
        """Compute Shannon entropy (base-2) of a discrete probability distribution."""
        total = sum(probs)
        if total <= 0:
            return 0.0
        normed = [p / total for p in probs if p > 0]
        return -sum(p * log(p, 2) for p in normed)

    entropy_by_game: list[dict] = []

    for gid, records in games.items():
        first = records[0]
        game_entry: dict = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "markets": {},
            "avg_entropy": 0.0,
            "avg_efficiency_ratio": 0.0,
        }

        market_entropies: list[float] = []
        market_ratios: list[float] = []

        for market_type in ("spread", "moneyline", "total"):
            # Determine which probability keys to collect per side
            if market_type in ("spread", "moneyline"):
                side_keys = {
                    "home": "home_implied_prob",
                    "away": "away_implied_prob",
                }
            else:
                side_keys = {
                    "over": "over_implied_prob",
                    "under": "under_implied_prob",
                }

            # Collect per-book implied probs for each side
            side_probs: dict[str, list[dict]] = {}
            for r in records:
                mkt = r.get("markets", {}).get(market_type, {})
                if not mkt:
                    continue
                for side_label, prob_key in side_keys.items():
                    prob = mkt.get(prob_key)
                    if prob is not None and prob > 0:
                        side_probs.setdefault(side_label, []).append({
                            "sportsbook": r["sportsbook"],
                            "implied_prob": prob,
                        })

            if not side_probs:
                continue

            # Compute entropy per side, then average.
            # Higher entropy = more book disagreement on this side's probability.
            side_results: dict[str, dict] = {}
            for side_label, entries in side_probs.items():
                if len(entries) < 2:
                    continue
                probs = [e["implied_prob"] for e in entries]
                n = len(probs)
                h = _shannon_entropy(probs)
                max_h = log(n, 2) if n > 1 else 1.0
                ratio = h / max_h if max_h > 0 else 0.0

                side_results[side_label] = {
                    "entropy_bits": round(h, 6),
                    "max_entropy_bits": round(max_h, 6),
                    "efficiency_ratio": round(ratio, 6),
                    "book_count": n,
                    "prob_range": [round(min(probs), 6), round(max(probs), 6)],
                    "prob_spread": round(max(probs) - min(probs), 6),
                    "books": [
                        {"sportsbook": e["sportsbook"], "implied_prob": round(e["implied_prob"], 6)}
                        for e in sorted(entries, key=lambda x: x["implied_prob"], reverse=True)
                    ],
                }

            if side_results:
                avg_entropy = sum(s["entropy_bits"] for s in side_results.values()) / len(side_results)
                avg_ratio = sum(s["efficiency_ratio"] for s in side_results.values()) / len(side_results)
                game_entry["markets"][market_type] = {
                    "sides": side_results,
                    "avg_entropy_bits": round(avg_entropy, 6),
                    "avg_efficiency_ratio": round(avg_ratio, 6),
                }
                market_entropies.append(avg_entropy)
                market_ratios.append(avg_ratio)

        if market_entropies:
            game_entry["avg_entropy"] = round(sum(market_entropies) / len(market_entropies), 6)
            game_entry["avg_efficiency_ratio"] = round(sum(market_ratios) / len(market_ratios), 6)

        entropy_by_game.append(game_entry)

    # Sort by entropy descending — most inefficient (exploitable) games first
    entropy_by_game.sort(key=lambda g: g["avg_entropy"], reverse=True)

    # Overall stats
    all_entropies = [g["avg_entropy"] for g in entropy_by_game if g["avg_entropy"] > 0]
    overall_avg = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0

    # Per-sport breakdown
    sport_buckets: dict[str, list[float]] = {}
    for g in entropy_by_game:
        sport = g.get("sport", "Unknown")
        if g["avg_entropy"] > 0:
            sport_buckets.setdefault(sport, []).append(g["avg_entropy"])
    sport_summary = []
    for sport, vals in sport_buckets.items():
        sport_summary.append({
            "sport": sport,
            "avg_entropy": round(sum(vals) / len(vals), 6),
            "games": len(vals),
            "most_inefficient": max(vals),
        })
    sport_summary.sort(key=lambda s: s["avg_entropy"], reverse=True)

    # Top exploitable games (top 5)
    top_exploitable = [
        {
            "game_id": g["game_id"],
            "matchup": f"{g.get('away_team', '?')} @ {g.get('home_team', '?')}",
            "sport": g.get("sport"),
            "avg_entropy": g["avg_entropy"],
            "avg_efficiency_ratio": g["avg_efficiency_ratio"],
        }
        for g in entropy_by_game[:5]
    ]

    result = {
        "entropy_by_game": entropy_by_game,
        "overall_avg_entropy": round(overall_avg, 6),
        "sport_breakdown": sport_summary,
        "most_exploitable_games": top_exploitable,
        "total_games_analyzed": len(entropy_by_game),
        "context": (
            f"Shannon entropy across {len(entropy_by_game)} games (avg {round(overall_avg, 4)} bits). "
            f"Higher entropy = more book disagreement = potentially exploitable. "
            f"Most inefficient: {top_exploitable[0]['matchup']} ({top_exploitable[0]['avg_entropy']:.4f} bits)."
            if top_exploitable else "No games with sufficient data for entropy analysis."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)



# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Phase 4: New Recommended Tools
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_best_bets_today(filename: Optional[str] = None, count: int = 10) -> str:
    """Get the top-N best bets right now, ranked by a composite value score, with Kelly bet sizing.

    Combines +EV edge (using Pinnacle no-vig as true probability), low vig,
    outlier value, and line freshness into a single ranked list of actionable
    recommendations. Each bet includes Kelly Criterion sizing showing how much
    of your bankroll to wager. This is the best tool for answering "what should
    I bet on and how much?"

    Args:
        filename: Data file to load. Optional.
        count: Number of top bets to return. Default 10.

    Returns a ranked list of the best bets with reasoning and Kelly sizing for each.
    """
    cache_key = f"best_bets:{count}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    pinnacle_probs = _get_pinnacle_fair_probs(games)

    scored_bets = []

    for game_id, records in games.items():
        first = records[0]

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds", "side_a_prob"), ("away", "away_odds", "side_b_prob")]
            else:
                sides = [("over", "over_odds", "side_a_prob"), ("under", "under_odds", "side_b_prob")]

            for side, odds_key, prob_key in sides:
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

                # Use Pinnacle fair prob if available, else consensus
                game_fair = pinnacle_probs.get(game_id, {}).get(market_type)
                if game_fair:
                    fair_prob = game_fair[prob_key]
                    prob_source = game_fair["source"]
                else:
                    probs = [implied_probability(e["odds"]) for e in all_entries]
                    fair_prob = sum(probs) / len(probs)
                    prob_source = "consensus"

                # Find the best odds entry (highest American odds), skip Pinnacle
                non_pinnacle = [e for e in all_entries if e["sportsbook"].lower() != SHARP_BOOK.lower()]
                if not non_pinnacle:
                    continue
                best_entry = max(non_pinnacle, key=lambda e: e["odds"])
                best_prob = implied_probability(best_entry["odds"])

                # EV edge
                ev_edge = (fair_prob - best_prob) * 100
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

                # Kelly sizing
                kelly_full = kelly_criterion(best_entry["odds"], fair_prob, fraction=1.0)
                kelly_half = kelly_criterion(best_entry["odds"], fair_prob, fraction=0.5)
                kelly_quarter = kelly_criterion(best_entry["odds"], fair_prob, fraction=0.25)

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
                    "book_implied_prob": round(best_prob, 6),
                    "fair_prob": round(fair_prob, 6),
                    "fair_prob_source": prob_source,
                    "vig_at_book": round(vig, 6),
                    "composite_score": round(composite_score, 3),
                    "kelly": {
                        "full_kelly": kelly_full["recommended_pct"],
                        "half_kelly": kelly_half["recommended_pct"],
                        "quarter_kelly": kelly_quarter["recommended_pct"],
                        "quarter_kelly_example": kelly_quarter["bankroll_example"],
                    },
                    "last_updated": best_entry.get("last_updated", ""),
                    "freshness_penalty": round(freshness_penalty, 3),
                    "reasons": [],
                })

                # Build reasoning
                bet = scored_bets[-1]
                bet["reasons"].append(f"+EV edge: {round(ev_edge, 2)}% vs {prob_source} fair prob")
                bet["reasons"].append(f"Kelly sizing (quarter): {kelly_quarter['recommended_pct']} — {kelly_quarter['bankroll_example']}")
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
        "fair_prob_source": "Pinnacle no-vig (preferred) with consensus fallback",
        "kelly_note": "Quarter Kelly (25%) is recommended. Kelly sizing uses Pinnacle no-vig probabilities as truth.",
        "context": f"Top {len(top)} bets by composite score (EV + outlier value - vig - staleness) with Kelly sizing." + (f" #1: {top[0]['side']} {top[0]['market_type']} at {top[0]['sportsbook']} ({top[0]['odds']}) — {', '.join(top[0]['reasons'])}" if top else ""),
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
            "hold_pct": f"{round(avg_vig * 100, 2)}%",
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
def get_implied_scores(
    game_id: Optional[str] = None, filename: Optional[str] = None
) -> str:
    """Estimate implied final scores for each team by combining the consensus spread and total.

    Formula:
        Home implied score = (Total + HomeSpread) / 2
        Away implied score = (Total - HomeSpread) / 2

    e.g., spread -5.5 + total 220 → Home ~107.25, Away ~112.75
    (negative home spread means home is favored)

    Args:
        game_id: Filter to a single game. Optional — returns all games if omitted.
        filename: Data file to load. Optional.

    Returns implied scores per game sorted by largest margin of victory.
    """
    cache_key = f"implied_scores:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    consensus = _cache.load_consensus(filename)
    results = []

    for gid, records in games.items():
        if game_id and gid != game_id:
            continue

        game_consensus = consensus.get(gid, {})
        spread_c = game_consensus.get("spread", {})
        total_c = game_consensus.get("total", {})

        avg_home_line = spread_c.get("avg_home_line")
        avg_total = total_c.get("avg_line")

        if avg_home_line is None or avg_total is None:
            continue

        first = records[0]
        home_team = first.get("home_team", "Home")
        away_team = first.get("away_team", "Away")

        # Home spread is negative when home is favored, so:
        # home_implied = (total + home_line) / 2
        # away_implied = (total - home_line) / 2
        home_implied = round((avg_total + avg_home_line) / 2, 2)
        away_implied = round((avg_total - avg_home_line) / 2, 2)
        margin = round(abs(home_implied - away_implied), 2)

        # Determine the favorite
        if home_implied > away_implied:
            favorite = home_team
        elif away_implied > home_implied:
            favorite = away_team
        else:
            favorite = "Pick'em"

        results.append(
            {
                "game_id": gid,
                "sport": first.get("sport"),
                "home_team": home_team,
                "away_team": away_team,
                "consensus_spread": avg_home_line,
                "consensus_total": avg_total,
                "home_implied_score": home_implied,
                "away_implied_score": away_implied,
                "margin_of_victory": margin,
                "favorite": favorite,
                "spread_books": spread_c.get("book_count", 0),
                "total_books": total_c.get("book_count", 0),
                "context": (
                    f"{home_team} {home_implied} - {away_team} {away_implied} "
                    f"(spread {avg_home_line}, total {avg_total}). "
                    f"{favorite} by {margin}."
                ),
            }
        )

    results.sort(key=lambda r: r["margin_of_victory"], reverse=True)

    result = {
        "implied_scores": results,
        "count": len(results),
        "closest_game": results[-1] if results else None,
        "biggest_blowout": results[0] if results else None,
        "context": (
            f"Implied final scores for {len(results)} games based on consensus "
            f"spread + total. "
            + (
                f"Closest: {results[-1]['home_team']} vs "
                f"{results[-1]['away_team']} "
                f"(margin {results[-1]['margin_of_victory']}). "
                f"Biggest gap: {results[0]['home_team']} vs "
                f"{results[0]['away_team']} "
                f"(margin {results[0]['margin_of_victory']})."
                if len(results) >= 2
                else ""
            )
        ),
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
    power_data = json.loads(get_power_rankings(filename))
    movement_data = json.loads(infer_odds_movement(filename))
    cross_mkt_data = json.loads(find_cross_market_arbitrage(filename))

    # Build lookup sets for cross-referencing stale lines, outliers, and sharp movement
    stale_keys = set()
    for stale in summary_data.get("stale_lines", {}).get("top_3", []):
        stale_keys.add((stale.get("game_id"), stale.get("sportsbook")))

    outlier_lookup = {}
    for outlier in summary_data.get("outliers", {}).get("top_3", []):
        key = (outlier.get("game_id"), outlier.get("sportsbook"), outlier.get("market_type"), outlier.get("side"))
        outlier_lookup[key] = outlier

    # Build sharp movement lookup by game_id
    movement_lookup: dict[str, list[dict]] = {}
    for mv in movement_data.get("odds_movements", []):
        gid = mv.get("game_id")
        if gid:
            movement_lookup.setdefault(gid, []).append(mv)

    # Categorize
    must_bet = []
    for bet in best_bets_data.get("best_bets", []):
        if bet.get("composite_score", 0) > 2.0:
            # Cross-reference: check if this bet's book+game is flagged as stale
            warnings = []
            is_stale = (bet.get("game_id"), bet.get("sportsbook")) in stale_keys
            if is_stale:
                warnings.append("⚠️ This book's line for this game is flagged as stale — verify before betting")

            # Cross-reference: check if this bet is an outlier
            outlier_key = (bet.get("game_id"), bet.get("sportsbook"), bet.get("market_type"), bet.get("side"))
            outlier_match = outlier_lookup.get(outlier_key)
            if outlier_match:
                warnings.append(f"⚠️ Outlier: odds {bet['odds']} vs consensus avg {outlier_match.get('consensus_avg')} — may be mispriced/stale")

            # Cross-reference: check for sharp money movement on this game
            game_movements = movement_lookup.get(bet.get("game_id"), [])
            sharp_note = None
            for mv in game_movements:
                for mtype, delta in mv.get("market_movements", {}).items():
                    interp = delta.get("interpretation", "")
                    if interp and "unchanged" not in interp.lower() and "stable" not in interp.lower():
                        sharp_note = f"📊 Sharp movement: {interp}"
                        break
                if sharp_note:
                    break
            if sharp_note:
                warnings.append(sharp_note)

            # Confidence tier
            has_freshness_issue = bet.get("freshness_penalty", 0) > 0 or is_stale
            has_outlier_flag = outlier_match is not None
            if not has_freshness_issue and not has_outlier_flag and bet.get("ev_edge_pct", 0) >= 2.0:
                confidence = "high"
            elif has_freshness_issue or has_outlier_flag:
                confidence = "speculative"
            else:
                confidence = "moderate"

            # Include Kelly sizing if available
            kelly_info = bet.get("kelly", {})

            must_bet.append({
                "action": f"Bet {bet['side']} {bet['market_type']} at {bet['sportsbook']}",
                "game": f"{bet.get('away_team', '?')} @ {bet.get('home_team', '?')}",
                "odds": bet["odds"],
                "ev_edge": f"{bet['ev_edge_pct']}%",
                "book_implied_prob": f"{round(bet.get('book_implied_prob', 0) * 100, 1)}%",
                "fair_prob": f"{round(bet.get('fair_prob', bet.get('consensus_fair_prob', 0)) * 100, 1)}%",
                "fair_prob_source": bet.get("fair_prob_source", "consensus"),
                "kelly_quarter": kelly_info.get("quarter_kelly", "N/A"),
                "kelly_example": kelly_info.get("quarter_kelly_example", "N/A"),
                "last_updated": bet.get("last_updated", "unknown"),
                "confidence": confidence,
                "warnings": warnings,
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
    for mv in movement_data.get("odds_movements", [])[:3]:
        if mv.get("dominant_sharp_direction") not in ("no_significant_movement", "no_movement"):
            interesting.append({
                "type": "sharp_movement",
                "game": f"{mv.get('away_team', '?')} @ {mv.get('home_team', '?')}",
                "stale_book": mv.get("stale_sportsbook"),
                "staleness_minutes": mv.get("staleness_minutes"),
                "direction": mv.get("dominant_sharp_direction"),
                "description": mv.get("context", ""),
            })
    for xm in cross_mkt_data.get("cross_market_opportunities", [])[:3]:
        interesting.append({
            "type": "cross_market_arbitrage",
            "arb_type": xm.get("type", ""),
            "game": f"{xm.get('away_team', '?')} @ {xm.get('home_team', '?')}",
            "edge_pct": xm.get("edge_pct") or xm.get("prob_discrepancy_pct") or xm.get("gap_pct"),
            "description": xm.get("context", ""),
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
        "sharp_money_movement": {
            "count": movement_data.get("count", 0),
            "direction_summary": movement_data.get("sharp_direction_summary", {}),
            "top_movers": [
                {
                    "game": f"{mv.get('away_team', '?')} @ {mv.get('home_team', '?')}",
                    "stale_book": mv.get("stale_sportsbook"),
                    "staleness_minutes": mv.get("staleness_minutes"),
                    "direction": mv.get("dominant_sharp_direction"),
                    "details": mv.get("context", ""),
                }
                for mv in movement_data.get("odds_movements", [])[:5]
                if mv.get("dominant_sharp_direction") not in ("no_significant_movement", "no_movement")
            ],
        },
        "book_grades": book_grades,
        "power_rankings": [
            {"rank": r["rank"], "team": r["team"], "strength": r["strength_pct"], "tier": r["tier"]}
            for r in power_data.get("power_rankings", [])
        ],
        "context": f"Daily digest: {len(must_bet)} must-bet opps, {len(avoid)} lines to avoid, {len(interesting)} interesting situations, sharp movement on {movement_data.get('count', 0)} lines across {summary_data.get('summary', {}).get('unique_games', 0)} games.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_power_rankings(filename: Optional[str] = None) -> str:
    """Derive implied team strength ratings from moneyline odds and rank all teams.

    Aggregates each team's no-vig (fair) moneyline win probability across every
    game they appear in, then averages those probabilities to produce a single
    strength rating per team.  Teams are ranked from strongest to weakest.

    The rating reflects how the *market* prices each team — a higher rating means
    sportsbooks collectively view that team as more likely to win its games.

    Args:
        filename: Data file to load. Optional.

    Returns a ranked list of teams with strength ratings, game-level detail,
    and contextual notes.
    """
    cache_key = "power_rankings"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data available"})

    # Collect per-team fair win probabilities from each game's moneyline.
    # We use fair (no-vig) probabilities so the bookmaker margin doesn't inflate
    # one side.  Each game contributes two data points — one for each team.
    team_games: dict[str, list[dict]] = {}  # team_name -> [{game detail}, ...]

    by_game = _cache.load_by_game(filename)
    consensus = _cache.load_consensus(filename)

    for game_id, records in by_game.items():
        # Identify home/away team names from the first record of this game
        sample = records[0]
        home_team = sample.get("home_team", "Unknown")
        away_team = sample.get("away_team", "Unknown")

        # Use consensus moneyline odds to derive fair probabilities for this game
        game_consensus = consensus.get(game_id, {})
        ml_consensus = game_consensus.get("moneyline")
        if not ml_consensus:
            continue

        avg_home_odds = ml_consensus.get("avg_home_odds")
        avg_away_odds = ml_consensus.get("avg_away_odds")
        if avg_home_odds is None or avg_away_odds is None:
            continue

        fair = no_vig_probabilities(avg_home_odds, avg_away_odds)
        home_fair_prob = fair["fair_a"]
        away_fair_prob = fair["fair_b"]

        game_detail_home = {
            "game_id": game_id,
            "opponent": away_team,
            "location": "home",
            "fair_win_prob": round(home_fair_prob, 4),
            "consensus_odds": round(avg_home_odds, 1),
            "book_count": ml_consensus.get("book_count", 0),
        }
        game_detail_away = {
            "game_id": game_id,
            "opponent": home_team,
            "location": "away",
            "fair_win_prob": round(away_fair_prob, 4),
            "consensus_odds": round(avg_away_odds, 1),
            "book_count": ml_consensus.get("book_count", 0),
        }

        team_games.setdefault(home_team, []).append(game_detail_home)
        team_games.setdefault(away_team, []).append(game_detail_away)

    # Build rankings: average fair win probability across all appearances
    rankings = []
    for team, games in team_games.items():
        probs = [g["fair_win_prob"] for g in games]
        avg_prob = sum(probs) / len(probs)
        rankings.append({
            "team": team,
            "strength_rating": round(avg_prob, 4),
            "strength_pct": f"{round(avg_prob * 100, 1)}%",
            "games_sampled": len(games),
            "game_details": sorted(games, key=lambda g: g["fair_win_prob"], reverse=True),
        })

    # Sort strongest to weakest
    rankings.sort(key=lambda r: r["strength_rating"], reverse=True)

    # Assign rank numbers
    for i, r in enumerate(rankings, 1):
        r["rank"] = i

    # Tier labels for readability
    for r in rankings:
        pct = r["strength_rating"] * 100
        if pct >= 65:
            r["tier"] = "elite"
        elif pct >= 55:
            r["tier"] = "strong"
        elif pct >= 45:
            r["tier"] = "average"
        elif pct >= 35:
            r["tier"] = "below_average"
        else:
            r["tier"] = "weak"

    result = {
        "power_rankings": rankings,
        "total_teams": len(rankings),
        "methodology": (
            "Each team's consensus (cross-book average) moneyline odds are converted "
            "to no-vig fair win probabilities.  A team's strength rating is the average "
            "of its fair win probabilities across all games it appears in.  Higher = "
            "the market views the team as stronger."
        ),
        "context": (
            f"Power rankings for {len(rankings)} teams derived from moneyline odds "
            f"across {len(by_game)} games.  Ratings reflect market-implied strength, "
            f"not predictions — a team priced as a heavy favorite in easy matchups may "
            f"rate higher than a good team facing tough opponents."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)




# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Sharpness & Cross-Market Correlation
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_sharpness_scores(filename: Optional[str] = None, benchmark_book: str = "Pinnacle") -> str:
    """Score each sportsbook's sharpness by how closely their lines track a benchmark sharp book (default: Pinnacle).

    Pinnacle is widely considered the sharpest book -- their lines are set by
    high-volume sharp action and carry the lowest vig. A sportsbook whose lines
    closely mirror Pinnacle is likely sharp and fast-updating. Books that
    consistently deviate are slower to adjust or cater to recreational bettors.

    Metrics per book:
    - avg_spread_line_diff: Mean absolute difference in spread line vs benchmark
    - avg_odds_diff: Mean absolute difference in implied probability vs benchmark across all markets
    - max_deviation: Largest single implied-prob deviation from benchmark
    - correlation: Pearson correlation of implied probabilities with benchmark (1.0 = identical)
    - sharpness_score: 0-100 composite (100 = perfectly sharp / tracks benchmark exactly)

    Args:
        filename: Data file to load. Optional.
        benchmark_book: The sharp book to use as baseline. Default "Pinnacle".

    Returns a ranked list of sportsbooks by sharpness score with per-game detail.
    """
    cache_key = f"sharpness:{benchmark_book}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)

    # Collect benchmark data per game
    benchmark_data: dict[str, dict] = {}  # game_id -> markets dict
    for game_id, records in games.items():
        for r in records:
            if r.get("sportsbook", "").lower() == benchmark_book.lower():
                benchmark_data[game_id] = r.get("markets", {})
                break

    if not benchmark_data:
        return json.dumps({
            "error": f"Benchmark book '{benchmark_book}' not found in dataset. "
                     f"Available books: {sorted(set(r.get('sportsbook','') for r in odds))}",
        })

    # For each non-benchmark book, compute deviations from benchmark per game
    book_deviations: dict[str, dict] = {}

    for game_id, records in games.items():
        if game_id not in benchmark_data:
            continue
        bench_markets = benchmark_data[game_id]

        for r in records:
            book = r.get("sportsbook", "")
            if book.lower() == benchmark_book.lower():
                continue

            if book not in book_deviations:
                book_deviations[book] = {
                    "prob_diffs": [],
                    "spread_line_diffs": [],
                    "total_line_diffs": [],
                    "max_dev": 0.0,
                    "bench_probs": [],
                    "book_probs": [],
                    "game_details": [],
                }

            stats = book_deviations[book]
            game_detail = {"game_id": game_id, "deviations": {}}

            for market_type in ("spread", "moneyline", "total"):
                book_market = r.get("markets", {}).get(market_type, {})
                bench_market = bench_markets.get(market_type, {})
                if not book_market or not bench_market:
                    continue

                if market_type in ("spread", "moneyline"):
                    odds_keys = [("home_odds", "home"), ("away_odds", "away")]
                else:
                    odds_keys = [("over_odds", "over"), ("under_odds", "under")]

                for odds_key, side in odds_keys:
                    if odds_key in book_market and odds_key in bench_market:
                        book_prob = implied_probability(book_market[odds_key])
                        bench_prob = implied_probability(bench_market[odds_key])
                        diff = abs(book_prob - bench_prob)
                        stats["prob_diffs"].append(diff)
                        stats["bench_probs"].append(bench_prob)
                        stats["book_probs"].append(book_prob)
                        if diff > stats["max_dev"]:
                            stats["max_dev"] = diff

                # Spread line deviation
                if market_type == "spread" and "home_line" in book_market and "home_line" in bench_market:
                    line_diff = abs(book_market["home_line"] - bench_market["home_line"])
                    stats["spread_line_diffs"].append(line_diff)
                    if line_diff > 0:
                        game_detail["deviations"]["spread_line"] = {
                            "book_line": book_market["home_line"],
                            "benchmark_line": bench_market["home_line"],
                            "diff": line_diff,
                        }

                # Total line deviation
                if market_type == "total" and "line" in book_market and "line" in bench_market:
                    line_diff = abs(book_market["line"] - bench_market["line"])
                    stats["total_line_diffs"].append(line_diff)
                    if line_diff > 0:
                        game_detail["deviations"]["total_line"] = {
                            "book_line": book_market["line"],
                            "benchmark_line": bench_market["line"],
                            "diff": line_diff,
                        }

            if game_detail["deviations"]:
                stats["game_details"].append(game_detail)

    # Build rankings
    rankings = []
    for book, stats in book_deviations.items():
        if not stats["prob_diffs"]:
            continue

        avg_prob_diff = sum(stats["prob_diffs"]) / len(stats["prob_diffs"])
        avg_spread_diff = (
            sum(stats["spread_line_diffs"]) / len(stats["spread_line_diffs"])
            if stats["spread_line_diffs"] else 0
        )
        avg_total_diff = (
            sum(stats["total_line_diffs"]) / len(stats["total_line_diffs"])
            if stats["total_line_diffs"] else 0
        )

        # Pearson correlation of implied probabilities with benchmark
        correlation = 0.0
        n = len(stats["bench_probs"])
        if n >= 2:
            mean_b = sum(stats["bench_probs"]) / n
            mean_k = sum(stats["book_probs"]) / n
            cov = sum(
                (b - mean_b) * (k - mean_k)
                for b, k in zip(stats["bench_probs"], stats["book_probs"])
            ) / n
            std_b = sqrt(sum((b - mean_b) ** 2 for b in stats["bench_probs"]) / n)
            std_k = sqrt(sum((k - mean_k) ** 2 for k in stats["book_probs"]) / n)
            if std_b > 0 and std_k > 0:
                correlation = cov / (std_b * std_k)

        # Sharpness score: 0-100
        # avg prob diff (40), spread line diff (20), total line diff (10),
        # max deviation (15), correlation (15)
        prob_score = max(0, 1 - avg_prob_diff / 0.05) * 40
        spread_score = max(0, 1 - avg_spread_diff / 3.0) * 20
        total_score = max(0, 1 - avg_total_diff / 3.0) * 10
        max_dev_score = max(0, 1 - stats["max_dev"] / 0.08) * 15
        corr_score = max(0, correlation) * 15

        sharpness = round(
            prob_score + spread_score + total_score + max_dev_score + corr_score, 1
        )

        rankings.append({
            "sportsbook": book,
            "sharpness_score": sharpness,
            "avg_prob_diff": round(avg_prob_diff, 6),
            "avg_prob_diff_pct": f"{round(avg_prob_diff * 100, 2)}%",
            "avg_spread_line_diff": round(avg_spread_diff, 2),
            "avg_total_line_diff": round(avg_total_diff, 2),
            "max_deviation": round(stats["max_dev"], 6),
            "max_deviation_pct": f"{round(stats['max_dev'] * 100, 2)}%",
            "correlation_with_benchmark": round(correlation, 4),
            "samples": len(stats["prob_diffs"]),
            "notable_deviations": stats["game_details"][:3],
        })

    rankings.sort(key=lambda r: r["sharpness_score"], reverse=True)

    # Classify books
    for r in rankings:
        s = r["sharpness_score"]
        if s >= 85:
            r["classification"] = "sharp"
        elif s >= 70:
            r["classification"] = "semi-sharp"
        elif s >= 50:
            r["classification"] = "moderate"
        else:
            r["classification"] = "recreational"

    result = {
        "benchmark_book": benchmark_book,
        "rankings": rankings,
        "sharpest_book": rankings[0] if rankings else None,
        "softest_book": rankings[-1] if rankings else None,
        "context": (
            f"Sharpness scores vs {benchmark_book}. "
            + (
                f"Sharpest: {rankings[0]['sportsbook']} "
                f"({rankings[0]['sharpness_score']}/100, {rankings[0]['classification']}). "
                f"Softest: {rankings[-1]['sportsbook']} "
                f"({rankings[-1]['sharpness_score']}/100, {rankings[-1]['classification']})."
                if rankings else "No comparison data."
            )
        ),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def get_market_correlations(game_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Analyze correlations between spread, moneyline, and total markets across sportsbooks.

    Checks whether a book's pricing is internally consistent:
    - Spread vs Moneyline: A bigger spread favorite should correspond to shorter ML
      odds. Mismatches can reveal mispriced markets.
    - Spread/ML vs Total: Examines whether the total line moves with the spread
      (e.g., high-total games with big spreads may indicate an expected blowout).

    Also detects per-book inconsistencies where one market implies a very different
    probability than another for the same outcome.

    Args:
        game_id: Filter to a specific game. Optional (analyzes all games).
        filename: Data file to load. Optional.

    Returns correlation data, per-book consistency scores, and flagged inconsistencies.
    """
    cache_key = f"market_corr:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games_grouped = _group_by_game(enriched)

    # Collect paired data points for correlation analysis
    spread_ml_pairs: list[tuple[float, float]] = []
    spread_total_pairs: list[tuple[float, float]] = []
    inconsistencies: list[dict] = []
    book_consistency: dict[str, list[float]] = {}

    for gid, records in games_grouped.items():
        first = records[0]

        for r in records:
            book = r.get("sportsbook", "")
            markets = r.get("markets", {})
            spread = markets.get("spread", {})
            ml = markets.get("moneyline", {})
            total = markets.get("total", {})

            # --- Spread vs Moneyline consistency ---
            if spread and ml:
                spread_home_prob = spread.get("home_implied_prob", 0)
                ml_home_prob = ml.get("home_implied_prob", 0)

                if spread_home_prob > 0 and ml_home_prob > 0:
                    spread_ml_pairs.append((spread_home_prob, ml_home_prob))
                    diff = abs(spread_home_prob - ml_home_prob)

                    book_consistency.setdefault(book, []).append(diff)

                    # Flag large inconsistencies (>8% gap)
                    if diff > 0.08:
                        inconsistencies.append({
                            "game_id": gid,
                            "sport": first.get("sport"),
                            "home_team": first.get("home_team"),
                            "away_team": first.get("away_team"),
                            "sportsbook": book,
                            "type": "spread_vs_moneyline",
                            "spread_home_implied": round(spread_home_prob, 4),
                            "spread_home_implied_pct": f"{round(spread_home_prob * 100, 1)}%",
                            "ml_home_implied": round(ml_home_prob, 4),
                            "ml_home_implied_pct": f"{round(ml_home_prob * 100, 1)}%",
                            "diff_pct": f"{round(diff * 100, 1)}%",
                            "spread_home_line": spread.get("home_line"),
                            "spread_home_odds": spread.get("home_odds"),
                            "ml_home_odds": ml.get("home_odds"),
                            "context": (
                                f"INCONSISTENCY at {book}: Spread implies "
                                f"{round(spread_home_prob * 100, 1)}% home win "
                                f"but ML implies {round(ml_home_prob * 100, 1)}% "
                                f"({round(diff * 100, 1)}% gap). "
                                f"Possible mispricing opportunity."
                            ),
                        })

            # --- Spread vs Total pairing ---
            if spread and total:
                home_line = spread.get("home_line")
                total_line = total.get("line")
                if home_line is not None and total_line is not None:
                    spread_total_pairs.append((abs(home_line), total_line))

    # Compute Pearson correlations
    def _pearson(pairs):
        if len(pairs) < 3:
            return None
        n = len(pairs)
        xs, ys = zip(*pairs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / n
        std_x = sqrt(sum((x - mean_x) ** 2 for x in xs) / n)
        std_y = sqrt(sum((y - mean_y) ** 2 for y in ys) / n)
        if std_x == 0 or std_y == 0:
            return None
        return round(cov / (std_x * std_y), 4)

    spread_ml_corr = _pearson(spread_ml_pairs)
    spread_total_corr = _pearson(spread_total_pairs)

    # Build per-book consistency scores
    book_scores = []
    for book, diffs in book_consistency.items():
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        # Score: 100 = perfectly consistent, 0 = 10%+ average gap
        consistency_score = round(max(0, (1 - avg_diff / 0.10)) * 100, 1)
        book_scores.append({
            "sportsbook": book,
            "consistency_score": consistency_score,
            "avg_spread_ml_diff_pct": f"{round(avg_diff * 100, 2)}%",
            "max_spread_ml_diff_pct": f"{round(max_diff * 100, 2)}%",
            "samples": len(diffs),
        })
    book_scores.sort(key=lambda b: b["consistency_score"], reverse=True)

    inconsistencies.sort(
        key=lambda i: float(i["diff_pct"].rstrip("%")), reverse=True
    )

    # Interpretation helpers
    def _interpret_corr(label, val):
        if val is None:
            return f"{label}: insufficient data"
        strength = (
            "strong" if abs(val) > 0.7
            else "moderate" if abs(val) > 0.4
            else "weak"
        )
        direction = "positive" if val > 0 else "negative"
        return f"{label}: {val} ({strength} {direction})"

    result = {
        "correlations": {
            "spread_vs_moneyline": {
                "pearson_r": spread_ml_corr,
                "sample_count": len(spread_ml_pairs),
                "interpretation": _interpret_corr(
                    "Spread vs Moneyline implied prob", spread_ml_corr
                ),
                "note": (
                    "Expected: strong positive correlation (~0.9+). Spread and ML "
                    "should imply similar win probabilities. Lower values suggest "
                    "market inefficiencies."
                ),
            },
            "spread_size_vs_total": {
                "pearson_r": spread_total_corr,
                "sample_count": len(spread_total_pairs),
                "interpretation": _interpret_corr(
                    "|Spread| vs Total line", spread_total_corr
                ),
                "note": (
                    "Positive correlation means bigger favorites tend to have higher "
                    "totals (expected blowouts). Weak/no correlation is normal for "
                    "some sports."
                ),
            },
        },
        "book_consistency_scores": book_scores,
        "inconsistencies": inconsistencies,
        "inconsistency_count": len(inconsistencies),
        "most_consistent_book": book_scores[0] if book_scores else None,
        "least_consistent_book": book_scores[-1] if book_scores else None,
        "context": (
            f"Market correlations across {len(games_grouped)} games. "
            f"Spread vs ML r={spread_ml_corr}, "
            f"|Spread| vs Total r={spread_total_corr}. "
            f"{len(inconsistencies)} cross-market inconsistencies flagged "
            f"(>8% spread-ML gap). "
            + (
                f"Most consistent: {book_scores[0]['sportsbook']} "
                f"({book_scores[0]['consistency_score']}/100). "
                f"Least consistent: {book_scores[-1]['sportsbook']} "
                f"({book_scores[-1]['consistency_score']}/100)."
                if book_scores else ""
            )
        ),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)



@mcp.tool()
def get_synthetic_hold_free_market(game_id: Optional[str] = None, filename: Optional[str] = None) -> str:
    """Build a synthetic "perfect book" by combining the best available odds across all sportsbooks for every side of every market.

    For each game and market, picks the highest (best-for-bettor) odds offered
    on each side across all books.  The resulting synthetic book represents
    what a sharp bettor could achieve by always line-shopping.

    Calculates:
    - The synthetic hold (combined implied probability minus 1).  A negative hold
      means the bettor has a guaranteed edge (arbitrage territory).
    - The total sharp-bettor edge: how much better the best-available odds are
      compared to the consensus fair odds.
    - Per-market and aggregate statistics across the entire dataset.

    Args:
        game_id: Filter to a specific game. Optional (shows all games).
        filename: Data file to load. Optional.

    Returns per-game synthetic hold-free market data plus aggregate statistics.
    """
    cache_key = f"synthetic_hold_free:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No odds data found"})

    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    synthetic_games = []
    all_holds = []            # every market hold value (for aggregate stats)
    all_edges = []            # every market edge value
    market_type_holds: dict[str, list[float]] = {"spread": [], "moneyline": [], "total": []}
    market_type_edges: dict[str, list[float]] = {"spread": [], "moneyline": [], "total": []}

    for gid, records in games.items():
        first = records[0]
        game_entry = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "book_count": len(records),
            "markets": {},
        }

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                side_a_key, side_b_key = "home_odds", "away_odds"
                side_a_label, side_b_label = "home", "away"
            else:
                side_a_key, side_b_key = "over_odds", "under_odds"
                side_a_label, side_b_label = "over", "under"

            # Find best odds for each side across all books
            best_a = {"odds": -99999, "book": "", "line": None}
            best_b = {"odds": -99999, "book": "", "line": None}

            for r in records:
                market = r.get("markets", {}).get(market_type, {})
                if side_a_key in market and market[side_a_key] > best_a["odds"]:
                    best_a = {
                        "odds": market[side_a_key],
                        "book": r["sportsbook"],
                        "line": market.get("home_line") if market_type == "spread" else market.get("line"),
                    }
                if side_b_key in market and market[side_b_key] > best_b["odds"]:
                    best_b = {
                        "odds": market[side_b_key],
                        "book": r["sportsbook"],
                        "line": market.get("home_line") if market_type == "spread" else market.get("line"),
                    }

            if not best_a["book"] or not best_b["book"]:
                continue

            prob_a = implied_probability(best_a["odds"])
            prob_b = implied_probability(best_b["odds"])
            combined_implied = prob_a + prob_b
            synthetic_hold = combined_implied - 1.0   # negative = bettor edge

            # Calculate fair probabilities from consensus to measure edge
            market_records = [r["markets"][market_type] for r in records if market_type in r.get("markets", {})]
            if market_type in ("spread", "moneyline"):
                raw_probs_a = [m.get("home_implied_prob", 0) for m in market_records]
                raw_probs_b = [m.get("away_implied_prob", 0) for m in market_records]
            else:
                raw_probs_a = [m.get("over_implied_prob", 0) for m in market_records]
                raw_probs_b = [m.get("under_implied_prob", 0) for m in market_records]

            avg_a = sum(raw_probs_a) / len(raw_probs_a) if raw_probs_a else 0.5
            avg_b = sum(raw_probs_b) / len(raw_probs_b) if raw_probs_b else 0.5
            total_raw = avg_a + avg_b
            fair_a = avg_a / total_raw if total_raw else 0.5
            fair_b = avg_b / total_raw if total_raw else 0.5

            # Edge: difference between fair prob and best-available implied prob
            edge_a = fair_a - prob_a   # positive = bettor has value on side A
            edge_b = fair_b - prob_b

            market_entry = {
                f"best_{side_a_label}": {
                    "odds": best_a["odds"],
                    "sportsbook": best_a["book"],
                    "implied_prob": round(prob_a, 6),
                    "implied_prob_pct": f"{round(prob_a * 100, 2)}%",
                    "fair_prob": round(fair_a, 6),
                    "edge_vs_fair": round(edge_a * 100, 3),
                },
                f"best_{side_b_label}": {
                    "odds": best_b["odds"],
                    "sportsbook": best_b["book"],
                    "implied_prob": round(prob_b, 6),
                    "implied_prob_pct": f"{round(prob_b * 100, 2)}%",
                    "fair_prob": round(fair_b, 6),
                    "edge_vs_fair": round(edge_b * 100, 3),
                },
                "combined_implied_prob": round(combined_implied, 6),
                "combined_implied_pct": f"{round(combined_implied * 100, 2)}%",
                "synthetic_hold": round(synthetic_hold, 6),
                "synthetic_hold_pct": f"{round(synthetic_hold * 100, 3)}%",
                "total_sharp_edge_pct": round(-synthetic_hold * 100, 3),
                "is_arb": combined_implied < 1.0,
            }

            if market_type == "spread":
                lines = [m.get("home_line", 0) for m in market_records if "home_line" in m]
                if lines:
                    market_entry["consensus_line"] = round(sum(lines) / len(lines), 1)
            elif market_type == "total":
                lines = [m.get("line", 0) for m in market_records if "line" in m]
                if lines:
                    market_entry["consensus_line"] = round(sum(lines) / len(lines), 1)

            game_entry["markets"][market_type] = market_entry
            all_holds.append(synthetic_hold)
            all_edges.append(-synthetic_hold)
            market_type_holds[market_type].append(synthetic_hold)
            market_type_edges[market_type].append(-synthetic_hold)

        if game_entry["markets"]:
            # Per-game aggregate
            game_holds = [m["synthetic_hold"] for m in game_entry["markets"].values()]
            game_entry["avg_synthetic_hold_pct"] = f"{round(sum(game_holds) / len(game_holds) * 100, 3)}%"
            game_entry["avg_sharp_edge_pct"] = f"{round(-sum(game_holds) / len(game_holds) * 100, 3)}%"
            game_entry["arb_markets"] = sum(1 for m in game_entry["markets"].values() if m["is_arb"])
            synthetic_games.append(game_entry)

    # Sort games by total sharp edge (most edge first)
    synthetic_games.sort(key=lambda g: -sum(m["synthetic_hold"] for m in g["markets"].values()) / len(g["markets"]), reverse=True)

    # Aggregate stats
    avg_hold = sum(all_holds) / len(all_holds) if all_holds else 0
    avg_edge = sum(all_edges) / len(all_edges) if all_edges else 0
    arb_count = sum(1 for h in all_holds if h < 0)

    by_market_summary = {}
    for mkt in ("spread", "moneyline", "total"):
        holds = market_type_holds[mkt]
        edges = market_type_edges[mkt]
        if holds:
            by_market_summary[mkt] = {
                "avg_synthetic_hold_pct": f"{round(sum(holds) / len(holds) * 100, 3)}%",
                "avg_sharp_edge_pct": f"{round(sum(edges) / len(edges) * 100, 3)}%",
                "min_hold_pct": f"{round(min(holds) * 100, 3)}%",
                "max_hold_pct": f"{round(max(holds) * 100, 3)}%",
                "arb_markets": sum(1 for h in holds if h < 0),
                "market_count": len(holds),
            }

    result = {
        "games": synthetic_games,
        "game_count": len(synthetic_games),
        "aggregate": {
            "total_markets_analyzed": len(all_holds),
            "avg_synthetic_hold_pct": f"{round(avg_hold * 100, 3)}%",
            "avg_sharp_edge_pct": f"{round(avg_edge * 100, 3)}%",
            "arb_market_count": arb_count,
            "best_edge_pct": f"{round(max(all_edges) * 100, 3)}%" if all_edges else "N/A",
            "worst_edge_pct": f"{round(min(all_edges) * 100, 3)}%" if all_edges else "N/A",
            "by_market_type": by_market_summary,
        },
        "methodology": (
            "For each game and market, selects the best (highest) odds available on "
            "each side across all sportsbooks to construct a synthetic 'perfect book'. "
            "The synthetic hold is the combined implied probability minus 1 — when negative, "
            "the bettor has a guaranteed edge.  The sharp-bettor edge shows how much value "
            "a disciplined line-shopper captures compared to fair odds."
        ),
        "context": (
            f"Synthetic hold-free market across {len(synthetic_games)} games and "
            f"{len(all_holds)} markets.  Average synthetic hold: {round(avg_hold * 100, 3)}% "
            f"(sharp edge: {round(avg_edge * 100, 3)}%).  "
            f"{arb_count} market(s) in pure arbitrage territory (hold < 0%)."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)




# ═══════════════════════════════════════════════════════════════════════════
# CLUSTER ANALYSIS — Sportsbook Pricing Similarity
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_sportsbook_clusters(filename: Optional[str] = None) -> str:
    """Cluster sportsbooks by how similarly they price lines across all games.

    Groups books that consistently post near-identical odds — revealing which
    sportsbooks likely share the same odds feed, risk model, or parent company.
    Also identifies books that price independently (e.g., Pinnacle).

    Methodology:
    1. For every game x market x side, each sportsbook contributes an odds value.
    2. A pairwise "distance" is computed for each sportsbook pair as the average
       absolute difference in American odds across all shared data points.
    3. Agglomerative (average-linkage) clustering groups books whose average
       pairwise distance falls below a configurable threshold.
    4. A correlation matrix and agreement-rate matrix add additional context.

    Args:
        filename: Data file to load.  Optional — defaults to the first
                  available file in the data directory.

    Returns JSON with:
        clusters          – list of clusters (groups of similar books)
        distance_matrix   – full pairwise avg-odds-distance table
        agreement_matrix  – pct of markets where two books post identical odds
        independents      – books that don't cluster tightly with anyone
        insights          – human-readable takeaways
    """
    cache_key = "sportsbook_clusters"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data available"})

    by_game = _cache.load_by_game(filename)

    # ------------------------------------------------------------------
    # Step 1: Build a per-sportsbook odds vector.
    # Each vector element is keyed by (game_id, market_type, side) and the
    # value is the American odds posted by that book.
    # ------------------------------------------------------------------
    book_vectors: dict[str, dict[tuple, float]] = {}

    for game_id, records in by_game.items():
        for record in records:
            book = record["sportsbook"]
            markets = record.get("markets", {})

            for market_type, market_data in markets.items():
                if market_type == "spread":
                    sides = [
                        ("home_spread", market_data.get("home_odds")),
                        ("away_spread", market_data.get("away_odds")),
                        ("home_line", market_data.get("home_line")),
                        ("away_line", market_data.get("away_line")),
                    ]
                elif market_type == "moneyline":
                    sides = [
                        ("home_ml", market_data.get("home_odds")),
                        ("away_ml", market_data.get("away_odds")),
                    ]
                elif market_type == "total":
                    sides = [
                        ("over", market_data.get("over_odds")),
                        ("under", market_data.get("under_odds")),
                        ("total_line", market_data.get("line")),
                    ]
                else:
                    continue

                for side_label, odds_val in sides:
                    if odds_val is None:
                        continue
                    key = (game_id, market_type, side_label)
                    book_vectors.setdefault(book, {})[key] = float(odds_val)

    books = sorted(book_vectors.keys())
    n = len(books)

    if n < 2:
        return json.dumps({"error": "Need at least 2 sportsbooks for cluster analysis"})

    # ------------------------------------------------------------------
    # Step 2: Pairwise distance & agreement matrices
    # Distance = mean |odds_a - odds_b| across all shared data points
    # Agreement = fraction of shared points where odds are exactly equal
    # ------------------------------------------------------------------
    dist_matrix: dict[str, dict[str, float]] = {b: {} for b in books}
    agree_matrix: dict[str, dict[str, float]] = {b: {} for b in books}
    pair_details: dict[tuple, dict] = {}

    for i in range(n):
        for j in range(i, n):
            ba, bb = books[i], books[j]
            shared_keys = set(book_vectors[ba].keys()) & set(book_vectors[bb].keys())
            if not shared_keys:
                dist_matrix[ba][bb] = dist_matrix[bb][ba] = float("inf")
                agree_matrix[ba][bb] = agree_matrix[bb][ba] = 0.0
                continue

            diffs = []
            exact_matches = 0
            for k in shared_keys:
                va, vb = book_vectors[ba][k], book_vectors[bb][k]
                diffs.append(abs(va - vb))
                if va == vb:
                    exact_matches += 1

            avg_diff = mean(diffs) if diffs else 0.0
            agree_pct = (exact_matches / len(shared_keys) * 100) if shared_keys else 0.0

            dist_matrix[ba][bb] = round(avg_diff, 2)
            dist_matrix[bb][ba] = round(avg_diff, 2)
            agree_matrix[ba][bb] = round(agree_pct, 1)
            agree_matrix[bb][ba] = round(agree_pct, 1)

            if i != j:
                pair_details[(ba, bb)] = {
                    "avg_odds_diff": round(avg_diff, 2),
                    "agreement_pct": round(agree_pct, 1),
                    "shared_data_points": len(shared_keys),
                    "exact_matches": exact_matches,
                    "max_diff": round(max(diffs), 1) if diffs else 0,
                }

    # ------------------------------------------------------------------
    # Step 3: Agglomerative clustering (average-linkage)
    # Merge the closest pair of clusters until the minimum inter-cluster
    # distance exceeds the threshold.
    # ------------------------------------------------------------------
    CLUSTER_THRESHOLD = 8.0  # avg odds diff <= 8 pts -> same cluster

    # Start with each book in its own cluster
    clusters: list[set[str]] = [{b} for b in books]

    def _cluster_dist(c1: set[str], c2: set[str]) -> float:
        """Average-linkage distance between two clusters."""
        dists = []
        for a in c1:
            for b in c2:
                d = dist_matrix[a].get(b, inf)
                if d < inf:
                    dists.append(d)
        return mean(dists) if dists else inf

    while True:
        best_dist = inf
        merge_i, merge_j = -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = _cluster_dist(clusters[i], clusters[j])
                if d < best_dist:
                    best_dist = d
                    merge_i, merge_j = i, j
        if best_dist > CLUSTER_THRESHOLD or merge_i < 0:
            break
        clusters[merge_i] = clusters[merge_i] | clusters[merge_j]
        clusters.pop(merge_j)

    # ------------------------------------------------------------------
    # Step 4: Label clusters and identify independents
    # ------------------------------------------------------------------
    cluster_results = []
    independents = []

    for idx, members in enumerate(sorted(clusters, key=len, reverse=True)):
        sorted_members = sorted(members)
        if len(members) == 1:
            book = sorted_members[0]
            # Find its closest neighbor
            closest = None
            closest_dist = inf
            for other in books:
                if other == book:
                    continue
                d = dist_matrix[book].get(other, inf)
                if d < closest_dist:
                    closest_dist = d
                    closest = other
            independents.append({
                "sportsbook": book,
                "closest_peer": closest,
                "distance_to_closest": round(closest_dist, 2),
                "interpretation": (
                    f"{book} prices independently — closest to {closest} "
                    f"(avg diff {round(closest_dist, 1)} pts) but still outside "
                    f"the clustering threshold of {CLUSTER_THRESHOLD} pts."
                ),
            })
            continue

        # Intra-cluster stats
        intra_dists = []
        intra_agreements = []
        for i, a in enumerate(sorted_members):
            for b in sorted_members[i + 1:]:
                intra_dists.append(dist_matrix[a][b])
                intra_agreements.append(agree_matrix[a][b])

        avg_intra = round(mean(intra_dists), 2) if intra_dists else 0
        avg_agree = round(mean(intra_agreements), 1) if intra_agreements else 0

        # Pairwise breakdown within cluster
        pairwise = []
        for i, a in enumerate(sorted_members):
            for b in sorted_members[i + 1:]:
                key = (a, b) if (a, b) in pair_details else (b, a)
                detail = pair_details.get(key, {})
                pairwise.append({
                    "books": [a, b],
                    "avg_odds_diff": detail.get("avg_odds_diff", 0),
                    "agreement_pct": detail.get("agreement_pct", 0),
                })

        cluster_results.append({
            "cluster_id": idx + 1,
            "members": sorted_members,
            "size": len(members),
            "avg_internal_distance": avg_intra,
            "avg_agreement_pct": avg_agree,
            "likely_shared_feed": avg_intra <= 3.0,
            "pairwise_details": sorted(pairwise, key=lambda p: p["avg_odds_diff"]),
            "interpretation": (
                f"These {len(members)} books price very similarly "
                f"(avg diff {avg_intra} pts, {avg_agree}% exact agreement). "
                + ("Likely sharing an odds feed or risk model." if avg_intra <= 3.0
                   else "Similar pricing philosophy but some independent adjustments.")
            ),
        })

    # ------------------------------------------------------------------
    # Step 5: Build sorted pair rankings (most similar -> least)
    # ------------------------------------------------------------------
    all_pairs = []
    for (a, b), detail in sorted(pair_details.items()):
        all_pairs.append({
            "books": [a, b],
            **detail,
        })
    all_pairs.sort(key=lambda p: p["avg_odds_diff"])

    # ------------------------------------------------------------------
    # Step 6: Generate insights
    # ------------------------------------------------------------------
    insights = []

    if cluster_results:
        biggest = cluster_results[0]
        insights.append(
            f"Largest cluster: {', '.join(biggest['members'])} "
            f"({biggest['size']} books, avg diff {biggest['avg_internal_distance']} pts)"
        )
        if biggest.get("likely_shared_feed"):
            insights.append(
                f"⚠️ {', '.join(biggest['members'])} likely share the same odds feed — "
                f"shopping between them adds little value."
            )

    if independents:
        indie_names = [ind["sportsbook"] for ind in independents]
        insights.append(
            f"Independent pricers: {', '.join(indie_names)} — "
            f"these books set their own lines and are valuable for line shopping."
        )

    if all_pairs:
        closest = all_pairs[0]
        insights.append(
            f"Most similar pair: {closest['books'][0]} & {closest['books'][1]} "
            f"(avg diff {closest['avg_odds_diff']} pts, {closest['agreement_pct']}% agreement)"
        )
        farthest = all_pairs[-1]
        insights.append(
            f"Most different pair: {farthest['books'][0]} & {farthest['books'][1]} "
            f"(avg diff {farthest['avg_odds_diff']} pts, {farthest['agreement_pct']}% agreement)"
        )

    result = {
        "clusters": cluster_results,
        "independents": independents,
        "most_similar_pairs": all_pairs[:5],
        "most_different_pairs": all_pairs[-5:][::-1] if len(all_pairs) >= 5 else list(reversed(all_pairs)),
        "distance_matrix": {b: {b2: dist_matrix[b][b2] for b2 in books} for b in books},
        "agreement_matrix": {b: {b2: agree_matrix[b][b2] for b2 in books} for b in books},
        "insights": insights,
        "methodology": (
            "For every game x market x side, each sportsbook's American odds (and lines "
            "for spread/totals) form a vector. Pairwise distance = mean absolute difference "
            "across all shared data points. Books are clustered using average-linkage "
            f"agglomerative clustering with a threshold of {CLUSTER_THRESHOLD} points. "
            "High agreement % + low distance -> likely shared odds feed or risk model."
        ),
        "cluster_threshold": CLUSTER_THRESHOLD,
        "total_sportsbooks": n,
        "total_data_points": sum(len(v) for v in book_vectors.values()),
        "context": (
            f"Clustered {n} sportsbooks into {len(cluster_results)} group(s) "
            f"with {len(independents)} independent pricer(s). "
            f"Books in the same cluster tend to share odds feeds or risk models."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# RESOURCES — Context Data
# ═══════════════════════════════════════════════════════════════════════════



@mcp.tool()
def get_zscore_anomalies(
    game_id: Optional[str] = None,
    filename: Optional[str] = None,
    z_threshold: float = 2.0,
) -> str:
    """Detect anomalous odds using z-score analysis across sportsbooks.

    For each game's market (spread, moneyline, total), computes the mean and
    standard deviation of odds across all books, then flags any individual
    book's odds whose absolute z-score exceeds the threshold.

    High z-scores indicate the book is pricing significantly differently from
    the consensus - could signal a stale line, pricing error, or sharp move.

    Args:
        game_id: Optional game to limit analysis to.
        filename: Data file to load. Optional.
        z_threshold: Minimum absolute z-score to flag (default 2.0).

    Returns anomalies sorted by z-score descending with full context.
    """
    cache_key = f"zscore_anomalies:{game_id or 'all'}:{z_threshold}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    anomalies: list[dict] = []

    target_games = {game_id: games[game_id]} if game_id and game_id in games else games

    for gid, records in target_games.items():
        first = records[0]

        market_keys_map = {
            "spread": ["home_odds", "away_odds"],
            "moneyline": ["home_odds", "away_odds"],
            "total": ["over_odds", "under_odds"],
        }

        for market_type, odds_keys in market_keys_map.items():
            for odds_key in odds_keys:
                entries = []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key in market:
                        entries.append({
                            "sportsbook": r["sportsbook"],
                            "odds": market[odds_key],
                            "last_updated": r.get("last_updated"),
                        })

                if len(entries) < 3:
                    continue

                values = [e["odds"] for e in entries]
                mu = mean(values)
                sigma = pstdev(values)

                if sigma == 0:
                    continue

                for e in entries:
                    z = (e["odds"] - mu) / sigma
                    if abs(z) >= z_threshold:
                        side = odds_key.replace("_odds", "")
                        direction = (
                            "better_for_bettor" if e["odds"] > mu else "worse_for_bettor"
                        )
                        anomalies.append({
                            "game_id": gid,
                            "sport": first.get("sport"),
                            "home_team": first.get("home_team"),
                            "away_team": first.get("away_team"),
                            "market_type": market_type,
                            "side": side,
                            "sportsbook": e["sportsbook"],
                            "odds": e["odds"],
                            "z_score": round(z, 3),
                            "abs_z_score": round(abs(z), 3),
                            "mean_odds": round(mu, 2),
                            "std_dev": round(sigma, 2),
                            "deviation": round(e["odds"] - mu, 2),
                            "direction": direction,
                            "last_updated": e.get("last_updated"),
                            "context": (
                                f"Z-SCORE ANOMALY: {e['sportsbook']} {side} {market_type} "
                                f"at {e['odds']} (z={round(z, 2)}, mean={round(mu, 1)}, "
                                f"σ={round(sigma, 1)}). "
                                f"{'Potential value - priced above market!' if direction == 'better_for_bettor' else 'Avoid - priced below market.'}"
                            ),
                        })

        # Also check spread LINES and total LINES for z-score anomalies
        line_checks = [
            ("spread", "home_line", "spread line"),
            ("total", "line", "total line"),
        ]
        for market_type, line_key, label in line_checks:
            entries = []
            for r in records:
                market = r.get("markets", {}).get(market_type, {})
                if line_key in market:
                    entries.append({
                        "sportsbook": r["sportsbook"],
                        "line": market[line_key],
                        "last_updated": r.get("last_updated"),
                    })

            if len(entries) < 3:
                continue

            values = [e["line"] for e in entries]
            mu = mean(values)
            sigma = pstdev(values)

            if sigma == 0:
                continue

            for e in entries:
                z = (e["line"] - mu) / sigma
                if abs(z) >= z_threshold:
                    anomalies.append({
                        "game_id": gid,
                        "sport": first.get("sport"),
                        "home_team": first.get("home_team"),
                        "away_team": first.get("away_team"),
                        "market_type": market_type,
                        "type": "line_anomaly",
                        "sportsbook": e["sportsbook"],
                        "line": e["line"],
                        "z_score": round(z, 3),
                        "abs_z_score": round(abs(z), 3),
                        "mean_line": round(mu, 2),
                        "std_dev": round(sigma, 2),
                        "deviation": round(e["line"] - mu, 2),
                        "last_updated": e.get("last_updated"),
                        "context": (
                            f"Z-SCORE LINE ANOMALY: {e['sportsbook']} {label} "
                            f"at {e['line']} (z={round(z, 2)}, mean={round(mu, 1)}, "
                            f"σ={round(sigma, 2)})"
                        ),
                    })

    anomalies.sort(key=lambda a: a["abs_z_score"], reverse=True)

    odds_anomalies = [a for a in anomalies if a.get("type") != "line_anomaly"]
    line_anomalies = [a for a in anomalies if a.get("type") == "line_anomaly"]

    result = {
        "anomalies": anomalies,
        "count": len(anomalies),
        "odds_anomalies": len(odds_anomalies),
        "line_anomalies": len(line_anomalies),
        "z_threshold": z_threshold,
        "games_analyzed": len(target_games),
        "context": (
            f"Found {len(anomalies)} z-score anomalies (|z| > {z_threshold}) "
            f"across {len(target_games)} game(s): "
            f"{len(odds_anomalies)} odds anomalies, {len(line_anomalies)} line anomalies."
            + (f" Most extreme: {anomalies[0]['context']}" if anomalies else "")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


@mcp.tool()
def find_cross_market_arbitrage(filename: Optional[str] = None, min_edge_pct: float = 0.0) -> str:
    """Find arbitrage opportunities ACROSS different market types at different sportsbooks.

    Unlike standard arbitrage (which compares the same market across books), this
    looks for mispricings between market types — e.g., a moneyline at one book vs.
    a spread at another that covers an overlapping outcome at better combined value.

    Cross-market arb scenarios detected:
    1. ML vs Spread: Moneyline underdog at Book A + spread favorite at Book B
       (opposing sides, different market types — gap = "favorite wins by < spread")
    2. ML vs Spread (tight lines): When spread is small (<=3.5), ML and spread
       cover nearly identical outcomes — price discrepancies are exploitable
    3. Implied probability inconsistency: ML implies Team A wins 60% but spread
       odds at another book imply they cover 55% on a tight line — mispricing signal

    Args:
        filename: Data file to load. Optional.
        min_edge_pct: Minimum edge % to report (default 0 = show all). E.g., 1.0 = only edges >=1%.

    Returns cross-market arbitrage opportunities sorted by edge, including the
    gap/middle zone where neither bet wins.
    """
    cache_key = f"cross_market_arb:{min_edge_pct}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    games = _cache.load_by_game(filename)
    opportunities = []

    for game_id, records in games.items():
        first = records[0]
        game_info = {
            "game_id": game_id,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
        }

        # Collect best odds per (market_type, side) across all books
        # For spread, also track the line value
        best: dict[tuple, dict] = {}  # (market, side) -> {odds, book, line?}

        for r in records:
            book = r["sportsbook"]
            mkts = r.get("markets", {})

            # Moneyline
            ml = mkts.get("moneyline", {})
            if "home_odds" in ml:
                k = ("moneyline", "home")
                if k not in best or ml["home_odds"] > best[k]["odds"]:
                    best[k] = {"odds": ml["home_odds"], "book": book}
            if "away_odds" in ml:
                k = ("moneyline", "away")
                if k not in best or ml["away_odds"] > best[k]["odds"]:
                    best[k] = {"odds": ml["away_odds"], "book": book}

            # Spread
            sp = mkts.get("spread", {})
            if "home_odds" in sp and "home_line" in sp:
                k = ("spread", "home")
                if k not in best or sp["home_odds"] > best[k]["odds"]:
                    best[k] = {"odds": sp["home_odds"], "book": book, "line": sp["home_line"]}
            if "away_odds" in sp:
                a_line = sp.get("away_line", -sp.get("home_line", 0))
                k = ("spread", "away")
                if k not in best or sp["away_odds"] > best[k]["odds"]:
                    best[k] = {"odds": sp["away_odds"], "book": book, "line": a_line}

            # Total
            tot = mkts.get("total", {})
            if "over_odds" in tot and "line" in tot:
                k = ("total", "over")
                if k not in best or tot["over_odds"] > best[k]["odds"]:
                    best[k] = {"odds": tot["over_odds"], "book": book, "line": tot["line"]}
            if "under_odds" in tot and "line" in tot:
                k = ("total", "under")
                if k not in best or tot["under_odds"] > best[k]["odds"]:
                    best[k] = {"odds": tot["under_odds"], "book": book, "line": tot["line"]}

        # --- Cross-market comparisons ---

        # 1. ML Home vs Spread Away (opposing sides, different markets)
        #    ML Home = "home wins outright"
        #    Spread Away = "away covers +X" = "away wins OR loses by < X"
        #    Gap zone = "home wins by less than spread" (neither side fully wins)
        ml_home = best.get(("moneyline", "home"))
        sp_away = best.get(("spread", "away"))
        if ml_home and sp_away and ml_home["book"] != sp_away["book"]:
            prob_ml_h = implied_probability(ml_home["odds"])
            prob_sp_a = implied_probability(sp_away["odds"])
            combined = prob_ml_h + prob_sp_a
            if combined < 1.0:
                edge = round((1.0 - combined) * 100, 3)
                if edge >= min_edge_pct:
                    a_line = sp_away.get("line", "?")
                    abs_line = abs(a_line) if isinstance(a_line, (int, float)) else "?"
                    opportunities.append({
                        **game_info,
                        "type": "ml_home_vs_spread_away",
                        "leg_a": {
                            "market": "moneyline",
                            "side": "home",
                            "sportsbook": ml_home["book"],
                            "odds": ml_home["odds"],
                            "implied_prob": round(prob_ml_h, 6),
                            "covers": f"{first.get('home_team')} wins outright",
                        },
                        "leg_b": {
                            "market": "spread",
                            "side": "away",
                            "sportsbook": sp_away["book"],
                            "odds": sp_away["odds"],
                            "line": a_line,
                            "implied_prob": round(prob_sp_a, 6),
                            "covers": f"{first.get('away_team')} +{a_line} (wins or loses by < {abs_line})",
                        },
                        "combined_implied": round(combined, 6),
                        "edge_pct": edge,
                        "gap_zone": f"{first.get('home_team')} wins by exactly 1 to {int(abs_line) if isinstance(abs_line, (int, float)) else '?'} pts — neither bet wins",
                        "context": (
                            f"CROSS-MKT ARB: ML {first.get('home_team')} at {ml_home['book']} ({ml_home['odds']}) "
                            f"+ Spread {first.get('away_team')} +{a_line} at {sp_away['book']} ({sp_away['odds']}) "
                            f"= {edge}% edge (gap: home wins by <{int(abs_line) if isinstance(abs_line, (int, float)) else '?'})"
                        ),
                    })

        # 2. ML Away vs Spread Home (mirror of above)
        ml_away = best.get(("moneyline", "away"))
        sp_home = best.get(("spread", "home"))
        if ml_away and sp_home and ml_away["book"] != sp_home["book"]:
            prob_ml_a = implied_probability(ml_away["odds"])
            prob_sp_h = implied_probability(sp_home["odds"])
            combined = prob_ml_a + prob_sp_h
            if combined < 1.0:
                edge = round((1.0 - combined) * 100, 3)
                if edge >= min_edge_pct:
                    h_line = sp_home.get("line", "?")
                    abs_line = abs(h_line) if isinstance(h_line, (int, float)) else "?"
                    opportunities.append({
                        **game_info,
                        "type": "ml_away_vs_spread_home",
                        "leg_a": {
                            "market": "moneyline",
                            "side": "away",
                            "sportsbook": ml_away["book"],
                            "odds": ml_away["odds"],
                            "implied_prob": round(prob_ml_a, 6),
                            "covers": f"{first.get('away_team')} wins outright",
                        },
                        "leg_b": {
                            "market": "spread",
                            "side": "home",
                            "sportsbook": sp_home["book"],
                            "odds": sp_home["odds"],
                            "line": h_line,
                            "implied_prob": round(prob_sp_h, 6),
                            "covers": f"{first.get('home_team')} {h_line} (wins by > {abs_line})",
                        },
                        "combined_implied": round(combined, 6),
                        "edge_pct": edge,
                        "gap_zone": f"{first.get('home_team')} wins by 1 to {int(abs_line) if isinstance(abs_line, (int, float)) else '?'} pts — neither bet wins",
                        "context": (
                            f"CROSS-MKT ARB: ML {first.get('away_team')} at {ml_away['book']} ({ml_away['odds']}) "
                            f"+ Spread {first.get('home_team')} {h_line} at {sp_home['book']} ({sp_home['odds']}) "
                            f"= {edge}% edge (gap: home wins by <{int(abs_line) if isinstance(abs_line, (int, float)) else '?'})"
                        ),
                    })

        # 3. Tight-spread equivalence: when spread is small (<=3.5), ML and spread
        #    cover nearly the same outcome — compare implied probabilities across books
        #    to find cross-market mispricings
        sp_home_data = best.get(("spread", "home"))
        sp_away_data = best.get(("spread", "away"))

        for side_label, ml_entry, sp_entry in [
            ("home", ml_home, sp_home_data),
            ("away", ml_away, sp_away_data),
        ]:
            if not ml_entry or not sp_entry:
                continue
            spread_line = sp_entry.get("line")
            if spread_line is None or abs(spread_line) > 3.5:
                continue
            # Different books only — same-book isn't cross-market arb
            if ml_entry["book"] == sp_entry["book"]:
                continue

            ml_prob = implied_probability(ml_entry["odds"])
            sp_prob = implied_probability(sp_entry["odds"])
            prob_diff = abs(ml_prob - sp_prob) * 100
            cheaper_market = "spread" if sp_prob < ml_prob else "moneyline"
            cheaper_entry = sp_entry if sp_prob < ml_prob else ml_entry
            expensive_entry = ml_entry if sp_prob < ml_prob else sp_entry

            if prob_diff >= max(min_edge_pct, 1.0):  # At least 1% discrepancy
                team_name = first.get("home_team") if side_label == "home" else first.get("away_team")
                opportunities.append({
                    **game_info,
                    "type": "tight_spread_ml_mismatch",
                    "side": side_label,
                    "team": team_name,
                    "spread_line": spread_line,
                    "moneyline_leg": {
                        "sportsbook": ml_entry["book"],
                        "odds": ml_entry["odds"],
                        "implied_prob": round(ml_prob, 6),
                    },
                    "spread_leg": {
                        "sportsbook": sp_entry["book"],
                        "odds": sp_entry["odds"],
                        "line": spread_line,
                        "implied_prob": round(sp_prob, 6),
                    },
                    "prob_discrepancy_pct": round(prob_diff, 3),
                    "cheaper_market": cheaper_market,
                    "recommendation": (
                        f"Bet {team_name} via {cheaper_market} at {cheaper_entry['book']} "
                        f"({cheaper_entry['odds']}) — {round(prob_diff, 1)}% cheaper than "
                        f"{expensive_entry['book']}'s {'ML' if cheaper_market == 'spread' else 'spread'}"
                    ),
                    "context": (
                        f"TIGHT-SPREAD MISMATCH: {team_name} ML at {ml_entry['book']} "
                        f"({ml_entry['odds']}, {round(ml_prob*100,1)}%) vs spread {spread_line} "
                        f"at {sp_entry['book']} ({sp_entry['odds']}, {round(sp_prob*100,1)}%). "
                        f"Discrepancy: {round(prob_diff, 1)}% — bet the {cheaper_market}."
                    ),
                })

        # 4. Cross-market implied probability consistency check
        #    Compare what ML implies about win probability vs what spread odds imply
        #    across ALL books. Flag games where markets disagree significantly.
        ml_probs_home_list = []
        sp_probs_home_list = []
        for r in records:
            mkts = r.get("markets", {})
            ml_data = mkts.get("moneyline", {})
            sp_data = mkts.get("spread", {})
            if "home_odds" in ml_data:
                ml_probs_home_list.append(implied_probability(ml_data["home_odds"]))
            if "home_odds" in sp_data and "away_odds" in sp_data:
                nv = no_vig_probabilities(sp_data["home_odds"], sp_data["away_odds"])
                sp_probs_home_list.append(nv[0])

        if ml_probs_home_list and sp_probs_home_list:
            # Remove vig from ML too for fair comparison
            ml_fair_probs = []
            for r in records:
                ml_data = r.get("markets", {}).get("moneyline", {})
                if "home_odds" in ml_data and "away_odds" in ml_data:
                    nv = no_vig_probabilities(ml_data["home_odds"], ml_data["away_odds"])
                    ml_fair_probs.append(nv[0])
            if ml_fair_probs:
                avg_ml_fair = sum(ml_fair_probs) / len(ml_fair_probs)
                avg_sp_fair = sum(sp_probs_home_list) / len(sp_probs_home_list)
                cross_mkt_gap = abs(avg_ml_fair - avg_sp_fair) * 100
                if cross_mkt_gap >= max(min_edge_pct, 2.0):
                    favored_mkt = "moneyline" if avg_ml_fair > avg_sp_fair else "spread"
                    bullish_team = first.get("home_team") if avg_ml_fair > avg_sp_fair else first.get("away_team")
                    opportunities.append({
                        **game_info,
                        "type": "cross_market_probability_gap",
                        "ml_consensus_fair_prob_home": round(avg_ml_fair, 6),
                        "spread_consensus_fair_prob_home": round(avg_sp_fair, 6),
                        "gap_pct": round(cross_mkt_gap, 3),
                        "interpretation": (
                            f"Moneyline market implies {first.get('home_team')} has a "
                            f"{round(avg_ml_fair * 100, 1)}% chance, but spread market implies "
                            f"{round(avg_sp_fair * 100, 1)}% — a {round(cross_mkt_gap, 1)}% gap. "
                            f"The {favored_mkt} market is more bullish on {bullish_team}."
                        ),
                        "actionable": (
                            "Look for value betting against the more expensive market. "
                            "If ML is higher, the spread may be underpricing the favorite."
                        ),
                        "context": (
                            f"CROSS-MKT GAP: {first.get('home_team')} ML fair "
                            f"{round(avg_ml_fair*100,1)}% vs Spread fair "
                            f"{round(avg_sp_fair*100,1)}% — "
                            f"{round(cross_mkt_gap,1)}% inconsistency between markets."
                        ),
                    })

    # Sort by edge/discrepancy
    def _sort_key(opp):
        return opp.get("edge_pct", 0) or opp.get("prob_discrepancy_pct", 0) or opp.get("gap_pct", 0)
    opportunities.sort(key=_sort_key, reverse=True)

    arb_count = sum(1 for o in opportunities if o["type"] in ("ml_home_vs_spread_away", "ml_away_vs_spread_home"))
    mismatch_count = sum(1 for o in opportunities if o["type"] == "tight_spread_ml_mismatch")
    gap_count = sum(1 for o in opportunities if o["type"] == "cross_market_probability_gap")

    result = {
        "cross_market_opportunities": opportunities,
        "count": len(opportunities),
        "breakdown": {
            "ml_vs_spread_arbs": arb_count,
            "tight_spread_mismatches": mismatch_count,
            "cross_market_prob_gaps": gap_count,
        },
        "context": (
            f"Found {len(opportunities)} cross-market opportunities: "
            f"{arb_count} ML-vs-spread arbs, {mismatch_count} tight-spread mismatches, "
            f"{gap_count} cross-market probability gaps."
            + (f" Best: {opportunities[0]['context']}" if opportunities else " Markets are well-aligned across types.")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


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
            "synthetic_hold_free_market": "A constructed 'perfect book' built by combining the best available odds across all sportsbooks for every side of every market. Shows the edge a sharp bettor gets by always shopping for the best line. A negative synthetic hold means the bettor has an inherent advantage.",
            "cross_market_arbitrage": "Arbitrage across different market types (e.g., moneyline at one book vs. spread at another). Unlike standard arbs (same market, different books), cross-market arbs exploit mispricings between how different markets price the same game. Often have a 'gap zone' where neither bet wins.",
            "hold": "Hold percentage — the sportsbook's overall margin across all markets. Calculated as (sum of implied probabilities - 1) x 100. Lower hold = fairer book. Use get_hold_percentage for per-sportsbook breakdowns by market type.",
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
