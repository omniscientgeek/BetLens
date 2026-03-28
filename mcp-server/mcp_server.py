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
from math import sqrt, log, exp, lgamma, inf
from statistics import pstdev, mean
from typing import Optional

# Allow imports from sibling webservice/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))

from mcp.server.fastmcp import FastMCP
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, shin_probabilities, fair_odds_to_american, kelly_criterion, bayesian_update

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


def _compute_market_consistency(enriched_markets: dict) -> Optional[dict]:
    """Score cross-market consistency for a single sportsbook/game record.

    Each market type implies a win probability for the home team.  When those
    probabilities diverge within the *same* sportsbook the book has an internal
    inconsistency that a bettor can exploit.

    Comparisons performed (all using no-vig / fair probabilities):
      1. Spread vs Moneyline  — both directly imply a home-win probability
      2. Spread vs Total      — uses spread + total to derive implied scores;
         checks whether the score-gap direction agrees with the spread line
      3. Moneyline vs Total   — same implied-score check against ML probability

    Returns ``None`` when fewer than two markets are available (nothing to
    compare).
    """

    spread = enriched_markets.get("spread")
    ml = enriched_markets.get("moneyline")
    total = enriched_markets.get("total")

    # We need at least two markets to compare
    available = sum(1 for m in (spread, ml, total) if m is not None)
    if available < 2:
        return None

    pairs: dict[str, dict] = {}
    diffs: list[float] = []
    exploitable_details: list[str] = []

    # ---- 1. Spread vs Moneyline (primary comparison) ----------------------
    if spread and ml:
        spread_fair = no_vig_probabilities(spread["home_odds"], spread["away_odds"])
        ml_fair = no_vig_probabilities(ml["home_odds"], ml["away_odds"])

        sp_home = spread_fair["fair_a"]
        ml_home = ml_fair["fair_a"]
        diff = abs(sp_home - ml_home)

        pairs["spread_vs_moneyline"] = {
            "spread_implied_home_prob": round(sp_home, 4),
            "moneyline_implied_home_prob": round(ml_home, 4),
            "diff": round(diff, 4),
            "diff_pct": f"{round(diff * 100, 2)}%",
            "higher_market": "spread" if sp_home > ml_home else "moneyline",
        }
        diffs.append(diff)

        if diff > 0.05:
            cheaper = "moneyline" if sp_home > ml_home else "spread"
            exploitable_details.append(
                f"Spread implies {round(sp_home * 100, 1)}% home win but ML implies "
                f"{round(ml_home * 100, 1)}% — {round(diff * 100, 1)}pp gap. "
                f"Home side looks under-priced on the {cheaper} market."
            )

    # ---- 2. Spread vs Total (implied-score consistency) -------------------
    if spread and total:
        spread_line = spread.get("home_line")  # e.g. -5.5 means home favoured by 5.5
        total_line = total.get("line")          # e.g. 220.5

        if spread_line is not None and total_line is not None:
            # Implied scores: home = (total - spread_line) / 2
            #                 away = (total + spread_line) / 2
            implied_home_score = (total_line - spread_line) / 2
            implied_away_score = (total_line + spread_line) / 2
            score_gap = implied_home_score - implied_away_score  # positive = home favoured

            # The spread line already tells us the expected gap; check that
            # the total doesn't push implied scores into contradiction.
            # A contradiction arises when the spread says home is favoured but
            # implied scores say away scores more (or vice-versa).
            spread_favours_home = spread_line < 0
            scores_favour_home = score_gap > 0

            direction_match = spread_favours_home == scores_favour_home or score_gap == 0
            # Use score-gap magnitude vs spread magnitude as a soft diff
            expected_gap = abs(spread_line)
            actual_gap = abs(score_gap)
            gap_diff = abs(actual_gap - expected_gap) / max(expected_gap, 1)

            pairs["spread_vs_total"] = {
                "spread_line": spread_line,
                "total_line": total_line,
                "implied_home_score": round(implied_home_score, 2),
                "implied_away_score": round(implied_away_score, 2),
                "score_gap": round(score_gap, 2),
                "direction_consistent": direction_match,
                "gap_deviation_pct": f"{round(gap_diff * 100, 1)}%",
            }
            # Normalise to a 0-1 scale comparable to probability diffs
            normalised_diff = min(gap_diff * 0.1, 0.20)  # cap contribution
            diffs.append(normalised_diff)

            if not direction_match:
                exploitable_details.append(
                    f"Spread ({spread_line}) favours {'home' if spread_favours_home else 'away'} "
                    f"but total-implied scores ({round(implied_home_score, 1)}-"
                    f"{round(implied_away_score, 1)}) favour the other side."
                )

    # ---- 3. Moneyline vs Total (implied-score check) ----------------------
    if ml and total:
        ml_fair = no_vig_probabilities(ml["home_odds"], ml["away_odds"])
        ml_home_prob = ml_fair["fair_a"]
        total_line = total.get("line")

        if total_line is not None and spread:
            # Re-use implied scores computed above when available
            spread_line = spread.get("home_line", 0)
            implied_home_score = (total_line - spread_line) / 2
            implied_away_score = (total_line + spread_line) / 2
            ml_favours_home = ml_home_prob > 0.5
            scores_favour_home = implied_home_score > implied_away_score

            direction_match = ml_favours_home == scores_favour_home or implied_home_score == implied_away_score

            pairs["moneyline_vs_total"] = {
                "moneyline_home_prob": round(ml_home_prob, 4),
                "implied_home_score": round(implied_home_score, 2),
                "implied_away_score": round(implied_away_score, 2),
                "direction_consistent": direction_match,
            }

            if not direction_match:
                normalised_diff = 0.06  # fixed penalty for directional mismatch
                diffs.append(normalised_diff)
                exploitable_details.append(
                    f"ML implies home wins {round(ml_home_prob * 100, 1)}% of the time "
                    f"but implied scores favour {'away' if ml_favours_home else 'home'}."
                )

    if not diffs:
        return None

    # ---- Overall consistency score (0-100, higher = more consistent) ------
    avg_diff = sum(diffs) / len(diffs)
    max_diff = max(diffs)
    # Scale: 0% avg diff → 100 score, ≥10% avg diff → 0 score (linear)
    consistency_score = round(max(0.0, (1 - avg_diff / 0.10)) * 100, 1)

    exploitable = max_diff > 0.05  # >5 pp gap in any pair

    result: dict = {
        "consistency_score": consistency_score,
        "avg_diff": round(avg_diff, 4),
        "max_diff": round(max_diff, 4),
        "pairs": pairs,
        "exploitable": exploitable,
    }
    if exploitable_details:
        result["exploit_details"] = exploitable_details

    return result


def _enrich_record(record: dict) -> dict:
    """Enrich a single odds record with implied probabilities, vig, and fair odds."""
    markets = record.get("markets", {})
    enriched_markets = {}

    for market_name, market in markets.items():
        if market_name == "spread":
            vig_info = calculate_vig(market["home_odds"], market["away_odds"])
            fair = no_vig_probabilities(market["home_odds"], market["away_odds"])
            shin = shin_probabilities(market["home_odds"], market["away_odds"])
            enriched_markets["spread"] = {
                **market,
                "home_implied_prob": vig_info["implied_a"],
                "away_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
                "home_shin_prob": shin["shin_a"],
                "away_shin_prob": shin["shin_b"],
                "home_shin_fair_odds": fair_odds_to_american(shin["shin_a"]),
                "away_shin_fair_odds": fair_odds_to_american(shin["shin_b"]),
                "shin_z": shin["z"],
                "vig_on_home": shin["vig_on_a"],
                "vig_on_away": shin["vig_on_b"],
            }
        elif market_name == "moneyline":
            vig_info = calculate_vig(market["home_odds"], market["away_odds"])
            fair = no_vig_probabilities(market["home_odds"], market["away_odds"])
            shin = shin_probabilities(market["home_odds"], market["away_odds"])
            enriched_markets["moneyline"] = {
                **market,
                "home_implied_prob": vig_info["implied_a"],
                "away_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
                "home_shin_prob": shin["shin_a"],
                "away_shin_prob": shin["shin_b"],
                "home_shin_fair_odds": fair_odds_to_american(shin["shin_a"]),
                "away_shin_fair_odds": fair_odds_to_american(shin["shin_b"]),
                "shin_z": shin["z"],
                "vig_on_home": shin["vig_on_a"],
                "vig_on_away": shin["vig_on_b"],
            }
        elif market_name == "total":
            vig_info = calculate_vig(market["over_odds"], market["under_odds"])
            fair = no_vig_probabilities(market["over_odds"], market["under_odds"])
            shin = shin_probabilities(market["over_odds"], market["under_odds"])
            enriched_markets["total"] = {
                **market,
                "over_implied_prob": vig_info["implied_a"],
                "under_implied_prob": vig_info["implied_b"],
                "vig": vig_info["vig"],
                "vig_pct": vig_info["vig_pct"],
                "over_fair_odds": fair_odds_to_american(fair["fair_a"]),
                "under_fair_odds": fair_odds_to_american(fair["fair_b"]),
                "over_shin_prob": shin["shin_a"],
                "under_shin_prob": shin["shin_b"],
                "over_shin_fair_odds": fair_odds_to_american(shin["shin_a"]),
                "under_shin_fair_odds": fair_odds_to_american(shin["shin_b"]),
                "shin_z": shin["z"],
                "vig_on_over": shin["vig_on_a"],
                "vig_on_under": shin["vig_on_b"],
            }

    # --- 8. Market Consistency Scoring (Spread vs ML vs Total) -------------
    # Compare the win probability each market implies for the home team.
    # Large gaps within the same book signal an internal inconsistency that
    # bettors can exploit (e.g. spread implies 68% but ML implies 62%).
    market_consistency = _compute_market_consistency(enriched_markets)

    out = {**{k: v for k, v in record.items() if k != "markets"}, "markets": enriched_markets}
    if market_consistency is not None:
        out["market_consistency"] = market_consistency
    return out


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


# ── Basic Arithmetic Tools ──────────────────────────────────────────────


@mcp.tool()
def arithmetic_add(a: float, b: float) -> str:
    """Add two numbers together.

    Use this for any addition calculation — bankroll totals, combined payouts,
    summing edges, accumulating profits, etc.

    Args:
        a: First number
        b: Second number
    """
    result = a + b
    return json.dumps({
        "operation": "add",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} + {b} = {result}",
    }, indent=2)


@mcp.tool()
def arithmetic_subtract(a: float, b: float) -> str:
    """Subtract b from a (a - b).

    Use this for differences — edge calculations, line movement deltas,
    profit/loss, comparing odds across books, etc.

    Args:
        a: Number to subtract from
        b: Number to subtract
    """
    result = a - b
    return json.dumps({
        "operation": "subtract",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} - {b} = {result}",
    }, indent=2)


@mcp.tool()
def arithmetic_multiply(a: float, b: float) -> str:
    """Multiply two numbers (a × b).

    Use this for scaling — bet sizing from percentages, payout calculations,
    bankroll × kelly fraction, unit conversion, etc.

    Args:
        a: First number
        b: Second number
    """
    result = a * b
    return json.dumps({
        "operation": "multiply",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} × {b} = {result}",
    }, indent=2)


@mcp.tool()
def arithmetic_divide(a: float, b: float) -> str:
    """Divide a by b (a ÷ b).

    Use this for ratios — vig as percentage, odds conversion, ROI,
    per-unit profit, average edge, etc.

    Args:
        a: Numerator (dividend)
        b: Denominator (divisor) — must not be zero
    """
    if b == 0:
        return json.dumps({
            "operation": "divide",
            "a": a,
            "b": b,
            "result": None,
            "error": "Division by zero is undefined",
        }, indent=2)
    result = a / b
    return json.dumps({
        "operation": "divide",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} ÷ {b} = {result}",
    }, indent=2)


@mcp.tool()
def arithmetic_modulo(a: float, b: float) -> str:
    """Calculate the remainder of a ÷ b (a % b).

    Use this for remainder / modulus operations — checking divisibility,
    cyclic patterns, rounding logic, etc.

    Args:
        a: Dividend
        b: Divisor — must not be zero
    """
    if b == 0:
        return json.dumps({
            "operation": "modulo",
            "a": a,
            "b": b,
            "result": None,
            "error": "Modulo by zero is undefined",
        }, indent=2)
    result = a % b
    return json.dumps({
        "operation": "modulo",
        "a": a,
        "b": b,
        "result": result,
        "expression": f"{a} % {b} = {result}",
    }, indent=2)


@mcp.tool()
def arithmetic_evaluate(expression: str) -> str:
    """Evaluate a multi-step arithmetic expression safely.

    Supports: +, -, *, /, %, parentheses, and decimal numbers.
    Use this for compound calculations like "(bankroll * kelly_pct) - existing_exposure"
    or "((odds_a - odds_b) / 2) + margin".

    Args:
        expression: Arithmetic expression string, e.g. "(100 * 0.25) + 50"
    """
    import re
    # Whitelist: only digits, operators, parens, decimal points, spaces
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*/%\.\(\)]+$', sanitized):
        return json.dumps({
            "operation": "evaluate",
            "expression": expression,
            "result": None,
            "error": "Invalid expression — only numbers and +, -, *, /, %, () are allowed",
        }, indent=2)

    try:
        # Safe eval with no builtins or namespace access
        result = eval(sanitized, {"__builtins__": {}}, {})
        return json.dumps({
            "operation": "evaluate",
            "expression": expression,
            "result": result,
        }, indent=2)
    except ZeroDivisionError:
        return json.dumps({
            "operation": "evaluate",
            "expression": expression,
            "result": None,
            "error": "Division by zero in expression",
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "operation": "evaluate",
            "expression": expression,
            "result": None,
            "error": f"Could not evaluate expression: {str(e)}",
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
def simulate_bankroll_kelly(
    filename: Optional[str] = None,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.5,
    min_ev_pct: float = 0.0,
    max_bet_pct: float = 10.0,
    num_simulations: int = 1000,
) -> str:
    """Run a full bankroll management simulation using fractional Kelly across all games.

    For every +EV opportunity found across all games, this tool calculates
    Kelly-optimal bet sizes (scaled by kelly_fraction), then runs Monte Carlo
    simulations to project bankroll outcomes over the full slate.

    The Kelly formula: f* = (bp - q) / b
      where b = decimal odds - 1, p = true probability, q = 1 - p

    Fractional Kelly (e.g. half-Kelly with kelly_fraction=0.5) is applied to
    reduce variance and drawdown risk while retaining most of the edge.

    The simulation:
    1. Identifies all +EV bets across every game/market/sportsbook.
    2. For each bet, calculates the fractional Kelly wager (capped at max_bet_pct
       of current bankroll).
    3. Runs num_simulations Monte Carlo trials where each bet is resolved
       randomly according to its fair probability.
    4. Reports projected bankroll growth, risk of ruin, drawdown stats, and
       per-bet breakdowns.

    Args:
        filename: Data file to load. Optional.
        bankroll: Starting bankroll in dollars. Default $1,000.
        kelly_fraction: Fraction of full Kelly to bet (0.5 = half-Kelly recommended).
                        Common values: 1.0 (full, aggressive), 0.5 (half, balanced),
                        0.25 (quarter, conservative).
        min_ev_pct: Minimum EV edge (%) to include a bet. Default 0 (all +EV bets).
        max_bet_pct: Maximum single bet as % of current bankroll (risk cap). Default 10%.
        num_simulations: Number of Monte Carlo trials. Default 1,000.

    Returns a comprehensive bankroll simulation with per-bet sizing, projected
    outcomes, drawdown analysis, and risk metrics.
    """
    cache_key = f"bankroll_sim:{bankroll}:{kelly_fraction}:{min_ev_pct}:{max_bet_pct}:{num_simulations}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    pinnacle_probs = _get_pinnacle_fair_probs(games)

    # ── Step 1: Collect all +EV opportunities with Kelly sizing ──────────
    bet_slate: list[dict] = []

    for gid, records in games.items():
        first = records[0]
        home_team = first.get("home_team", "Unknown")
        away_team = first.get("away_team", "Unknown")
        sport = first.get("sport", "Unknown")

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [("home", "home_odds", "side_a_prob"),
                         ("away", "away_odds", "side_b_prob")]
            else:
                sides = [("over", "over_odds", "side_a_prob"),
                         ("under", "under_odds", "side_b_prob")]

            for side, odds_key, prob_key in sides:
                game_fair = pinnacle_probs.get(gid, {}).get(market_type)
                if not game_fair:
                    continue
                fair_prob = game_fair[prob_key]
                prob_source = game_fair["source"]

                for r in records:
                    if r.get("sportsbook", "").lower() == SHARP_BOOK.lower():
                        continue

                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key not in market:
                        continue

                    book_odds = market[odds_key]
                    book_prob = implied_probability(book_odds)
                    ev_edge = fair_prob - book_prob

                    if ev_edge <= 0 or (ev_edge * 100) < min_ev_pct:
                        continue

                    # Decimal net payout (b)
                    if book_odds < 0:
                        b = 100 / abs(book_odds)
                    elif book_odds > 0:
                        b = book_odds / 100
                    else:
                        b = 1.0

                    p = fair_prob
                    q = 1 - p

                    # Full Kelly: f* = (bp - q) / b
                    full_kelly = (b * p - q) / b if b != 0 else 0.0
                    full_kelly = max(0.0, min(full_kelly, 1.0))

                    if full_kelly <= 0:
                        continue

                    frac_kelly = full_kelly * kelly_fraction
                    # Cap at max_bet_pct of bankroll
                    frac_kelly = min(frac_kelly, max_bet_pct / 100.0)

                    decimal_odds = b + 1  # total payout per $1

                    bet_slate.append({
                        "game_id": gid,
                        "sport": sport,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market_type": market_type,
                        "side": side,
                        "sportsbook": r["sportsbook"],
                        "odds": book_odds,
                        "decimal_odds": round(decimal_odds, 4),
                        "book_implied_prob": round(book_prob, 6),
                        "fair_prob": round(fair_prob, 6),
                        "fair_prob_source": prob_source,
                        "ev_edge_pct": round(ev_edge * 100, 3),
                        "full_kelly_pct": round(full_kelly * 100, 3),
                        "fractional_kelly_pct": round(frac_kelly * 100, 3),
                        "wager_on_starting_bankroll": round(frac_kelly * bankroll, 2),
                        "net_payout_per_dollar": round(b, 4),
                        "expected_value_per_dollar": round(p * b - q, 4),
                        "last_updated": r.get("last_updated"),
                    })

    # Sort by EV edge descending (best opportunities first)
    bet_slate.sort(key=lambda x: x["ev_edge_pct"], reverse=True)

    if not bet_slate:
        result = {
            "error": "No +EV opportunities found for simulation.",
            "settings": {
                "bankroll": f"${bankroll}",
                "kelly_fraction": kelly_fraction,
                "min_ev_pct": min_ev_pct,
            },
        }
        _cache.set_analysis(filename, cache_key, result)
        return json.dumps(result, indent=2)

    # ── Step 2: Deterministic (expected) bankroll projection ─────────────
    # Walk through bets sequentially, sizing each off current bankroll
    expected_bankroll = bankroll
    bet_plan: list[dict] = []

    for bet in bet_slate:
        wager = round(bet["fractional_kelly_pct"] / 100.0 * expected_bankroll, 2)
        wager = min(wager, expected_bankroll)  # can't bet more than you have

        ev_per_dollar = bet["expected_value_per_dollar"]
        expected_profit = round(wager * ev_per_dollar, 2)
        expected_bankroll_after = round(expected_bankroll + expected_profit, 2)

        bet_plan.append({
            "order": len(bet_plan) + 1,
            "game": f"{bet['away_team']} @ {bet['home_team']}",
            "bet": f"{bet['side']} {bet['market_type']} at {bet['sportsbook']}",
            "odds": bet["odds"],
            "fair_prob": f"{round(bet['fair_prob'] * 100, 1)}%",
            "ev_edge": f"{bet['ev_edge_pct']}%",
            "kelly_full": f"{bet['full_kelly_pct']}%",
            "kelly_used": f"{bet['fractional_kelly_pct']}%",
            "wager": f"${wager}",
            "wager_raw": wager,
            "expected_profit": f"${expected_profit}",
            "bankroll_before": f"${round(expected_bankroll, 2)}",
            "bankroll_after_ev": f"${expected_bankroll_after}",
        })

        expected_bankroll = expected_bankroll_after

    total_wagered = round(sum(b["wager_raw"] for b in bet_plan), 2)
    expected_growth = round(expected_bankroll - bankroll, 2)
    expected_roi = round((expected_bankroll / bankroll - 1) * 100, 3)

    # ── Step 3: Monte Carlo simulation ───────────────────────────────────
    final_bankrolls: list[float] = []
    max_drawdowns: list[float] = []
    bust_count = 0  # bankroll drops below 1% of starting

    for _ in range(num_simulations):
        sim_bankroll = bankroll
        peak = bankroll
        worst_drawdown = 0.0

        for bet in bet_slate:
            if sim_bankroll <= 0:
                break

            wager = bet["fractional_kelly_pct"] / 100.0 * sim_bankroll
            wager = min(wager, sim_bankroll)

            # Resolve bet randomly using fair probability
            if random.random() < bet["fair_prob"]:
                # Win: profit = wager * net_payout
                sim_bankroll += wager * bet["net_payout_per_dollar"]
            else:
                # Loss: lose the wager
                sim_bankroll -= wager

            # Track peak and drawdown
            if sim_bankroll > peak:
                peak = sim_bankroll
            dd = (peak - sim_bankroll) / peak if peak > 0 else 0
            if dd > worst_drawdown:
                worst_drawdown = dd

        final_bankrolls.append(round(sim_bankroll, 2))
        max_drawdowns.append(round(worst_drawdown * 100, 2))
        if sim_bankroll < bankroll * 0.01:
            bust_count += 1

    final_bankrolls.sort()
    max_drawdowns.sort()
    n = len(final_bankrolls)

    avg_final = round(mean(final_bankrolls), 2)
    median_final = final_bankrolls[n // 2]
    p5 = final_bankrolls[int(n * 0.05)]
    p25 = final_bankrolls[int(n * 0.25)]
    p75 = final_bankrolls[int(n * 0.75)]
    p95 = final_bankrolls[int(n * 0.95)]
    worst_case = final_bankrolls[0]
    best_case = final_bankrolls[-1]

    avg_drawdown = round(mean(max_drawdowns), 2)
    median_drawdown = max_drawdowns[len(max_drawdowns) // 2]
    worst_drawdown_sim = max_drawdowns[-1]

    profitable_sims = sum(1 for b in final_bankrolls if b > bankroll)
    win_rate = round(profitable_sims / n * 100, 1)

    # ── Step 4: Per-game summary ─────────────────────────────────────────
    game_summaries: dict[str, dict] = {}
    for bet in bet_slate:
        gid = bet["game_id"]
        if gid not in game_summaries:
            game_summaries[gid] = {
                "game": f"{bet['away_team']} @ {bet['home_team']}",
                "sport": bet["sport"],
                "bet_count": 0,
                "total_kelly_exposure_pct": 0.0,
                "best_ev_edge_pct": 0.0,
                "bets": [],
            }
        gs = game_summaries[gid]
        gs["bet_count"] += 1
        gs["total_kelly_exposure_pct"] += bet["fractional_kelly_pct"]
        gs["best_ev_edge_pct"] = max(gs["best_ev_edge_pct"], bet["ev_edge_pct"])
        gs["bets"].append(
            f"{bet['side']} {bet['market_type']} at {bet['sportsbook']} "
            f"({bet['odds']}, edge {bet['ev_edge_pct']}%)"
        )

    for gs in game_summaries.values():
        gs["total_kelly_exposure_pct"] = round(gs["total_kelly_exposure_pct"], 3)

    game_list = sorted(
        game_summaries.values(),
        key=lambda g: g["best_ev_edge_pct"],
        reverse=True,
    )

    # ── Step 5: Assemble result ──────────────────────────────────────────
    result = {
        "simulation_settings": {
            "starting_bankroll": f"${bankroll}",
            "kelly_fraction": kelly_fraction,
            "kelly_label": {
                1.0: "Full Kelly",
                0.5: "Half Kelly",
                0.25: "Quarter Kelly",
            }.get(kelly_fraction, f"{kelly_fraction}x Kelly"),
            "min_ev_pct": min_ev_pct,
            "max_single_bet_pct": f"{max_bet_pct}%",
            "num_simulations": num_simulations,
            "fair_prob_source": "Pinnacle no-vig (preferred) with consensus fallback",
        },
        "bet_slate": {
            "total_bets": len(bet_slate),
            "unique_games": len(game_summaries),
            "total_wagered_on_starting_bankroll": f"${total_wagered}",
            "bankroll_pct_deployed": f"{round(total_wagered / bankroll * 100, 1)}%",
        },
        "expected_outcome": {
            "expected_final_bankroll": f"${round(expected_bankroll, 2)}",
            "expected_profit": f"${expected_growth}",
            "expected_roi": f"{expected_roi}%",
            "note": "Deterministic projection using EV of each bet sequentially.",
        },
        "monte_carlo_results": {
            "simulations_run": num_simulations,
            "average_final_bankroll": f"${avg_final}",
            "median_final_bankroll": f"${median_final}",
            "percentiles": {
                "p5_worst_realistic": f"${p5}",
                "p25": f"${p25}",
                "p50_median": f"${median_final}",
                "p75": f"${p75}",
                "p95_best_realistic": f"${p95}",
            },
            "worst_case": f"${worst_case}",
            "best_case": f"${best_case}",
            "profitable_simulations": f"{win_rate}%",
            "risk_of_ruin": f"{round(bust_count / n * 100, 2)}%",
            "ruin_note": "Ruin = bankroll drops below 1% of starting value.",
        },
        "drawdown_analysis": {
            "avg_max_drawdown": f"{avg_drawdown}%",
            "median_max_drawdown": f"{median_drawdown}%",
            "worst_max_drawdown": f"{worst_drawdown_sim}%",
            "drawdown_note": "Max drawdown = largest peak-to-trough decline during the simulation.",
        },
        "per_game_breakdown": game_list,
        "bet_plan": bet_plan,
        "kelly_comparison": {
            "note": "Comparing Kelly fractions. Higher fractions grow faster but with more variance and drawdown risk.",
            "fractions": [],
        },
        "context": (
            f"Bankroll simulation: {len(bet_slate)} +EV bets across "
            f"{len(game_summaries)} games. "
            f"Using {kelly_fraction}x Kelly on ${bankroll} bankroll. "
            f"Expected profit: ${expected_growth} ({expected_roi}% ROI). "
            f"Monte Carlo ({num_simulations} sims): median ${median_final}, "
            f"win rate {win_rate}%, risk of ruin {round(bust_count / n * 100, 2)}%, "
            f"avg max drawdown {avg_drawdown}%."
        ),
    }

    # ── Quick comparison of different Kelly fractions ─────────────────────
    for frac, label in [(0.25, "Quarter Kelly"), (0.5, "Half Kelly"), (1.0, "Full Kelly")]:
        sim_results: list[float] = []
        for _ in range(min(500, num_simulations)):
            sim_b = bankroll
            for bet in bet_slate:
                if sim_b <= 0:
                    break
                fk = (bet["full_kelly_pct"] / 100.0) * frac
                fk = min(fk, max_bet_pct / 100.0)
                w = fk * sim_b
                w = min(w, sim_b)
                if random.random() < bet["fair_prob"]:
                    sim_b += w * bet["net_payout_per_dollar"]
                else:
                    sim_b -= w
            sim_results.append(round(sim_b, 2))
        sim_results.sort()
        sn = len(sim_results)
        result["kelly_comparison"]["fractions"].append({
            "fraction": frac,
            "label": label,
            "avg_final": f"${round(mean(sim_results), 2)}",
            "median_final": f"${sim_results[sn // 2]}",
            "p5": f"${sim_results[int(sn * 0.05)]}",
            "p95": f"${sim_results[int(sn * 0.95)]}",
            "profitable_pct": f"{round(sum(1 for x in sim_results if x > bankroll) / sn * 100, 1)}%",
        })

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
    """Estimate implied final scores for each team by combining spread and total.

    Formula:
        Home implied score = (Total + HomeSpread) / 2
        Away implied score = (Total - HomeSpread) / 2

    e.g., spread -5.5 + total 220 → Home ~107.25, Away ~112.75
    (negative home spread means home is favored)

    Includes a per-book implied score matrix that computes each sportsbook's
    individual implied scores from their own spread + total lines, then
    compares across all books to highlight where scores diverge most.
    Divergence flags games where books disagree on expected final scores.

    Args:
        game_id: Filter to a single game. Optional — returns all games if omitted.
        filename: Data file to load. Optional.

    Returns implied scores per game sorted by largest margin of victory,
    with book_matrix and divergence analysis per game.
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

        # --- Per-book implied score matrix ---
        book_scores = []
        for r in records:
            spread_mkt = r.get("markets", {}).get("spread", {})
            total_mkt = r.get("markets", {}).get("total", {})
            book_home_line = spread_mkt.get("home_line")
            book_total = total_mkt.get("line")
            if book_home_line is None or book_total is None:
                continue
            book_home_score = round((book_total + book_home_line) / 2, 2)
            book_away_score = round((book_total - book_home_line) / 2, 2)
            book_scores.append({
                "sportsbook": r["sportsbook"],
                "spread": book_home_line,
                "total": book_total,
                "home_implied": book_home_score,
                "away_implied": book_away_score,
                "home_diff_from_consensus": round(book_home_score - home_implied, 2),
                "away_diff_from_consensus": round(book_away_score - away_implied, 2),
            })

        # Divergence stats across books
        if len(book_scores) >= 2:
            home_scores_list = [b["home_implied"] for b in book_scores]
            away_scores_list = [b["away_implied"] for b in book_scores]
            home_max = max(home_scores_list)
            home_min = min(home_scores_list)
            away_max = max(away_scores_list)
            away_min = min(away_scores_list)
            home_range = round(home_max - home_min, 2)
            away_range = round(away_max - away_min, 2)
            max_divergence = max(home_range, away_range)
            # Find which books are most divergent
            most_bullish_home = max(book_scores, key=lambda b: b["home_implied"])
            most_bearish_home = min(book_scores, key=lambda b: b["home_implied"])
            divergence_info = {
                "home_score_range": home_range,
                "away_score_range": away_range,
                "home_high": {"sportsbook": most_bullish_home["sportsbook"], "score": home_max},
                "home_low": {"sportsbook": most_bearish_home["sportsbook"], "score": home_min},
                "max_divergence": max_divergence,
            }
        else:
            divergence_info = None
            max_divergence = 0

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
                "book_matrix": book_scores,
                "divergence": divergence_info,
                "context": (
                    f"{home_team} {home_implied} - {away_team} {away_implied} "
                    f"(spread {avg_home_line}, total {avg_total}). "
                    f"{favorite} by {margin}."
                    + (
                        f" Max book divergence: {max_divergence} pts "
                        f"({most_bullish_home['sportsbook']} highest at {home_max}, "
                        f"{most_bearish_home['sportsbook']} lowest at {home_min} for {home_team})."
                        if divergence_info
                        else ""
                    )
                ),
            }
        )

    results.sort(key=lambda r: r["margin_of_victory"], reverse=True)

    # Find game with most divergent implied scores across books
    most_divergent = max(results, key=lambda r: (r.get("divergence") or {}).get("max_divergence", 0)) if results else None

    result = {
        "implied_scores": results,
        "count": len(results),
        "closest_game": results[-1] if results else None,
        "biggest_blowout": results[0] if results else None,
        "most_divergent_game": {
            "game": f"{most_divergent['home_team']} vs {most_divergent['away_team']}",
            "max_divergence": most_divergent.get("divergence", {}).get("max_divergence", 0) if most_divergent else 0,
            "divergence": most_divergent.get("divergence"),
        } if most_divergent and most_divergent.get("divergence") else None,
        "context": (
            f"Implied final scores for {len(results)} games based on consensus "
            f"spread + total, with per-book score matrix showing divergence. "
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
            + (
                f" Most divergent: {most_divergent['home_team']} vs "
                f"{most_divergent['away_team']} "
                f"({most_divergent.get('divergence', {}).get('max_divergence', 0)} pts spread across books)."
                if most_divergent and most_divergent.get("divergence")
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
# SPORTSBOOK CORRELATION NETWORK — Pairwise Pearson Correlation Matrix
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_sportsbook_correlation_network(
    market_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Build a Pearson correlation matrix showing how closely each pair of sportsbooks tracks each other.

    For every pair of sportsbooks, computes the Pearson correlation of their
    implied-probability vectors across all shared (game, market, side) data points.
    High correlation (>0.95) suggests a shared odds feed; low correlation indicates
    independent pricing and potential value-source books.

    The output includes:
    - Full NxN correlation matrix
    - Ranked list of most- and least-correlated pairs
    - Network-style "edges" with strength labels (shared feed / closely aligned /
      moderately aligned / independently priced)
    - Per-book summary: average correlation with all other books, identifying
      which books are the most "connected" vs. the most "independent"

    Args:
        market_type: Optional filter — "spread", "moneyline", or "total".
                     If omitted, all market types are combined.
        filename: Data file to load.  Defaults to the first available file.

    Returns JSON with correlation_matrix, edges, book_summaries, and insights.
    """
    cache_key = f"sportsbook_corr_network:{market_type or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data available"})

    by_game = _cache.load_by_game(filename)

    # ------------------------------------------------------------------
    # Step 1: Build per-sportsbook implied-probability vectors
    # Key = (game_id, market, side) -> value = implied probability
    # ------------------------------------------------------------------
    book_vectors: dict[str, dict[tuple, float]] = {}

    allowed_markets = (
        {market_type} if market_type in ("spread", "moneyline", "total") else None
    )

    for game_id, records in by_game.items():
        for record in records:
            book = record["sportsbook"]
            markets = record.get("markets", {})

            for mtype, mdata in markets.items():
                if allowed_markets and mtype not in allowed_markets:
                    continue

                if mtype == "spread":
                    sides = [
                        ("home_spread", mdata.get("home_odds")),
                        ("away_spread", mdata.get("away_odds")),
                    ]
                elif mtype == "moneyline":
                    sides = [
                        ("home_ml", mdata.get("home_odds")),
                        ("away_ml", mdata.get("away_odds")),
                    ]
                elif mtype == "total":
                    sides = [
                        ("over", mdata.get("over_odds")),
                        ("under", mdata.get("under_odds")),
                    ]
                else:
                    continue

                for side_label, odds_val in sides:
                    if odds_val is None:
                        continue
                    prob = implied_probability(odds_val)
                    if prob and prob > 0:
                        key = (game_id, mtype, side_label)
                        book_vectors.setdefault(book, {})[key] = prob

    books = sorted(book_vectors.keys())
    n = len(books)

    if n < 2:
        return json.dumps({"error": "Need at least 2 sportsbooks for correlation network"})

    # ------------------------------------------------------------------
    # Step 2: Pairwise Pearson correlation on shared implied-prob vectors
    # ------------------------------------------------------------------
    def _pearson_from_vecs(va: dict[tuple, float], vb: dict[tuple, float]):
        shared = set(va.keys()) & set(vb.keys())
        if len(shared) < 3:
            return None, len(shared)
        xs = [va[k] for k in shared]
        ys = [vb[k] for k in shared]
        n_pts = len(xs)
        mx = sum(xs) / n_pts
        my = sum(ys) / n_pts
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n_pts
        sx = sqrt(sum((x - mx) ** 2 for x in xs) / n_pts)
        sy = sqrt(sum((y - my) ** 2 for y in ys) / n_pts)
        if sx == 0 or sy == 0:
            return None, n_pts
        return round(cov / (sx * sy), 6), n_pts

    corr_matrix: dict[str, dict[str, float | None]] = {b: {} for b in books}
    edges: list[dict] = []

    for i in range(n):
        corr_matrix[books[i]][books[i]] = 1.0
        for j in range(i + 1, n):
            ba, bb = books[i], books[j]
            r, shared_n = _pearson_from_vecs(book_vectors[ba], book_vectors[bb])
            corr_matrix[ba][bb] = r
            corr_matrix[bb][ba] = r

            # Classify the relationship
            if r is None:
                strength = "insufficient_data"
            elif r >= 0.98:
                strength = "shared_feed"
            elif r >= 0.95:
                strength = "closely_aligned"
            elif r >= 0.85:
                strength = "moderately_aligned"
            else:
                strength = "independently_priced"

            edges.append({
                "book_a": ba,
                "book_b": bb,
                "pearson_r": r,
                "shared_data_points": shared_n,
                "strength": strength,
            })

    # Sort edges by correlation descending
    edges.sort(key=lambda e: e["pearson_r"] if e["pearson_r"] is not None else -2, reverse=True)

    # ------------------------------------------------------------------
    # Step 3: Per-book summary — average correlation with all others
    # ------------------------------------------------------------------
    book_summaries: list[dict] = []
    for book in books:
        peer_corrs = [
            corr_matrix[book][other]
            for other in books
            if other != book and corr_matrix[book][other] is not None
        ]
        avg_corr = round(mean(peer_corrs), 4) if peer_corrs else None

        # Most and least correlated peer
        peer_pairs = [
            (other, corr_matrix[book][other])
            for other in books
            if other != book and corr_matrix[book][other] is not None
        ]
        peer_pairs.sort(key=lambda p: p[1], reverse=True)

        most_correlated = peer_pairs[0] if peer_pairs else (None, None)
        least_correlated = peer_pairs[-1] if peer_pairs else (None, None)

        # Classification
        if avg_corr is None:
            role = "unknown"
        elif avg_corr >= 0.97:
            role = "feed_follower"
        elif avg_corr >= 0.92:
            role = "mainstream"
        else:
            role = "independent_pricer"

        book_summaries.append({
            "sportsbook": book,
            "avg_correlation": avg_corr,
            "role": role,
            "most_correlated_with": most_correlated[0],
            "most_correlated_r": most_correlated[1],
            "least_correlated_with": least_correlated[0],
            "least_correlated_r": least_correlated[1],
            "data_points": len(book_vectors.get(book, {})),
        })

    book_summaries.sort(key=lambda b: b["avg_correlation"] or 0, reverse=True)

    # ------------------------------------------------------------------
    # Step 4: Identify shared-feed groups and independent value sources
    # ------------------------------------------------------------------
    shared_feed_pairs = [e for e in edges if e["strength"] == "shared_feed"]
    independent_books = [b for b in book_summaries if b["role"] == "independent_pricer"]

    # ------------------------------------------------------------------
    # Step 5: Insights
    # ------------------------------------------------------------------
    insights = []

    if shared_feed_pairs:
        names = set()
        for p in shared_feed_pairs:
            names.add(p["book_a"])
            names.add(p["book_b"])
        insights.append(
            f"Likely shared odds feeds detected among: {', '.join(sorted(names))} "
            f"({len(shared_feed_pairs)} pair(s) with r >= 0.98). "
            f"Shopping between these books adds minimal value."
        )

    if independent_books:
        indie_names = [b["sportsbook"] for b in independent_books]
        insights.append(
            f"Independent pricers: {', '.join(indie_names)} — "
            f"these books set lines independently and are valuable for line shopping."
        )

    if edges:
        top = edges[0]
        bot = edges[-1]
        insights.append(
            f"Most correlated pair: {top['book_a']} & {top['book_b']} (r={top['pearson_r']})"
        )
        insights.append(
            f"Least correlated pair: {bot['book_a']} & {bot['book_b']} (r={bot['pearson_r']})"
        )

    market_label = market_type if market_type else "all markets"

    result = {
        "correlation_matrix": {b: {b2: corr_matrix[b][b2] for b2 in books} for b in books},
        "edges": edges,
        "most_correlated_pairs": edges[:5],
        "least_correlated_pairs": edges[-5:][::-1] if len(edges) >= 5 else list(reversed(edges)),
        "book_summaries": book_summaries,
        "shared_feed_pairs": shared_feed_pairs,
        "independent_value_sources": independent_books,
        "insights": insights,
        "methodology": (
            f"Built implied-probability vectors for each sportsbook across {market_label}. "
            "For each pair, computed Pearson correlation on all shared data points. "
            "r >= 0.98 = likely shared feed, r >= 0.95 = closely aligned, "
            "r >= 0.85 = moderately aligned, r < 0.85 = independently priced. "
            "Per-book avg correlation classifies books as feed-followers (>=0.97), "
            "mainstream (>=0.92), or independent pricers (<0.92)."
        ),
        "market_filter": market_type or "all",
        "total_sportsbooks": n,
        "total_pairs": len(edges),
        "context": (
            f"Sportsbook correlation network for {n} books across {market_label}. "
            f"{len(shared_feed_pairs)} shared-feed pair(s), "
            f"{len(independent_books)} independent pricer(s). "
            + (insights[0] if insights else "")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# ODDS SHAPE ANALYSIS — Visual Heatmap & Pattern Recognition for Integrity
# ═══════════════════════════════════════════════════════════════════════════


def _build_odds_matrix(records: list[dict], market_type: str) -> tuple[list[str], list[str], list[list[float]]]:
    """Build a sportsbook x metric matrix of implied probabilities for a game.

    Returns (book_names, metric_labels, matrix) where matrix[i][j] is book i's
    value for metric j.  Values are implied probabilities (0-1) so that spreads,
    moneylines, and totals are on a comparable scale.
    """
    if market_type in ("spread", "moneyline"):
        odds_keys = [("home_odds", "Home"), ("away_odds", "Away")]
    else:
        odds_keys = [("over_odds", "Over"), ("under_odds", "Under")]

    metric_labels = [label for _, label in odds_keys]
    if market_type == "spread":
        metric_labels.append("Line")
    elif market_type == "total":
        metric_labels.append("Line")

    books: list[str] = []
    matrix: list[list[float]] = []

    for rec in records:
        market = rec.get("markets", {}).get(market_type)
        if not market:
            continue
        row: list[float] = []
        for key, _ in odds_keys:
            odds_val = market.get(key)
            if odds_val is not None:
                row.append(round(implied_probability(odds_val), 6))
            else:
                row.append(0.0)
        # Add line value (normalized) for spread/total
        if market_type == "spread" and "home_line" in market:
            row.append(market["home_line"])
        elif market_type == "total" and "line" in market:
            row.append(market["line"])
        books.append(rec["sportsbook"])
        matrix.append(row)

    return books, metric_labels, matrix


def _ascii_heatmap(books: list[str], metrics: list[str], matrix: list[list[float]],
                   consensus_row: list[float]) -> str:
    """Render a compact ASCII heatmap showing each book's deviation from consensus.

    Each cell shows a symbol indicating deviation magnitude:
        .  = within 0.5 std (normal)
        *  = 0.5-1.0 std (mild)
        #  = 1.0-1.5 std (moderate)
        @  = 1.5-2.0 std (notable)
        X  = >2.0 std (extreme -- potential anomaly)
    Plus sign (+/-) indicates direction.
    """
    if not matrix or not consensus_row:
        return "(no data)"

    n_metrics = len(metrics)

    # Compute stdev per column
    stdevs = []
    for j in range(n_metrics):
        col = [matrix[i][j] for i in range(len(matrix))]
        stdevs.append(pstdev(col) if len(col) > 1 else 0.001)

    # Header
    max_book = max(len(b) for b in books) if books else 10
    header = " " * (max_book + 2) + "  ".join(f"{m:>8s}" for m in metrics)
    lines = [header, " " * (max_book + 2) + "-" * (10 * n_metrics)]

    # Consensus row
    consensus_line = f"{'[CONSENSUS]':<{max_book + 2}}" + "  ".join(
        f"{v:>8.4f}" for v in consensus_row
    )
    lines.append(consensus_line)
    lines.append(" " * (max_book + 2) + "-" * (10 * n_metrics))

    symbols = [".", "*", "#", "@", "X"]

    for i, book in enumerate(books):
        cells = []
        for j in range(n_metrics):
            diff = matrix[i][j] - consensus_row[j]
            if stdevs[j] > 0:
                z = abs(diff) / stdevs[j]
            else:
                z = 0
            sign = "+" if diff > 0 else "-" if diff < 0 else " "
            idx = min(int(z * 2), 4)  # map z-score to 0-4
            sym = symbols[idx]
            val_str = f"{sign}{sym}{matrix[i][j]:.4f}"
            cells.append(f"{val_str:>8s}")
        lines.append(f"{book:<{max_book + 2}}" + "  ".join(cells))

    return "\n".join(lines)


def _detect_shape_anomalies(books: list[str], matrix: list[list[float]],
                            consensus_row: list[float], stdev_threshold: float = 1.8
                            ) -> list[dict]:
    """Detect abnormal odds 'shapes' using multi-dimensional z-score analysis.

    Inspired by research in Nature/Scientific Reports demonstrating that
    converting odds into visual patterns and applying pattern recognition
    achieves 92%+ accuracy in detecting manipulated/fixed matches.

    The method:
    1. Each book's odds profile forms a 'shape' (vector of implied probs).
    2. The consensus shape is the average across all books.
    3. Per-book deviation is measured as a composite z-score across all metrics.
    4. Books with systematically skewed shapes (not just one outlier metric
       but a coordinated shift) receive higher anomaly scores.
    5. Shape asymmetry (e.g., home-side way off but away-side normal) flags
       potential directional manipulation.
    """
    if not matrix or len(matrix) < 3:
        return []

    n_metrics = len(consensus_row)
    anomalies = []

    # Column stdevs
    col_stdevs = []
    for j in range(n_metrics):
        col = [matrix[i][j] for i in range(len(matrix))]
        col_stdevs.append(pstdev(col) if len(col) > 1 else 0.001)

    for i, book in enumerate(books):
        z_scores = []
        deviations = []
        for j in range(n_metrics):
            diff = matrix[i][j] - consensus_row[j]
            z = diff / col_stdevs[j] if col_stdevs[j] > 0 else 0
            z_scores.append(z)
            deviations.append(diff)

        # Composite anomaly score: RMS of z-scores (catches coordinated shifts)
        rms_z = sqrt(sum(z * z for z in z_scores) / len(z_scores)) if z_scores else 0

        # Shape asymmetry: if first two metrics (home/away or over/under)
        # deviate in opposite directions beyond threshold, flag it
        asymmetry = 0.0
        if len(z_scores) >= 2:
            # In a normal market, if home prob goes up, away prob should go down
            # (they're complementary). Anomaly = both shift same direction significantly.
            asymmetry = abs(z_scores[0] + z_scores[1])  # should be near 0 for complementary odds

        # Max single-metric z-score
        max_z = max(abs(z) for z in z_scores) if z_scores else 0

        # Directional coherence: do all deviations point the same way?
        signs = [1 if d > 0 else -1 if d < 0 else 0 for d in deviations[:2]]
        directional_bias = abs(sum(signs)) / max(len(signs), 1)

        is_anomalous = rms_z >= stdev_threshold or max_z >= (stdev_threshold + 0.5) or asymmetry >= 2.5

        if is_anomalous:
            # Classify the anomaly type
            if asymmetry >= 2.5:
                anomaly_type = "shape_asymmetry"
                desc = (
                    f"Complementary odds deviate in unexpected pattern -- "
                    f"both sides shifted similarly (asymmetry score: {round(asymmetry, 2)}) "
                    f"which is inconsistent with normal market behavior."
                )
            elif directional_bias >= 0.8 and rms_z >= stdev_threshold:
                anomaly_type = "coordinated_shift"
                desc = (
                    f"All metrics shifted in the same direction vs consensus -- "
                    f"suggests systematic mispricing or delayed adjustment "
                    f"(RMS z-score: {round(rms_z, 2)})."
                )
            elif max_z >= stdev_threshold + 0.5:
                anomaly_type = "single_metric_spike"
                spike_idx = max(range(len(z_scores)), key=lambda k: abs(z_scores[k]))
                desc = (
                    f"Extreme deviation on a single metric "
                    f"(z-score: {round(z_scores[spike_idx], 2)} on metric index {spike_idx}) -- "
                    f"possible stale line or intentional outlier."
                )
            else:
                anomaly_type = "general_deviation"
                desc = f"Overall shape deviates significantly from market consensus (RMS z: {round(rms_z, 2)})."

            anomalies.append({
                "sportsbook": book,
                "anomaly_type": anomaly_type,
                "rms_z_score": round(rms_z, 3),
                "max_z_score": round(max_z, 3),
                "asymmetry_score": round(asymmetry, 3),
                "directional_bias": round(directional_bias, 2),
                "z_scores_by_metric": {
                    m: round(z, 3) for m, z in zip(
                        [f"metric_{k}" for k in range(n_metrics)] if not consensus_row else
                        [f"dim_{k}" for k in range(n_metrics)],
                        z_scores
                    )
                },
                "raw_deviations": [round(d, 6) for d in deviations],
                "description": desc,
                "severity": "high" if rms_z >= stdev_threshold + 0.5 else "medium",
            })

    # Sort by severity (highest anomaly score first)
    anomalies.sort(key=lambda a: -a["rms_z_score"])
    return anomalies


def _compute_game_integrity_score(all_market_anomalies: list[dict]) -> dict:
    """Compute a per-game integrity score (0-100) from cross-market anomaly patterns.

    A game with anomalies in multiple markets, or anomalies from the same book
    across markets, receives a lower integrity score.  This mirrors the
    multi-feature pattern recognition approach from scientific literature that
    achieves 92%+ detection accuracy on known fixed matches.
    """
    if not all_market_anomalies:
        return {"integrity_score": 100, "risk_level": "clean", "flags": []}

    # Count anomalies per book across all markets
    book_anomaly_counts: dict[str, int] = {}
    book_rms_scores: dict[str, list[float]] = {}
    total_anomalies = len(all_market_anomalies)
    high_severity_count = sum(1 for a in all_market_anomalies if a.get("severity") == "high")

    for a in all_market_anomalies:
        b = a["sportsbook"]
        book_anomaly_counts[b] = book_anomaly_counts.get(b, 0) + 1
        book_rms_scores.setdefault(b, []).append(a["rms_z_score"])

    # Penalty factors
    multi_market_penalty = sum(1 for c in book_anomaly_counts.values() if c >= 2) * 15
    high_severity_penalty = high_severity_count * 10
    volume_penalty = min(total_anomalies * 5, 30)

    raw_score = 100 - multi_market_penalty - high_severity_penalty - volume_penalty
    integrity_score = max(0, min(100, raw_score))

    # Risk classification
    if integrity_score >= 85:
        risk_level = "clean"
    elif integrity_score >= 70:
        risk_level = "low_risk"
    elif integrity_score >= 50:
        risk_level = "moderate_risk"
    elif integrity_score >= 30:
        risk_level = "elevated_risk"
    else:
        risk_level = "high_risk"

    flags = []
    for book, count in book_anomaly_counts.items():
        if count >= 2:
            avg_rms = mean(book_rms_scores[book])
            flags.append(
                f"{book} flagged in {count} markets (avg RMS z-score: {round(avg_rms, 2)}) -- "
                f"cross-market anomaly pattern detected"
            )
    if high_severity_count >= 2:
        flags.append(
            f"{high_severity_count} high-severity anomalies detected -- "
            f"warrants closer inspection"
        )

    return {
        "integrity_score": integrity_score,
        "risk_level": risk_level,
        "total_anomalies": total_anomalies,
        "high_severity_count": high_severity_count,
        "multi_market_books": {b: c for b, c in book_anomaly_counts.items() if c >= 2},
        "flags": flags,
    }


@mcp.tool()
def get_odds_shape_analysis(game_id: Optional[str] = None,
                            filename: Optional[str] = None,
                            stdev_threshold: float = 1.8) -> str:
    """Visualize odds across all sportsbooks as heatmaps and detect abnormal shapes.

    Converts each game's odds across all 8 sportsbooks into an ASCII heatmap
    and uses multi-dimensional pattern recognition to spot anomalous odds
    profiles.  Based on methodology from Nature/Scientific Reports research
    showing that visual odds-pattern analysis achieves 92%+ accuracy in
    detecting fixed or manipulated matches.

    How it works:
    1. For each game and market (spread, moneyline, total), odds are converted
       to implied probabilities to create a sportsbook-by-metric matrix.
    2. The matrix is rendered as an ASCII heatmap showing each book's deviation
       from the consensus shape (./*/# /@/X = normal to extreme).
    3. Pattern recognition scores each book's odds "shape" using:
       - RMS z-score: catches coordinated multi-metric shifts
       - Shape asymmetry: detects when complementary odds break expected patterns
       - Directional bias: flags systematic one-way deviations
    4. A per-game integrity score (0-100) synthesizes cross-market anomalies.

    Args:
        game_id:  Analyze a specific game.  If omitted, analyzes all games.
        filename: Data file to load.  Optional -- defaults to first available.
        stdev_threshold: Z-score threshold for flagging anomalies (default 1.8).

    Returns JSON with:
        games[].heatmaps       -- ASCII heatmaps per market
        games[].anomalies      -- Flagged sportsbooks with anomaly details
        games[].integrity      -- Integrity score (0-100) and risk level
        summary                -- Aggregate counts and high-risk games
    """
    cache_key = f"odds_shape:{game_id or 'all'}:{stdev_threshold}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No odds data found"})

    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    if not games:
        return json.dumps({"error": f"No games found{' for ' + game_id if game_id else ''}"})

    game_results = []
    total_anomalies = 0
    high_risk_games = []

    for gid, records in games.items():
        first = records[0]
        game_entry = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "sportsbook_count": len(records),
            "heatmaps": {},
            "anomalies_by_market": {},
        }

        all_game_anomalies = []

        for market_type in ("spread", "moneyline", "total"):
            books, metrics, matrix = _build_odds_matrix(records, market_type)
            if len(books) < 2:
                continue

            # Consensus row = column means
            n_cols = len(metrics)
            consensus_row = []
            for j in range(n_cols):
                col = [matrix[i][j] for i in range(len(matrix))]
                consensus_row.append(round(mean(col), 6))

            # ASCII heatmap
            heatmap = _ascii_heatmap(books, metrics, matrix, consensus_row)
            game_entry["heatmaps"][market_type] = heatmap

            # Pattern recognition
            anomalies = _detect_shape_anomalies(
                books, matrix, consensus_row, stdev_threshold
            )
            if anomalies:
                # Re-label z_scores_by_metric with actual metric names
                for a in anomalies:
                    a["z_scores_by_metric"] = {
                        metrics[k]: round(list(a["z_scores_by_metric"].values())[k], 3)
                        for k in range(min(len(metrics), len(a["z_scores_by_metric"])))
                    }
                game_entry["anomalies_by_market"][market_type] = anomalies
                all_game_anomalies.extend(anomalies)

        # Per-game integrity score
        integrity = _compute_game_integrity_score(all_game_anomalies)
        game_entry["integrity"] = integrity
        total_anomalies += integrity["total_anomalies"]

        if integrity["risk_level"] in ("elevated_risk", "high_risk"):
            high_risk_games.append({
                "game_id": gid,
                "matchup": f"{first.get('away_team')} @ {first.get('home_team')}",
                "integrity_score": integrity["integrity_score"],
                "risk_level": integrity["risk_level"],
                "flags": integrity["flags"],
            })

        game_results.append(game_entry)

    # Sort games by integrity score (lowest = most suspicious first)
    game_results.sort(key=lambda g: g["integrity"]["integrity_score"])

    result = {
        "games": game_results,
        "summary": {
            "games_analyzed": len(game_results),
            "total_anomalies_detected": total_anomalies,
            "high_risk_games": high_risk_games,
            "high_risk_count": len(high_risk_games),
            "clean_game_count": sum(
                1 for g in game_results if g["integrity"]["risk_level"] == "clean"
            ),
        },
        "methodology": (
            "Each game's odds across all sportsbooks are converted into implied-probability "
            "matrices (one per market type).  An ASCII heatmap visualizes each book's "
            "deviation from consensus using z-score bands (. * # @ X).  Multi-dimensional "
            "pattern recognition then scores each book's odds 'shape' via: (1) RMS z-score "
            "for coordinated shifts, (2) shape asymmetry for broken complementary-odds "
            "patterns, and (3) directional bias for systematic skew.  Per-game integrity "
            "scores (0-100) synthesize cross-market anomalies.  Inspired by research from "
            "Nature/Scientific Reports demonstrating that visual odds-pattern analysis "
            "achieves 92%+ accuracy in detecting fixed or manipulated matches."
        ),
        "thresholds": {
            "stdev_threshold": stdev_threshold,
            "heatmap_bands": {
                ".": "within 0.5 std (normal)",
                "*": "0.5-1.0 std (mild deviation)",
                "#": "1.0-1.5 std (moderate)",
                "@": "1.5-2.0 std (notable)",
                "X": ">2.0 std (extreme -- potential anomaly)",
            },
            "risk_levels": {
                "clean": "85-100",
                "low_risk": "70-84",
                "moderate_risk": "50-69",
                "elevated_risk": "30-49",
                "high_risk": "0-29",
            },
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)




# ═══════════════════════════════════════════════════════════════════════════
# GAMLSS Statistical Modeling — Location, Scale, and Shape
# ═══════════════════════════════════════════════════════════════════════════


def _gamlss_fit(values: list[float]) -> dict:
    """Fit a GAMLSS-style model to a 1-D sample of odds/implied-probability values.

    Models three distributional parameters:
      - mu    (location): central tendency -- trimmed mean (robust to outliers)
      - sigma (scale):    dispersion -- MAD-based robust standard deviation
      - nu    (shape):    asymmetry -- adjusted Fisher-Pearson skewness

    The trimmed mean removes the top and bottom 10% of values to resist outlier
    pull.  MAD (median absolute deviation) x 1.4826 is a robust scale estimator
    that equals sigma for Gaussian data but resists heavy tails.  The skewness
    captures directional bias in the distribution (e.g., most books low but one
    book extremely high).

    Returns dict with mu, sigma, nu, n, plus derived diagnostics.
    """
    n = len(values)
    if n < 3:
        return {"mu": mean(values) if values else 0, "sigma": 0, "nu": 0, "n": n,
                "kurtosis_excess": 0, "sigma_method": "insufficient_data"}

    sorted_vals = sorted(values)

    # --- Location (mu): trimmed mean (10% each tail) ---
    trim = max(1, int(n * 0.1))
    trimmed = sorted_vals[trim: n - trim] if n > 4 else sorted_vals
    mu = sum(trimmed) / len(trimmed)

    # --- Scale (sigma): MAD-based robust estimator ---
    median_val = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    abs_devs = sorted(abs(v - median_val) for v in values)
    mad = abs_devs[len(abs_devs) // 2] if abs_devs else 0
    sigma = mad * 1.4826  # consistency constant for Gaussian equivalence

    # Fallback: if MAD is 0 (all values identical or near-identical), use pstdev
    if sigma < 1e-9:
        sigma = pstdev(values)

    # --- Shape (nu): adjusted Fisher-Pearson skewness ---
    if sigma > 1e-9:
        m3 = sum((v - mu) ** 3 for v in values) / n
        nu = m3 / (sigma ** 3)
        # Apply small-sample adjustment (N/(N-1)(N-2)) if n >= 3
        if n >= 3:
            nu = nu * (n * n) / ((n - 1) * (n - 2)) if (n - 1) * (n - 2) > 0 else nu
    else:
        nu = 0.0

    # --- Excess kurtosis (bonus shape parameter, "tau" in full GAMLSS) ---
    if sigma > 1e-9 and n >= 4:
        m4 = sum((v - mu) ** 4 for v in values) / n
        kurtosis_excess = (m4 / (sigma ** 4)) - 3.0
    else:
        kurtosis_excess = 0.0

    return {
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "nu": round(nu, 4),
        "kurtosis_excess": round(kurtosis_excess, 4),
        "median": round(median_val, 6),
        "mad": round(mad, 6),
        "n": n,
        "sigma_method": "MAD" if mad * 1.4826 >= 1e-9 else "pstdev_fallback",
    }


def _gamlss_zscore(value: float, fit: dict) -> float:
    """Compute a GAMLSS-aware z-score that accounts for skewness.

    For symmetric distributions this reduces to the standard z-score.
    For skewed distributions, the Azzalini skew-normal approximation adjusts
    the effective sigma based on which tail the value falls in:
      - If nu > 0 (right-skewed), values above mu get a *wider* effective sigma
        (making moderate highs less anomalous) while values below mu get a
        *narrower* effective sigma (making lows more anomalous).
      - Vice versa for nu < 0.

    This is the key insight: in GAMLSS you don't just ask "how far from the mean?"
    -- you ask "how far from the mean *given the shape of the distribution*?"
    """
    sigma = fit["sigma"]
    if sigma < 1e-9:
        return 0.0
    nu = fit["nu"]
    mu = fit["mu"]
    diff = value - mu

    # Skew-adjusted sigma: widen sigma on the heavy-tail side, narrow on the thin side
    # This uses a simplified Azzalini skew-normal approximation
    skew_adjustment = 1.0
    if abs(nu) > 0.05:
        # Positive nu = right-skewed: right tail is heavier
        # If diff > 0 (value above mean), increase effective sigma (less anomalous)
        # If diff < 0 (value below mean), decrease effective sigma (more anomalous)
        direction = 1.0 if diff > 0 else -1.0
        # Clamp adjustment factor to [0.5, 2.0] for stability
        raw_adj = 1.0 + direction * nu * 0.15
        skew_adjustment = max(0.5, min(2.0, raw_adj))

    effective_sigma = sigma * skew_adjustment
    return diff / effective_sigma if effective_sigma > 1e-9 else 0.0


def _gamlss_analyze_game_market(records: list[dict], market_type: str) -> dict | None:
    """Run GAMLSS modeling on a single game-market combination.

    For a given market (spread/moneyline/total), collects the implied
    probabilities from each sportsbook, fits GAMLSS parameters, then
    scores each book using the skew-aware z-score.

    Returns a dict with the fit parameters, per-book scores, and flagged anomalies.
    """
    if market_type in ("spread", "moneyline"):
        odds_keys = [("home_odds", "home"), ("away_odds", "away")]
    else:
        odds_keys = [("over_odds", "over"), ("under_odds", "under")]

    # Collect implied probs per side
    sides_data: dict[str, list[tuple[str, float]]] = {label: [] for _, label in odds_keys}
    line_data: list[tuple[str, float]] = []

    for rec in records:
        market = rec.get("markets", {}).get(market_type)
        if not market:
            continue
        book = rec.get("sportsbook", "unknown")
        for key, label in odds_keys:
            if key in market:
                prob = implied_probability(market[key])
                sides_data[label].append((book, prob))
        # Collect lines too
        if market_type == "spread" and "home_line" in market:
            line_data.append((book, market["home_line"]))
        elif market_type == "total" and "line" in market:
            line_data.append((book, market["line"]))

    # Need at least 3 books for meaningful distributional analysis
    min_books = min(len(v) for v in sides_data.values()) if sides_data else 0
    if min_books < 3:
        return None

    # Fit GAMLSS for each side
    fits: dict[str, dict] = {}
    for label, book_vals in sides_data.items():
        vals = [v for _, v in book_vals]
        fits[label] = _gamlss_fit(vals)

    # Fit GAMLSS for line if available
    if len(line_data) >= 3:
        fits["line"] = _gamlss_fit([v for _, v in line_data])

    # Score each book using skew-aware z-scores
    book_scores: list[dict] = []
    # Build a book -> values map across all dimensions
    book_map: dict[str, dict[str, float]] = {}
    for label, book_vals in sides_data.items():
        for book, val in book_vals:
            book_map.setdefault(book, {})[label] = val
    for book, val in line_data:
        book_map.setdefault(book, {})["line"] = val

    anomalies = []
    for book, dims in book_map.items():
        z_scores = {}
        for dim_name, val in dims.items():
            if dim_name in fits:
                z = _gamlss_zscore(val, fits[dim_name])
                z_scores[dim_name] = round(z, 3)

        # Composite: RMS of skew-adjusted z-scores
        if z_scores:
            rms_z = round(sqrt(sum(z * z for z in z_scores.values()) / len(z_scores)), 3)
        else:
            rms_z = 0.0

        max_abs_z = max((abs(z) for z in z_scores.values()), default=0.0)

        entry = {
            "sportsbook": book,
            "z_scores": z_scores,
            "rms_z_score": rms_z,
            "max_abs_z": round(max_abs_z, 3),
            "raw_values": {k: round(v, 6) for k, v in dims.items()},
        }
        book_scores.append(entry)

        # Flag anomalies: RMS >= 1.8 or any single dimension >= 2.5
        if rms_z >= 1.8 or max_abs_z >= 2.5:
            anomaly_type = "distributional_outlier"
            if max_abs_z >= 2.5 and rms_z < 1.8:
                anomaly_type = "tail_outlier"
                tail_dim = max(z_scores, key=lambda k: abs(z_scores[k]))
                nu_val = fits.get(tail_dim, {}).get("nu", 0)
                desc = (
                    f"Extreme value on {tail_dim} (skew-adjusted z={z_scores[tail_dim]}) -- "
                    f"distribution skewness nu={round(nu_val, 3)} means this "
                    f"{'is even more anomalous (against the skew)' if (z_scores[tail_dim] * nu_val) < 0 else 'is partially explained by heavy tail'}"
                )
            elif any(abs(fits.get(d, {}).get("nu", 0)) > 1.0 for d in z_scores):
                anomaly_type = "skew_anomaly"
                skewed_dims = [d for d in z_scores if abs(fits.get(d, {}).get("nu", 0)) > 1.0]
                desc = (
                    f"Anomalous under skewed distribution (RMS z={rms_z}) -- "
                    f"high-skew dimensions: {', '.join(skewed_dims)} -- "
                    f"GAMLSS shape modeling reveals this book deviates from the "
                    f"non-Gaussian pattern the market exhibits"
                )
            else:
                desc = (
                    f"Multi-dimensional distributional outlier (RMS z={rms_z}) -- "
                    f"deviates from consensus location, scale, AND shape"
                )

            anomalies.append({
                **entry,
                "anomaly_type": anomaly_type,
                "description": desc,
                "severity": "high" if rms_z >= 2.5 or max_abs_z >= 3.0 else "medium",
            })

    book_scores.sort(key=lambda b: -b["rms_z_score"])
    anomalies.sort(key=lambda a: -a["rms_z_score"])

    # Distribution characterization
    dist_summary = {}
    for dim_name, fit in fits.items():
        shape_desc = "symmetric"
        if fit["nu"] > 0.5:
            shape_desc = "right-skewed (heavy upper tail)"
        elif fit["nu"] < -0.5:
            shape_desc = "left-skewed (heavy lower tail)"
        tail_desc = "normal tails"
        if fit["kurtosis_excess"] > 1.0:
            tail_desc = "heavy-tailed (leptokurtic)"
        elif fit["kurtosis_excess"] < -0.5:
            tail_desc = "thin-tailed (platykurtic)"

        dist_summary[dim_name] = {
            "mu_location": fit["mu"],
            "sigma_scale": fit["sigma"],
            "nu_shape_skewness": fit["nu"],
            "kurtosis_excess": fit["kurtosis_excess"],
            "shape_description": shape_desc,
            "tail_description": tail_desc,
            "book_count": fit["n"],
        }

    return {
        "market": market_type,
        "distribution_parameters": dist_summary,
        "book_scores": book_scores,
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
    }


@mcp.tool()
def get_gamlss_analysis(
    game_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Apply GAMLSS (Generalized Additive Model for Location, Scale, Shape) to odds distributions.

    Standard anomaly detection assumes odds are normally distributed across
    sportsbooks -- but real odds distributions are often *skewed* (a few books
    price aggressively on one side) or *heavy-tailed* (one outlier pulls
    everything).  GAMLSS models three parameters of the distribution:

      mu (location) -- WHERE is the center of the odds distribution?
                       Uses a robust trimmed mean to resist outlier pull.
      sigma (scale) -- HOW SPREAD OUT are the odds?
                       Uses MAD x 1.4826 (robust to heavy tails).
      nu (shape)    -- IS THE DISTRIBUTION SKEWED?
                       Positive = right-skewed (some books pricing high),
                       Negative = left-skewed (some books pricing low).

    WHY THIS MATTERS: A book offering +150 when the consensus is +130 might
    look like a 2-sigma outlier under Gaussian assumptions -- but if the
    distribution is right-skewed (nu > 0), that +150 is actually within the
    heavy tail and less anomalous than it appears.  Conversely, a book at +115
    in that same right-skewed distribution is suspiciously low and MORE
    anomalous than a standard z-score would suggest.

    The tool:
    1. Collects implied probabilities for each market across all sportsbooks
    2. Fits mu, sigma, nu (and excess kurtosis) per market dimension
    3. Computes skew-adjusted z-scores using Azzalini skew-normal approximation
    4. Flags books whose adjusted z-scores exceed thresholds (RMS >= 1.8 or single >= 2.5)
    5. Classifies anomalies: distributional_outlier, tail_outlier, skew_anomaly

    Args:
        game_id:  Analyze a specific game. If omitted, analyzes all games.
        filename: Data file to load. Optional -- defaults to first available.

    Returns JSON with:
        games[].markets[].distribution_parameters -- mu, sigma, nu per dimension
        games[].markets[].book_scores            -- per-book skew-adjusted z-scores
        games[].markets[].anomalies              -- flagged distributional outliers
        summary -- aggregate anomaly counts and skew statistics
    """
    cache_key = f"gamlss:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No odds data found"})

    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    if not games:
        return json.dumps({"error": f"No games found{' for ' + game_id if game_id else ''}"})

    game_results = []
    total_anomalies = 0
    skewed_markets = 0
    heavy_tail_markets = 0
    all_skew_values = []

    for gid, records in games.items():
        first = records[0]
        game_entry = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "sportsbook_count": len(records),
            "markets": [],
        }

        game_anomaly_count = 0
        for market_type in ("spread", "moneyline", "total"):
            market_result = _gamlss_analyze_game_market(records, market_type)
            if market_result is None:
                continue
            game_entry["markets"].append(market_result)
            game_anomaly_count += market_result["anomaly_count"]

            # Track distribution shape stats
            for dim_name, params in market_result["distribution_parameters"].items():
                nu = params["nu_shape_skewness"]
                all_skew_values.append(nu)
                if abs(nu) > 0.5:
                    skewed_markets += 1
                if params["kurtosis_excess"] > 1.0:
                    heavy_tail_markets += 1

        game_entry["total_anomalies"] = game_anomaly_count
        total_anomalies += game_anomaly_count
        game_results.append(game_entry)

    # Sort: most anomalous games first
    game_results.sort(key=lambda g: -g["total_anomalies"])

    avg_abs_skew = round(mean([abs(s) for s in all_skew_values]), 3) if all_skew_values else 0

    result = {
        "games": game_results,
        "summary": {
            "games_analyzed": len(game_results),
            "total_anomalies": total_anomalies,
            "skewed_market_dimensions": skewed_markets,
            "heavy_tail_market_dimensions": heavy_tail_markets,
            "average_absolute_skewness": avg_abs_skew,
            "distribution_insight": (
                f"{'Most' if skewed_markets > len(all_skew_values) / 2 else 'Some'} "
                f"market dimensions show non-Gaussian skew (avg |nu|={avg_abs_skew}). "
                f"{'Standard z-score methods would miss tail-aware anomalies in these markets.' if avg_abs_skew > 0.3 else 'Distributions are roughly symmetric -- standard methods are adequate here.'}"
            ) if all_skew_values else "No market data available.",
        },
        "methodology": (
            "GAMLSS (Generalized Additive Model for Location, Scale, and Shape) -- "
            "instead of assuming Gaussian odds distributions across sportsbooks, this "
            "models three parameters:\n"
            "  mu (location): Trimmed mean -- robust central tendency (10% winsorized)\n"
            "  sigma (scale): MAD x 1.4826 -- robust dispersion resistant to heavy tails\n"
            "  nu (shape): Adjusted Fisher-Pearson skewness -- directional asymmetry\n\n"
            "Anomaly scoring uses the Azzalini skew-normal approximation: the effective "
            "sigma expands on the heavy-tail side and contracts on the thin-tail side. "
            "This means a book deviating *against* the distribution's skew is flagged as "
            "MORE anomalous (it's fighting the market's natural shape), while a book "
            "deviating *with* the skew is treated as LESS anomalous (it's in the expected "
            "heavy tail).\n\n"
            "Excess kurtosis (tau) provides a fourth parameter detecting heavy-tailed or "
            "thin-tailed distributions -- useful for spotting markets where one or two "
            "books are extreme outliers (leptokurtic) vs. markets where all books agree "
            "tightly (platykurtic)."
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
                sp_probs_home_list.append(nv["fair_a"])

        if ml_probs_home_list and sp_probs_home_list:
            # Remove vig from ML too for fair comparison
            ml_fair_probs = []
            for r in records:
                ml_data = r.get("markets", {}).get("moneyline", {})
                if "home_odds" in ml_data and "away_odds" in ml_data:
                    nv = no_vig_probabilities(ml_data["home_odds"], ml_data["away_odds"])
                    ml_fair_probs.append(nv["fair_a"])
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




# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Unsupervised Clustering / KNN Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════


def _parse_ts_safe(ts_str) -> Optional[datetime]:
    """Parse an ISO timestamp string, returning None on failure."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _extract_anomaly_features(records: list[dict], consensus: dict, by_game: dict) -> list[dict]:
    """Extract a 10-dimensional feature vector from each odds record for anomaly detection.

    Features per record:
      1. spread_odds_dev    — abs deviation of spread home_odds from consensus avg
      2. ml_odds_dev        — abs deviation of moneyline home_odds from consensus avg
      3. total_odds_dev     — abs deviation of total over_odds from consensus avg
      4. spread_line_dev    — abs deviation of spread home_line from consensus avg
      5. total_line_dev     — abs deviation of total line from consensus avg
      6. vig_spread         — spread market vig (higher = more anomalous)
      7. vig_ml             — moneyline market vig
      8. vig_total          — total market vig
      9. staleness_minutes  — how stale this record is vs the freshest in its game
     10. line_move_direction — directional signal: +1 (toward underdog),
                              -1 (toward favorite), 0 (no movement detected)
    """
    features = []

    # Pre-compute newest timestamp per game for staleness
    newest_per_game: dict[str, datetime] = {}
    for r in records:
        gid = r.get("game_id", "unknown")
        ts = _parse_ts_safe(r.get("last_updated"))
        if ts and (gid not in newest_per_game or ts > newest_per_game[gid]):
            newest_per_game[gid] = ts

    for r in records:
        gid = r.get("game_id", "unknown")
        markets = r.get("markets", {})
        game_consensus = consensus.get(gid, {})

        # --- Odds deviation features ---
        spread = markets.get("spread", {})
        ml = markets.get("moneyline", {})
        total = markets.get("total", {})

        sc = game_consensus.get("spread", {})
        mc = game_consensus.get("moneyline", {})
        tc = game_consensus.get("total", {})

        spread_odds_dev = abs(spread.get("home_odds", 0) - sc.get("avg_home_odds", 0)) if sc.get("avg_home_odds") is not None and "home_odds" in spread else 0.0
        ml_odds_dev = abs(ml.get("home_odds", 0) - mc.get("avg_home_odds", 0)) if mc.get("avg_home_odds") is not None and "home_odds" in ml else 0.0
        total_odds_dev = abs(total.get("over_odds", 0) - tc.get("avg_over_odds", 0)) if tc.get("avg_over_odds") is not None and "over_odds" in total else 0.0

        # --- Line deviation features ---
        spread_line_dev = abs(spread.get("home_line", 0) - sc.get("avg_home_line", 0)) if sc.get("avg_home_line") is not None and "home_line" in spread else 0.0
        total_line_dev = abs(total.get("line", 0) - tc.get("avg_line", 0)) if tc.get("avg_line") is not None and "line" in total else 0.0

        # --- Vig features (higher vig = more unusual pricing) ---
        vig_spread = spread.get("vig", 0.0) if "vig" in spread else 0.0
        vig_ml = ml.get("vig", 0.0) if "vig" in ml else 0.0
        vig_total = total.get("vig", 0.0) if "vig" in total else 0.0

        # --- Staleness feature ---
        ts = _parse_ts_safe(r.get("last_updated"))
        newest = newest_per_game.get(gid)
        staleness_minutes = 0.0
        if ts and newest:
            staleness_minutes = max(0.0, (newest - ts).total_seconds() / 60.0)

        # --- Line movement direction ---
        # Compare this record's spread line to the consensus avg.
        # Positive = line moved toward underdog, Negative = toward favorite.
        line_move_direction = 0.0
        if sc.get("avg_home_line") is not None and "home_line" in spread:
            diff = spread["home_line"] - sc["avg_home_line"]
            if abs(diff) >= 0.5:
                line_move_direction = -1.0 if diff < 0 else 1.0

        feature_vec = [
            spread_odds_dev,
            ml_odds_dev,
            total_odds_dev,
            spread_line_dev,
            total_line_dev,
            vig_spread,
            vig_ml,
            vig_total,
            staleness_minutes,
            line_move_direction,
        ]

        features.append({
            "record": r,
            "vector": feature_vec,
            "feature_names": [
                "spread_odds_dev", "ml_odds_dev", "total_odds_dev",
                "spread_line_dev", "total_line_dev",
                "vig_spread", "vig_ml", "vig_total",
                "staleness_minutes", "line_move_direction",
            ],
            "feature_details": {
                "spread_odds_dev": round(spread_odds_dev, 2),
                "ml_odds_dev": round(ml_odds_dev, 2),
                "total_odds_dev": round(total_odds_dev, 2),
                "spread_line_dev": round(spread_line_dev, 2),
                "total_line_dev": round(total_line_dev, 2),
                "vig_spread": round(vig_spread, 4),
                "vig_ml": round(vig_ml, 4),
                "vig_total": round(vig_total, 4),
                "staleness_minutes": round(staleness_minutes, 1),
                "line_move_direction": line_move_direction,
            },
        })

    return features


def _normalize_features(features: list[dict]) -> list[list[float]]:
    """Min-max normalize feature vectors to [0, 1] range."""
    if not features:
        return []

    vectors = [f["vector"] for f in features]
    n_features = len(vectors[0])

    mins = [min(v[i] for v in vectors) for i in range(n_features)]
    maxs = [max(v[i] for v in vectors) for i in range(n_features)]

    normalized = []
    for v in vectors:
        norm = []
        for i in range(n_features):
            rng = maxs[i] - mins[i]
            norm.append((v[i] - mins[i]) / rng if rng > 0 else 0.0)
        normalized.append(norm)

    return normalized


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Euclidean distance between two vectors."""
    return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _knn_anomaly_scores(normalized: list[list[float]], k: int = 5) -> list[float]:
    """KNN anomaly score: average Euclidean distance to K nearest neighbors.

    Higher score = more anomalous (the record lives in a sparse region of
    the feature space, far from its nearest neighbors).
    """
    n = len(normalized)
    k = min(k, n - 1)
    if k <= 0:
        return [0.0] * n

    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(_euclidean_distance(normalized[i], normalized[j]))
        dists.sort()
        avg_knn_dist = sum(dists[:k]) / k
        scores.append(avg_knn_dist)

    return scores


def _isolation_forest_scores(
    vectors: list[list[float]], n_trees: int = 100, sample_size: int = 32, seed: int = 42
) -> list[float]:
    """Pure-Python Isolation Forest anomaly scoring.

    Builds an ensemble of random isolation trees. Each tree recursively
    partitions data by picking a random feature and a random split value.
    Anomalies are isolated (reach a leaf) in fewer splits on average.

    Returns anomaly scores in [0, 1] where higher = more anomalous.
    """
    n = len(vectors)
    if n < 4:
        return [0.0] * n

    n_features = len(vectors[0])
    max_depth = max(2, int(log(max(sample_size, 2)) / log(2)) + 1)
    rng = random.Random(seed)

    def _build_tree(indices: list[int], depth: int) -> dict:
        if len(indices) <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": len(indices), "depth": depth}

        feat = rng.randint(0, n_features - 1)
        vals = [vectors[idx][feat] for idx in indices]
        lo, hi = min(vals), max(vals)

        if lo == hi:
            return {"type": "leaf", "size": len(indices), "depth": depth}

        split = rng.uniform(lo, hi)
        left_idx = [idx for idx in indices if vectors[idx][feat] < split]
        right_idx = [idx for idx in indices if vectors[idx][feat] >= split]

        if not left_idx or not right_idx:
            return {"type": "leaf", "size": len(indices), "depth": depth}

        return {
            "type": "split",
            "feature": feat,
            "split_value": split,
            "left": _build_tree(left_idx, depth + 1),
            "right": _build_tree(right_idx, depth + 1),
        }

    def _path_length(node: dict, vec: list[float]) -> float:
        if node["type"] == "leaf":
            size = node["size"]
            if size <= 1:
                return node["depth"]
            # Average path length of unsuccessful BST search (Euler-Mascheroni)
            c = 2.0 * (log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size
            return node["depth"] + c
        if vec[node["feature"]] < node["split_value"]:
            return _path_length(node["left"], vec)
        else:
            return _path_length(node["right"], vec)

    # Build forest
    trees = []
    all_indices = list(range(n))
    actual_sample = min(sample_size, n)
    for _ in range(n_trees):
        sample_idx = rng.sample(all_indices, actual_sample) if actual_sample < n else list(all_indices)
        trees.append(_build_tree(sample_idx, 0))

    # Score each record
    c_n = 2.0 * (log(max(actual_sample - 1, 1)) + 0.5772156649) - 2.0 * (actual_sample - 1) / max(actual_sample, 1)
    if c_n == 0:
        c_n = 1.0

    scores = []
    for i in range(n):
        avg_path = sum(_path_length(tree, vectors[i]) for tree in trees) / n_trees
        # Anomaly score: 2^(-avg_path / c(n))  — closer to 1 = more anomalous
        score = 2.0 ** (-avg_path / c_n)
        scores.append(score)

    return scores


@mcp.tool()
def detect_knn_anomalies(
    game_id: Optional[str] = None,
    filename: Optional[str] = None,
    k: int = 5,
    n_trees: int = 100,
    top_n: int = 15,
    anomaly_percentile: float = 80.0,
) -> str:
    """Unsupervised anomaly detection using KNN distance + Isolation Forest on odds records.

    Extracts a 10-dimensional feature vector from each odds record and applies
    two complementary ML methods (no labeled training data needed):

    1. **K-Nearest Neighbor (KNN) distance** — records far from their K nearest
       neighbors in feature space are anomalous (they sit in sparse regions).
    2. **Isolation Forest** — an ensemble of random decision trees that isolate
       anomalies with fewer splits (anomalies have shorter average path lengths).

    The composite anomaly score (0-100) combines both signals. Records above the
    specified percentile are flagged.

    Features analyzed per record (10 dimensions):
      - Odds deviation from consensus mean (spread, moneyline, total)
      - Line deviation from consensus (spread line, total line)
      - Vig per market (spread, moneyline, total)
      - Timestamp staleness (minutes behind freshest update for same game)
      - Line movement direction (+1 toward underdog, -1 toward favorite)

    Args:
        game_id: Optional — limit analysis to a specific game.
        filename: Data file to load. Optional.
        k: Number of nearest neighbors for KNN scoring. Default 5.
        n_trees: Number of isolation trees. Default 100.
        top_n: Max anomalies to return. Default 15.
        anomaly_percentile: Score percentile above which records are flagged (0-100). Default 80.

    Returns flagged anomalies with composite scores, feature breakdowns,
    contributing factors, and methodology explanation.
    """
    cache_key = f"knn_anomalies:{game_id or 'all'}:{k}:{n_trees}:{top_n}:{anomaly_percentile}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data available"})

    by_game = _cache.load_by_game(filename)
    consensus = _cache.load_consensus(filename)

    # Filter to target game if specified
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
        if not enriched:
            return json.dumps({"error": f"No records found for game_id={game_id}"})

    # --- Step 1: Feature extraction ---
    features = _extract_anomaly_features(enriched, consensus, by_game)

    if len(features) < 4:
        return json.dumps({
            "error": "Need at least 4 records for meaningful clustering analysis",
            "record_count": len(features),
        })

    # --- Step 2: Normalize features to [0,1] ---
    normalized = _normalize_features(features)

    # --- Step 3: KNN anomaly scores ---
    knn_scores = _knn_anomaly_scores(normalized, k=k)

    # --- Step 4: Isolation Forest scores ---
    raw_vectors = [f["vector"] for f in features]
    iso_scores = _isolation_forest_scores(raw_vectors, n_trees=n_trees)

    # --- Step 5: Normalize both score sets to [0, 1] ---
    def _min_max_norm(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        return [(v - lo) / rng if rng > 0 else 0.0 for v in vals]

    knn_norm = _min_max_norm(knn_scores)
    iso_norm = _min_max_norm(iso_scores)

    # --- Step 6: Composite score (50% KNN + 50% iForest, scaled to 0-100) ---
    composite = [
        round((kn * 0.5 + iso * 0.5) * 100, 1)
        for kn, iso in zip(knn_norm, iso_norm)
    ]

    # --- Step 7: Determine threshold from percentile ---
    sorted_scores = sorted(composite)
    pct_idx = min(int(len(sorted_scores) * anomaly_percentile / 100.0), len(sorted_scores) - 1)
    threshold = sorted_scores[pct_idx]

    # --- Step 8: Build anomaly results ---
    anomalies = []
    for i, feat in enumerate(features):
        if composite[i] < threshold:
            continue

        r = feat["record"]
        details = feat["feature_details"]

        # Identify top contributing factors
        contributing = []
        if details["staleness_minutes"] > 10:
            contributing.append(f"Stale by {details['staleness_minutes']} min")
        if details["spread_odds_dev"] > 10:
            contributing.append(f"Spread odds {details['spread_odds_dev']} pts from consensus")
        if details["ml_odds_dev"] > 10:
            contributing.append(f"ML odds {details['ml_odds_dev']} pts from consensus")
        if details["total_odds_dev"] > 10:
            contributing.append(f"Total odds {details['total_odds_dev']} pts from consensus")
        if details["spread_line_dev"] > 0.5:
            contributing.append(f"Spread line {details['spread_line_dev']} pts from consensus")
        if details["total_line_dev"] > 0.5:
            contributing.append(f"Total line {details['total_line_dev']} pts from consensus")
        if details["vig_spread"] > 0.06:
            contributing.append(f"High spread vig ({round(details['vig_spread'] * 100, 1)}%)")
        if details["vig_ml"] > 0.06:
            contributing.append(f"High ML vig ({round(details['vig_ml'] * 100, 1)}%)")
        if details["vig_total"] > 0.06:
            contributing.append(f"High total vig ({round(details['vig_total'] * 100, 1)}%)")
        if details["line_move_direction"] != 0:
            direction_label = "toward underdog" if details["line_move_direction"] > 0 else "toward favorite"
            contributing.append(f"Line moved {direction_label}")

        if not contributing:
            contributing.append("Multi-dimensional deviation (no single dominant factor)")

        anomalies.append({
            "game_id": r.get("game_id"),
            "sport": r.get("sport"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "sportsbook": r.get("sportsbook"),
            "last_updated": r.get("last_updated"),
            "composite_anomaly_score": composite[i],
            "knn_score": round(knn_norm[i] * 100, 1),
            "isolation_forest_score": round(iso_norm[i] * 100, 1),
            "feature_breakdown": details,
            "contributing_factors": contributing,
            "context": (
                f"ANOMALY: {r.get('sportsbook')} on {r.get('home_team')} vs {r.get('away_team')} — "
                f"score {composite[i]}/100 (KNN={round(knn_norm[i]*100,1)}, "
                f"iForest={round(iso_norm[i]*100,1)}). "
                f"Factors: {'; '.join(contributing[:3])}"
            ),
        })

    anomalies.sort(key=lambda a: a["composite_anomaly_score"], reverse=True)
    anomalies = anomalies[:top_n]

    # --- Summary statistics ---
    all_scores = sorted(composite, reverse=True)
    score_distribution = {
        "mean": round(sum(composite) / len(composite), 1),
        "median": round(sorted_scores[len(sorted_scores) // 2], 1),
        "p90": round(sorted_scores[min(int(len(sorted_scores) * 0.9), len(sorted_scores) - 1)], 1),
        "p95": round(sorted_scores[min(int(len(sorted_scores) * 0.95), len(sorted_scores) - 1)], 1),
        "max": round(all_scores[0], 1),
        "threshold_used": round(threshold, 1),
    }

    # Per-sportsbook anomaly summary
    book_anomaly_counts: dict[str, list[float]] = {}
    for i, feat in enumerate(features):
        book = feat["record"].get("sportsbook", "unknown")
        book_anomaly_counts.setdefault(book, []).append(composite[i])

    book_summary = []
    for book, scores_list in book_anomaly_counts.items():
        avg_score = sum(scores_list) / len(scores_list)
        flagged = sum(1 for s in scores_list if s >= threshold)
        book_summary.append({
            "sportsbook": book,
            "avg_anomaly_score": round(avg_score, 1),
            "flagged_count": flagged,
            "total_records": len(scores_list),
            "flagged_pct": round(flagged / len(scores_list) * 100, 1) if scores_list else 0,
        })
    book_summary.sort(key=lambda b: b["avg_anomaly_score"], reverse=True)

    result = {
        "anomalies": anomalies,
        "count": len(anomalies),
        "total_records_analyzed": len(features),
        "anomaly_percentile": anomaly_percentile,
        "score_distribution": score_distribution,
        "sportsbook_anomaly_summary": book_summary,
        "parameters": {
            "k_neighbors": k,
            "n_trees": n_trees,
            "top_n": top_n,
            "anomaly_percentile": anomaly_percentile,
            "features_used": 10,
        },
        "methodology": (
            "Unsupervised anomaly detection combining two complementary ML methods "
            "(no labeled training data needed):\n"
            "1. K-Nearest Neighbor (KNN) Distance: Each record is scored by its average "
            "Euclidean distance to its K nearest neighbors in 10-D feature space. "
            "Records in sparse regions (far from neighbors) score higher.\n"
            "2. Isolation Forest: An ensemble of random decision trees attempts to "
            "isolate each record. Anomalies are isolated with fewer random splits "
            "(shorter average path length = higher anomaly score).\n"
            "Composite score = 50% KNN + 50% Isolation Forest, scaled to 0-100.\n\n"
            "Features: odds deviation from consensus (spread/ML/total), line deviation "
            "(spread/total), vig per market, timestamp staleness, line movement direction."
        ),
        "context": (
            f"Analyzed {len(features)} records with unsupervised KNN+iForest clustering. "
            f"Flagged {len(anomalies)} anomalies above {anomaly_percentile}th percentile "
            f"(threshold={round(threshold, 1)}). "
            + (f"Most anomalous: {anomalies[0]['context']}" if anomalies else "No anomalies detected.")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL — Bayesian Updating of True Probabilities
# ═══════════════════════════════════════════════════════════════════════════


def _get_bayesian_fair_probs(
    games: dict[str, list[dict]],
    prior_kappa: float = 20.0,
    evidence_kappa: float = 5.0,
) -> dict[str, dict[str, dict]]:
    """Compute Bayesian posterior probabilities for every game & market.

    Uses Pinnacle's no-vig line as the Beta prior, then sequentially updates
    with each additional sportsbook's no-vig probability as evidence.

    Falls back to a uniform Beta(1,1) prior (uninformative) when Pinnacle is
    unavailable, then updates with all available books.

    Returns
    -------
    {game_id: {market_type: {bayesian_update result dict + metadata}}}
    """
    result: dict[str, dict[str, dict]] = {}

    for game_id, records in games.items():
        result[game_id] = {}

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                odds_key_a, odds_key_b = "home_odds", "away_odds"
                side_a_label, side_b_label = "home", "away"
            else:
                odds_key_a, odds_key_b = "over_odds", "under_odds"
                side_a_label, side_b_label = "over", "under"

            # Separate Pinnacle from the rest
            pinnacle_fair_a, pinnacle_fair_b = None, None
            evidence_a, evidence_b = [], []
            evidence_books = []

            for r in records:
                market = r.get("markets", {}).get(market_type, {})
                if odds_key_a not in market or odds_key_b not in market:
                    continue

                fair = no_vig_probabilities(market[odds_key_a], market[odds_key_b])
                book = r.get("sportsbook", "Unknown")

                if book.lower() == SHARP_BOOK.lower():
                    pinnacle_fair_a = fair["fair_a"]
                    pinnacle_fair_b = fair["fair_b"]
                else:
                    evidence_a.append(fair["fair_a"])
                    evidence_b.append(fair["fair_b"])
                    evidence_books.append(book)

            if not evidence_a and pinnacle_fair_a is None:
                continue  # no data at all

            # Set the prior
            if pinnacle_fair_a is not None:
                prior_a = pinnacle_fair_a
                prior_b = pinnacle_fair_b
                prior_source = SHARP_BOOK
            else:
                # Uninformative prior (equivalent to Beta(1,1) mapped through kappa)
                prior_a = 0.5
                prior_b = 0.5
                prior_source = "uninformative (no Pinnacle data)"
                # Use all books as evidence since there's no sharp prior
                evidence_a, evidence_b, evidence_books = [], [], []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key_a not in market or odds_key_b not in market:
                        continue
                    fair = no_vig_probabilities(market[odds_key_a], market[odds_key_b])
                    evidence_a.append(fair["fair_a"])
                    evidence_b.append(fair["fair_b"])
                    evidence_books.append(r.get("sportsbook", "Unknown"))

            # Run Bayesian updates for both sides
            update_a = bayesian_update(prior_a, evidence_a, prior_kappa, evidence_kappa)
            update_b = bayesian_update(prior_b, evidence_b, prior_kappa, evidence_kappa)

            # Normalize posteriors so they sum to 1.0
            raw_a = update_a["posterior_prob"]
            raw_b = update_b["posterior_prob"]
            total = raw_a + raw_b
            norm_a = raw_a / total if total else 0.5
            norm_b = raw_b / total if total else 0.5

            result[game_id][market_type] = {
                f"{side_a_label}_posterior_prob": round(norm_a, 6),
                f"{side_b_label}_posterior_prob": round(norm_b, 6),
                f"{side_a_label}_posterior_prob_pct": f"{round(norm_a * 100, 2)}%",
                f"{side_b_label}_posterior_prob_pct": f"{round(norm_b * 100, 2)}%",
                f"{side_a_label}_posterior_odds": fair_odds_to_american(max(0.01, min(0.99, norm_a))),
                f"{side_b_label}_posterior_odds": fair_odds_to_american(max(0.01, min(0.99, norm_b))),
                f"{side_a_label}_raw_posterior": round(raw_a, 6),
                f"{side_b_label}_raw_posterior": round(raw_b, 6),
                f"{side_a_label}_credible_interval_90": update_a["credible_interval_90"],
                f"{side_b_label}_credible_interval_90": update_b["credible_interval_90"],
                f"{side_a_label}_std_dev": update_a["std_dev"],
                f"{side_b_label}_std_dev": update_b["std_dev"],
                f"{side_a_label}_shift_from_prior": update_a["shift_from_prior_pct"],
                f"{side_b_label}_shift_from_prior": update_b["shift_from_prior_pct"],
                "prior_source": prior_source,
                f"{side_a_label}_prior": round(prior_a, 6),
                f"{side_b_label}_prior": round(prior_b, 6),
                "evidence_books": evidence_books,
                "evidence_count": len(evidence_a),
                "prior_kappa": prior_kappa,
                "evidence_kappa": evidence_kappa,
                "total_kappa": update_a["total_kappa"],
                f"{side_a_label}_update_trace": update_a["update_trace"],
                f"{side_b_label}_update_trace": update_b["update_trace"],
            }

    return result


@mcp.tool()
def get_bayesian_probabilities(
    game_id: Optional[str] = None,
    prior_kappa: float = 20.0,
    evidence_kappa: float = 5.0,
    filename: Optional[str] = None,
) -> str:
    """Bayesian updating of true probabilities — starts with Pinnacle's no-vig
    line as a Beta prior and sequentially updates with each sportsbook's odds.

    Unlike simple averaging, this approach:
    - Weights the sharp prior (Pinnacle) more heavily than any single rec book
    - Produces a posterior distribution, not just a point estimate
    - Provides 90% credible intervals showing uncertainty
    - Shows a step-by-step trace of how each book shifted the estimate

    The prior_kappa controls how strongly to trust the Pinnacle prior (higher =
    more trust), while evidence_kappa controls how much each additional book
    shifts the posterior (higher = more influence per book).

    Args:
        game_id: Filter to a specific game. Optional (shows all games).
        prior_kappa: Concentration for the Pinnacle prior (default 20). Higher
                     means more trust in Pinnacle. Range 5-50 recommended.
        evidence_kappa: Concentration per evidence sportsbook (default 5).
                        Higher means each book has more influence. Range 2-15.
        filename: Data file to load. Optional.

    Returns Bayesian posterior probabilities, credible intervals, fair American
    odds, and step-by-step update traces per game per market.
    """
    cache_key = f"bayesian_probs:{game_id or 'all'}:{prior_kappa}:{evidence_kappa}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    games = _group_by_game(enriched)

    bayesian_probs = _get_bayesian_fair_probs(games, prior_kappa, evidence_kappa)

    # Also get Pinnacle probs for comparison
    pinnacle_probs = _get_pinnacle_fair_probs(games)

    games_list = []
    significant_shifts = []

    for gid, records in games.items():
        first = records[0]
        game_entry = {
            "game_id": gid,
            "sport": first.get("sport"),
            "home_team": first.get("home_team"),
            "away_team": first.get("away_team"),
            "markets": {},
        }

        for market_type in ("spread", "moneyline", "total"):
            bayes = bayesian_probs.get(gid, {}).get(market_type)
            if not bayes:
                continue

            pinnacle = pinnacle_probs.get(gid, {}).get(market_type, {})

            if market_type in ("spread", "moneyline"):
                side_a, side_b = "home", "away"
            else:
                side_a, side_b = "over", "under"

            # Build comparison vs simple methods
            comparison = {}
            if pinnacle:
                comparison["pinnacle_no_vig"] = {
                    f"{side_a}_prob": pinnacle.get("side_a_prob"),
                    f"{side_b}_prob": pinnacle.get("side_b_prob"),
                    "source": pinnacle.get("source"),
                }

            bayesian_a = bayes[f"{side_a}_posterior_prob"]
            pinnacle_a = pinnacle.get("side_a_prob", bayesian_a)
            diff = abs(bayesian_a - pinnacle_a)

            comparison["bayesian_vs_pinnacle_diff_pp"] = round(diff * 100, 2)

            market_entry = {**bayes, "comparison": comparison}

            # Add spread/total line context
            if market_type == "spread":
                lines = [
                    r["markets"]["spread"].get("home_line", 0)
                    for r in records
                    if "spread" in r.get("markets", {})
                ]
                if lines:
                    market_entry["consensus_line"] = round(sum(lines) / len(lines), 1)
            elif market_type == "total":
                lines = [
                    r["markets"]["total"].get("line", 0)
                    for r in records
                    if "total" in r.get("markets", {})
                ]
                if lines:
                    market_entry["consensus_line"] = round(sum(lines) / len(lines), 1)

            game_entry["markets"][market_type] = market_entry

            # Flag significant shifts (>= 1pp Bayesian vs Pinnacle)
            if diff >= 0.01:
                significant_shifts.append({
                    "game_id": gid,
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market": market_type,
                    f"{side_a}_shift_pp": round((bayesian_a - pinnacle_a) * 100, 2),
                    "bayesian_prob": round(bayesian_a, 4),
                    "pinnacle_prob": round(pinnacle_a, 4),
                    "evidence_count": bayes["evidence_count"],
                })

        games_list.append(game_entry)

    significant_shifts.sort(key=lambda x: abs(list(x.values())[4]), reverse=True)

    result = {
        "games": games_list,
        "count": len(games_list),
        "significant_shifts": significant_shifts,
        "significant_shift_count": len(significant_shifts),
        "parameters": {
            "prior_kappa": prior_kappa,
            "evidence_kappa": evidence_kappa,
            "prior_source": f"{SHARP_BOOK} no-vig (fallback: uninformative Beta(1,1))",
        },
        "methodology": (
            "Bayesian updating with Beta-conjugate priors. "
            f"Pinnacle no-vig probability is encoded as a Beta prior with kappa={prior_kappa} "
            f"(equivalent sample size). Each additional sportsbook's no-vig probability "
            f"contributes {evidence_kappa} pseudo-counts of evidence, proportionally split "
            "by their fair probability. The posterior mean is the Bayesian estimate of "
            "the true probability. 90% credible intervals quantify remaining uncertainty. "
            "Posteriors for both sides are normalized to sum to 1.0."
        ),
        "context": (
            f"Bayesian posterior probabilities for {len(games_list)} games. "
            f"{len(significant_shifts)} market(s) show >= 1pp shift from Pinnacle after "
            f"incorporating evidence from recreational books — these represent cases where "
            f"the broader market meaningfully disagrees with the sharp line."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)



# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — Closing Line Value (CLV) Simulation
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_closing_line_value(
    game_id: Optional[str] = None,
    filename: Optional[str] = None,
    closing_window_minutes: int = 15,
) -> str:
    """Simulate Closing Line Value (CLV) analysis using last_updated timestamps.

    Consistently beating the closing line is the #1 indicator of a sharp bettor.

    CLV measures whether a bet placed at earlier odds offered better value than the
    "closing line" (the latest-updated odds before game time). This tool uses
    last_updated timestamps to identify which sportsbook line is the closing line
    (most recently updated) for each game/market/side, then measures how much value
    earlier lines offered relative to that close.

    For each game, market, and side:
      1. The "closing line" is the latest-updated odds across all sportsbooks.
      2. Every earlier line is compared: did it offer better implied probability?
      3. CLV is expressed as the probability edge (early implied prob vs closing implied prob).
         Positive CLV = the early line beat the close (sharp signal).

    Also produces per-sportsbook CLV profiles showing which books are consistently
    slow to update (exploitable) and which are the sharpest closers.

    Args:
        game_id: Optional — filter to a specific game.
        filename: Data file to load. Optional.
        closing_window_minutes: Lines updated within this many minutes of the newest
            are considered "closing group" — the latest single update is the closing
            line, but lines within this window are near-close. Default 15 min.

    Returns:
        JSON with per-game CLV analysis, per-book CLV profiles, and a summary of
        which sportsbooks consistently offer beatable lines.
    """
    cache_key = f"clv:{game_id or 'all'}:{closing_window_minutes}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    if not games:
        return json.dumps({"error": "No data available"})

    if game_id:
        if game_id not in games:
            return json.dumps({"error": f"No records found for game_id={game_id}"})
        games = {game_id: games[game_id]}

    clv_opportunities = []
    game_summaries = []
    book_clv_stats: dict[str, list[float]] = {}

    for gid, records in games.items():
        first = records[0]
        game_label = f"{first.get('away_team', '?')} @ {first.get('home_team', '?')}"

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                sides = [
                    ("home", "home_odds", "home_line" if market_type == "spread" else None),
                    ("away", "away_odds", "away_line" if market_type == "spread" else None),
                ]
            else:
                sides = [
                    ("over", "over_odds", "line"),
                    ("under", "under_odds", "line"),
                ]

            for side, odds_key, line_key in sides:
                entries = []
                for r in records:
                    market = r.get("markets", {}).get(market_type, {})
                    if odds_key not in market:
                        continue
                    ts = _parse_ts_safe(r.get("last_updated"))
                    if ts is None:
                        continue
                    entries.append({
                        "sportsbook": r["sportsbook"],
                        "odds": market[odds_key],
                        "line": market.get(line_key) if line_key else None,
                        "ts": ts,
                        "last_updated": r.get("last_updated"),
                    })

                if len(entries) < 2:
                    continue

                # Sort by timestamp - latest is the closing line
                entries.sort(key=lambda e: e["ts"])
                closing_entry = entries[-1]
                closing_odds = closing_entry["odds"]
                closing_prob = implied_probability(closing_odds)
                closing_ts = closing_entry["ts"]

                for entry in entries[:-1]:
                    early_odds = entry["odds"]
                    early_prob = implied_probability(early_odds)
                    staleness = (closing_ts - entry["ts"]).total_seconds() / 60.0

                    # CLV: positive means the early line was better (lower implied
                    # prob = better odds for the bettor on that side)
                    clv_edge = closing_prob - early_prob
                    clv_pct = round(clv_edge * 100, 3)

                    near_close = staleness <= closing_window_minutes

                    book = entry["sportsbook"]
                    book_clv_stats.setdefault(book, []).append(clv_pct)

                    line_desc = ""
                    if entry["line"] is not None and closing_entry["line"] is not None:
                        if entry["line"] != closing_entry["line"]:
                            line_desc = f" (line moved {entry['line']} -> {closing_entry['line']})"

                    if clv_pct > 0:
                        clv_opportunities.append({
                            "game_id": gid,
                            "sport": first.get("sport"),
                            "game": game_label,
                            "market_type": market_type,
                            "side": side,
                            "sportsbook": book,
                            "early_odds": early_odds,
                            "closing_odds": closing_odds,
                            "closing_sportsbook": closing_entry["sportsbook"],
                            "early_implied_prob": round(early_prob, 6),
                            "closing_implied_prob": round(closing_prob, 6),
                            "clv_edge_pct": clv_pct,
                            "staleness_minutes": round(staleness, 1),
                            "near_close": near_close,
                            "line_movement": line_desc.strip() if line_desc else None,
                            "early_updated": entry["last_updated"],
                            "closing_updated": closing_entry["last_updated"],
                            "context": (
                                f"CLV +{clv_pct}%: {side} {market_type} at {book} "
                                f"({early_odds}) beat the close ({closing_odds} at "
                                f"{closing_entry['sportsbook']}). "
                                f"{round(staleness)} min before close.{line_desc}"
                            ),
                        })

        # Game-level summary
        game_clvs = [o["clv_edge_pct"] for o in clv_opportunities if o["game_id"] == gid]
        if game_clvs:
            game_summaries.append({
                "game_id": gid,
                "game": game_label,
                "sport": first.get("sport"),
                "avg_clv_pct": round(mean(game_clvs), 3),
                "max_clv_pct": round(max(game_clvs), 3),
                "positive_clv_count": len([c for c in game_clvs if c > 0]),
                "total_comparisons": len(game_clvs),
            })

    clv_opportunities.sort(key=lambda o: o["clv_edge_pct"], reverse=True)
    game_summaries.sort(key=lambda g: g["avg_clv_pct"], reverse=True)

    # -- Per-sportsbook CLV profile --
    book_profiles = []
    for book, edges in sorted(book_clv_stats.items()):
        positive = [e for e in edges if e > 0]
        avg_edge = mean(edges) if edges else 0
        beat_rate = len(positive) / len(edges) * 100 if edges else 0

        if beat_rate >= 60 and avg_edge > 0.5:
            classification = "EXPLOITABLE - consistently beatable closing lines"
        elif beat_rate >= 50 and avg_edge > 0:
            classification = "SOFT - mild CLV advantage available"
        elif beat_rate <= 30:
            classification = "SHARP - lines close near or better than peers"
        else:
            classification = "NEUTRAL - average market efficiency"

        book_profiles.append({
            "sportsbook": book,
            "total_lines": len(edges),
            "beat_close_count": len(positive),
            "beat_close_pct": round(beat_rate, 1),
            "avg_clv_pct": round(avg_edge, 3),
            "max_clv_pct": round(max(edges), 3) if edges else 0,
            "min_clv_pct": round(min(edges), 3) if edges else 0,
            "stddev_clv": round(pstdev(edges), 3) if len(edges) > 1 else 0,
            "classification": classification,
            "context": (
                f"{book}: beats close {round(beat_rate)}% of the time, "
                f"avg CLV {'+' if avg_edge > 0 else ''}{round(avg_edge, 2)}%. "
                f"{classification}"
            ),
        })

    book_profiles.sort(key=lambda b: b["avg_clv_pct"], reverse=True)

    all_edges = [o["clv_edge_pct"] for o in clv_opportunities]
    total_comparisons = sum(len(e) for e in book_clv_stats.values())

    result = {
        "clv_opportunities": clv_opportunities[:50],
        "clv_opportunity_count": len(clv_opportunities),
        "game_summaries": game_summaries,
        "sportsbook_profiles": book_profiles,
        "summary": {
            "total_games_analyzed": len(games),
            "total_comparisons": total_comparisons,
            "total_clv_positive": len(clv_opportunities),
            "avg_clv_edge_pct": round(mean(all_edges), 3) if all_edges else 0,
            "max_clv_edge_pct": round(max(all_edges), 3) if all_edges else 0,
            "closing_window_minutes": closing_window_minutes,
            "most_exploitable_book": book_profiles[0]["sportsbook"] if book_profiles else None,
            "sharpest_closer": book_profiles[-1]["sportsbook"] if book_profiles else None,
        },
        "methodology": (
            "Closing Line Value (CLV) simulation using last_updated timestamps. "
            "For each game/market/side, the most recently updated sportsbook line is "
            "treated as the 'closing line' (proxy for the final pre-game odds). Every "
            "earlier line is compared: if its implied probability was lower than the "
            "closing line's, it offered positive CLV - the bettor got a better price "
            "than the market settled on.\n\n"
            "Why CLV matters: Consistently beating the closing line is the single best "
            "predictor of long-term betting profitability. Even if individual bets lose, "
            "a bettor who regularly gets +CLV will profit over large sample sizes. "
            "Sharp sportsbooks (like Pinnacle) set efficient closing lines, so beating "
            "them is especially meaningful.\n\n"
            "Per-sportsbook profiles reveal which books are slow to update (exploitable) "
            "vs. which are sharp closers. Books classified as 'EXPLOITABLE' consistently "
            "have stale lines that get beaten by the eventual close - ideal targets for "
            "line-shopping bettors."
        ),
        "context": (
            f"CLV simulation across {len(games)} games: "
            f"{len(clv_opportunities)} positive-CLV opportunities found "
            f"(avg +{round(mean(all_edges), 2) if all_edges else 0}% edge). "
            + (f"Most exploitable book: {book_profiles[0]['sportsbook']} "
               f"(beats close {book_profiles[0]['beat_close_pct']}%). "
               if book_profiles else "")
            + (f"Best CLV: {clv_opportunities[0]['context']}"
               if clv_opportunities else "No CLV opportunities found.")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL — Bookmaker Margin Decomposition (Shin Model)
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def get_shin_fair_odds(
    game_id: str = "",
    market_type: str = "",
    sportsbook: str = "",
    filename: str = "",
) -> str:
    """Decompose bookmaker margins using the Shin model (asymmetric vig allocation).

    Instead of splitting vig equally across both sides (naive proportional method),
    the Shin model allocates MORE vig to the longshot side — which is how sportsbooks
    actually operate.  This produces more accurate "true" probabilities, especially
    for lopsided moneyline markets.

    **Why it matters:**
    - Naive vig removal assumes symmetric overround — it treats -300/+240 the same
      as -110/-110, just scaled.  This under-estimates favourite probability and
      over-estimates longshot probability.
    - The Shin model fixes this by modelling insider-trading risk: bookmakers inflate
      longshot odds more because an insider betting a longshot risks less for a bigger
      payoff.
    - The output includes the Shin parameter *z* (fraction of "informed" money),
      per-side vig allocation, and the delta vs naive method so you can see exactly
      where the naive approach goes wrong.

    Use cases:
    - Compare Shin vs naive fair odds to find where naive vig removal misprices.
    - Identify which side of a market is bearing more vig (longshot bias).
    - Use Shin-adjusted probabilities as a better "true" baseline for +EV detection.
    - Rank sportsbooks by how much insider-trading risk (z) their margins imply.

    Parameters
    ----------
    game_id    : Filter to a specific game (optional).
    market_type: Filter to 'spread', 'moneyline', or 'total' (optional, default all).
    sportsbook : Filter to a specific sportsbook (optional).
    filename   : Data file to use (optional, default latest).
    """
    enriched = _load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data found"})

    if game_id:
        enriched = [r for r in enriched if r.get("game_id") == game_id]
    if sportsbook:
        enriched = [r for r in enriched if r.get("sportsbook", "").lower() == sportsbook.lower()]
    if not enriched:
        return json.dumps({"error": "No matching records found"})

    market_types = [market_type] if market_type else ["spread", "moneyline", "total"]

    results = []
    for r in enriched:
        for mt in market_types:
            m = r.get("markets", {}).get(mt)
            if not m:
                continue

            # Determine side keys based on market type
            if mt in ("spread", "moneyline"):
                odds_a_key, odds_b_key = "home_odds", "away_odds"
                side_a_label, side_b_label = "home", "away"
            else:
                odds_a_key, odds_b_key = "over_odds", "under_odds"
                side_a_label, side_b_label = "over", "under"

            odds_a = m.get(odds_a_key)
            odds_b = m.get(odds_b_key)
            if odds_a is None or odds_b is None:
                continue

            shin = shin_probabilities(odds_a, odds_b)
            naive = no_vig_probabilities(odds_a, odds_b)

            # Determine which side is the longshot
            longshot_side = side_b_label if shin["shin_a"] > shin["shin_b"] else side_a_label
            longshot_vig_share = max(shin["vig_on_a"], shin["vig_on_b"])
            fav_vig_share = min(shin["vig_on_a"], shin["vig_on_b"])
            total_vig = shin["vig_on_a"] + shin["vig_on_b"]
            longshot_vig_pct_of_total = round(longshot_vig_share / total_vig * 100, 1) if total_vig > 0 else 50.0

            entry = {
                "game_id": r.get("game_id"),
                "sport": r.get("sport"),
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "sportsbook": r.get("sportsbook"),
                "market_type": mt,
                f"{side_a_label}_odds": odds_a,
                f"{side_b_label}_odds": odds_b,
                "shin_model": {
                    f"{side_a_label}_true_prob": shin["shin_a"],
                    f"{side_b_label}_true_prob": shin["shin_b"],
                    f"{side_a_label}_true_prob_pct": shin["shin_a_pct"],
                    f"{side_b_label}_true_prob_pct": shin["shin_b_pct"],
                    f"{side_a_label}_fair_odds": fair_odds_to_american(shin["shin_a"]),
                    f"{side_b_label}_fair_odds": fair_odds_to_american(shin["shin_b"]),
                    "z_parameter": shin["z"],
                    "z_pct": shin["z_pct"],
                    "interpretation": (
                        f"Shin z={shin['z_pct']} — the book prices as if {shin['z_pct']} of "
                        f"handle is from informed bettors."
                    ),
                },
                "naive_model": {
                    f"{side_a_label}_true_prob": naive["fair_a"],
                    f"{side_b_label}_true_prob": naive["fair_b"],
                    f"{side_a_label}_true_prob_pct": naive["fair_a_pct"],
                    f"{side_b_label}_true_prob_pct": naive["fair_b_pct"],
                    f"{side_a_label}_fair_odds": fair_odds_to_american(naive["fair_a"]),
                    f"{side_b_label}_fair_odds": fair_odds_to_american(naive["fair_b"]),
                },
                "comparison": {
                    f"{side_a_label}_delta": shin["delta_a"],
                    f"{side_b_label}_delta": shin["delta_b"],
                    f"{side_a_label}_delta_pct": f"{round(shin['delta_a'] * 100, 2)}pp",
                    f"{side_b_label}_delta_pct": f"{round(shin['delta_b'] * 100, 2)}pp",
                    "interpretation": (
                        f"Shin gives {side_a_label} {'+' if shin['delta_a'] >= 0 else ''}"
                        f"{round(shin['delta_a'] * 100, 2)}pp vs naive. "
                        f"Naive OVER-estimates {longshot_side} (longshot) probability."
                    ),
                },
                "vig_decomposition": {
                    "total_vig": shin["vig_pct"],
                    f"vig_on_{side_a_label}": shin["vig_on_a"],
                    f"vig_on_{side_b_label}": shin["vig_on_b"],
                    f"vig_on_{side_a_label}_pct": shin["vig_on_a_pct"],
                    f"vig_on_{side_b_label}_pct": shin["vig_on_b_pct"],
                    "longshot_side": longshot_side,
                    "longshot_vig_share_of_total": f"{longshot_vig_pct_of_total}%",
                    "interpretation": (
                        f"The {longshot_side} (longshot) bears {longshot_vig_pct_of_total}% of "
                        f"total vig vs the naive assumption of 50/50."
                    ),
                },
            }

            # Add line info for spread/total
            if mt == "spread" and "home_line" in m:
                entry["line"] = m["home_line"]
            elif mt == "total" and "line" in m:
                entry["line"] = m["line"]

            results.append(entry)

    # Aggregate: per-sportsbook z rankings (how much insider risk each book prices in)
    book_z_scores: dict[str, list[float]] = {}
    for r in results:
        book = r.get("sportsbook", "unknown")
        z = r["shin_model"]["z_parameter"]
        book_z_scores.setdefault(book, []).append(z)

    book_rankings = []
    for book, zs in book_z_scores.items():
        avg_z = sum(zs) / len(zs)
        book_rankings.append({
            "sportsbook": book,
            "avg_shin_z": round(avg_z, 6),
            "avg_shin_z_pct": f"{round(avg_z * 100, 4)}%",
            "markets_analyzed": len(zs),
            "interpretation": (
                f"{'High' if avg_z > 0.03 else 'Moderate' if avg_z > 0.015 else 'Low'} "
                f"insider-risk pricing — "
                f"{'heavy longshot bias (recreational book)' if avg_z > 0.03 else 'moderate longshot bias' if avg_z > 0.015 else 'relatively symmetric (sharp book)'}"
            ),
        })
    book_rankings.sort(key=lambda b: b["avg_shin_z"], reverse=True)

    result = {
        "markets": results,
        "count": len(results),
        "sportsbook_z_rankings": book_rankings,
        "methodology": (
            "The Shin model (Shin 1991, 1993) decomposes bookmaker margins asymmetrically "
            "by modelling a fraction z of bettors as 'insiders' (informed traders). "
            "Books protect against insiders by inflating odds, but longshots get inflated "
            "MORE because an insider betting a longshot risks less capital for a larger "
            "payout. The parameter z is solved numerically (bisection) so that the "
            "Shin-adjusted true probabilities sum to exactly 1.\n\n"
            "Formula: q_i = (sqrt(z² + 4(1-z)(p_i/S)²) - z) / (2(1-z))\n"
            "where p_i = implied prob, S = sum of implied probs, z = insider fraction.\n\n"
            "Key insight: naive proportional vig removal splits the overround 50/50 between "
            "both sides. Shin allocates more vig to the longshot, giving more accurate true "
            "probabilities — especially important for lopsided moneyline markets.\n\n"
            "The z parameter also serves as a sportsbook 'sharpness' indicator: higher z "
            "means the book prices in more insider risk (typical of recreational books "
            "with wider margins on longshots)."
        ),
        "context": (
            f"Shin margin decomposition for {len(results)} markets across "
            f"{len(book_rankings)} sportsbooks. "
            + (f"Highest insider-risk book: {book_rankings[0]['sportsbook']} "
               f"(avg z={book_rankings[0]['avg_shin_z_pct']}). "
               if book_rankings else "")
            + (f"Sharpest book: {book_rankings[-1]['sportsbook']} "
               f"(avg z={book_rankings[-1]['avg_shin_z_pct']})."
               if book_rankings else "")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def get_odds_elasticity(
    game_id: Optional[str] = None,
    market_type: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Odds Elasticity / Sensitivity Analysis — how much a 0.5-point line
    change impacts odds across sportsbooks.

    For spread and total markets, compares sportsbooks that post different
    lines for the same game and measures how dramatically their odds shift
    per half-point of line movement.  Books with *higher* elasticity (big
    odds swings for small line changes) may be less confident in their
    pricing, while low-elasticity books price through the line change.

    Args:
        game_id:     Filter to a single game. Optional (analyzes all games).
        market_type: "spread" or "total". Optional (analyzes both).
        filename:    Data file to load. Optional.

    Returns per-game elasticity pairs, per-sportsbook average elasticity,
    and a confidence ranking (lower elasticity = more confident pricing).
    """
    cache_key = f"odds_elasticity|{game_id or 'all'}|{market_type or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    enriched = _cache.load_enriched(filename)
    if not enriched:
        return json.dumps({"error": "No data available"})

    by_game = _cache.load_by_game(filename)

    target_markets = []
    if market_type:
        if market_type.lower() not in ("spread", "total"):
            return json.dumps({
                "error": "market_type must be 'spread' or 'total' (moneyline has no line to shift)"
            })
        target_markets = [market_type.lower()]
    else:
        target_markets = ["spread", "total"]

    game_ids = [game_id] if game_id else list(by_game.keys())

    elasticity_pairs: list[dict] = []
    book_elasticities: dict[str, list[float]] = {}

    for gid in game_ids:
        records = by_game.get(gid, [])
        if len(records) < 2:
            continue

        game_meta = {
            "game_id": gid,
            "sport": records[0].get("sport"),
            "home_team": records[0].get("home_team"),
            "away_team": records[0].get("away_team"),
        }

        for mkt in target_markets:
            # Collect (sportsbook, line, home/over odds, away/under odds) tuples
            book_lines: list[dict] = []
            for r in records:
                m = r.get("markets", {}).get(mkt)
                if not m:
                    continue
                if mkt == "spread":
                    line = m.get("home_line")
                    odds_a = m.get("home_odds")
                    odds_b = m.get("away_odds")
                    side_a_label = "home"
                    side_b_label = "away"
                else:  # total
                    line = m.get("line")
                    odds_a = m.get("over_odds")
                    odds_b = m.get("under_odds")
                    side_a_label = "over"
                    side_b_label = "under"

                if line is None or odds_a is None or odds_b is None:
                    continue
                book_lines.append({
                    "sportsbook": r.get("sportsbook"),
                    "line": line,
                    "odds_a": odds_a,
                    "odds_b": odds_b,
                    "side_a_label": side_a_label,
                    "side_b_label": side_b_label,
                })

            if len(book_lines) < 2:
                continue

            # Group by line value
            lines_map: dict[float, list[dict]] = {}
            for bl in book_lines:
                lines_map.setdefault(bl["line"], []).append(bl)

            distinct_lines = sorted(lines_map.keys())
            if len(distinct_lines) < 2:
                continue

            # Compare every pair of distinct lines
            for i in range(len(distinct_lines)):
                for j in range(i + 1, len(distinct_lines)):
                    line_lo = distinct_lines[i]
                    line_hi = distinct_lines[j]
                    line_diff = abs(line_hi - line_lo)
                    if line_diff < 0.25 or line_diff > 3.0:
                        continue  # skip trivially close or too-far lines

                    for bl_lo in lines_map[line_lo]:
                        for bl_hi in lines_map[line_hi]:
                            # Calculate odds movement per 0.5 points
                            odds_a_diff = abs(bl_hi["odds_a"] - bl_lo["odds_a"])
                            odds_b_diff = abs(bl_hi["odds_b"] - bl_lo["odds_b"])
                            avg_odds_diff = (odds_a_diff + odds_b_diff) / 2.0
                            elasticity = round(avg_odds_diff / (line_diff / 0.5), 1)

                            # Implied probability shift
                            prob_a_lo = implied_probability(bl_lo["odds_a"])
                            prob_a_hi = implied_probability(bl_hi["odds_a"])
                            prob_shift = round(abs(prob_a_hi - prob_a_lo) * 100, 2)

                            pair = {
                                **game_meta,
                                "market": mkt,
                                "book_a": bl_lo["sportsbook"],
                                "book_a_line": line_lo,
                                "book_a_odds": f"{bl_lo['odds_a']}/{bl_lo['odds_b']}",
                                "book_b": bl_hi["sportsbook"],
                                "book_b_line": line_hi,
                                "book_b_odds": f"{bl_hi['odds_a']}/{bl_hi['odds_b']}",
                                "line_difference": line_diff,
                                "avg_odds_shift_per_half_point": elasticity,
                                "implied_prob_shift_pct": prob_shift,
                            }
                            elasticity_pairs.append(pair)

                            # Track per-book elasticity
                            book_elasticities.setdefault(bl_lo["sportsbook"], []).append(elasticity)
                            book_elasticities.setdefault(bl_hi["sportsbook"], []).append(elasticity)

    # --- Per-sportsbook summary ---
    book_summary = []
    for book, vals in book_elasticities.items():
        avg_e = round(sum(vals) / len(vals), 1)
        max_e = round(max(vals), 1)
        min_e = round(min(vals), 1)
        std_e = round(pstdev(vals), 1) if len(vals) > 1 else 0.0
        book_summary.append({
            "sportsbook": book,
            "avg_elasticity": avg_e,
            "max_elasticity": max_e,
            "min_elasticity": min_e,
            "std_elasticity": std_e,
            "comparisons": len(vals),
            "confidence_rating": (
                "high" if avg_e < 15 else
                "medium" if avg_e < 30 else
                "low"
            ),
        })
    book_summary.sort(key=lambda b: b["avg_elasticity"])

    # Top pairs by elasticity
    elasticity_pairs.sort(key=lambda p: p["avg_odds_shift_per_half_point"], reverse=True)

    # Overall statistics
    all_elasticities = [p["avg_odds_shift_per_half_point"] for p in elasticity_pairs]
    overall_stats = {}
    if all_elasticities:
        overall_stats = {
            "mean_elasticity": round(mean(all_elasticities), 1),
            "median_elasticity": round(sorted(all_elasticities)[len(all_elasticities) // 2], 1),
            "max_elasticity": round(max(all_elasticities), 1),
            "min_elasticity": round(min(all_elasticities), 1),
            "std_elasticity": round(pstdev(all_elasticities), 1),
        }

    result = {
        "elasticity_pairs": elasticity_pairs[:50],  # top 50 most elastic
        "pair_count": len(elasticity_pairs),
        "sportsbook_elasticity_rankings": book_summary,
        "overall_statistics": overall_stats,
        "parameters": {
            "game_id": game_id or "all",
            "market_type": market_type or "all (spread + total)",
        },
        "methodology": (
            "Odds elasticity measures how much the odds change per 0.5-point "
            "line movement. For each game and market (spread/total), we compare "
            "sportsbooks posting different lines and calculate:\n"
            "  elasticity = avg_odds_change / (line_difference / 0.5)\n\n"
            "High elasticity (big odds swings for small line shifts) suggests a "
            "book is less confident in its pricing — they compensate for moving "
            "off the key number with dramatic odds adjustments. Low elasticity "
            "books price through the line change smoothly, indicating stronger "
            "underlying models.\n\n"
            "Confidence ratings: high (<15), medium (15-30), low (>30)."
        ),
        "context": (
            f"Analyzed {len(elasticity_pairs)} cross-book line comparisons. "
            + (f"Most elastic: {elasticity_pairs[0]['book_a']} vs {elasticity_pairs[0]['book_b']} "
               f"on {elasticity_pairs[0]['game_id']} {elasticity_pairs[0]['market']} "
               f"(elasticity={elasticity_pairs[0]['avg_odds_shift_per_half_point']}). "
               if elasticity_pairs else "No line differences found across books. ")
            + (f"Most confident book: {book_summary[0]['sportsbook']} "
               f"(avg elasticity={book_summary[0]['avg_elasticity']}). "
               f"Least confident: {book_summary[-1]['sportsbook']} "
               f"(avg elasticity={book_summary[-1]['avg_elasticity']})."
               if book_summary else "")
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)




# ---------------------------------------------------------------------------
# Poisson helpers
# ---------------------------------------------------------------------------

def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function computed in log-space for numerical stability."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(k * log(lam) - lam - lgamma(k + 1))


def _build_score_matrix(home_lambda: float, away_lambda: float, max_score: int):
    """Build a 2D probability matrix P[home_score][away_score].

    Returns:
        matrix: list[list[float]]  -- matrix[h][a] = P(home=h AND away=a)
        home_pmf: list[float]      -- marginal P(home=h)
        away_pmf: list[float]      -- marginal P(away=a)
    """
    home_pmf = [_poisson_pmf(k, home_lambda) for k in range(max_score + 1)]
    away_pmf = [_poisson_pmf(k, away_lambda) for k in range(max_score + 1)]
    matrix = [
        [home_pmf[h] * away_pmf[a] for a in range(max_score + 1)]
        for h in range(max_score + 1)
    ]
    return matrix, home_pmf, away_pmf


@mcp.tool()
def get_poisson_score_predictions(
    game_id: Optional[str] = None,
    max_score: Optional[int] = None,
    top_n_scores: int = 15,
    alt_spreads: Optional[str] = None,
    alt_totals: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Poisson model for score prediction -- derive every possible final-score probability
    from the consensus spread and total, then price alternative lines, props, and key numbers.

    Uses the implied total and spread to compute expected points per team (lambda),
    then applies a Poisson distribution to build a full score-probability matrix.

    This lets you:
    - See the most likely exact final scores
    - Price alternative spreads (e.g., what is the fair price on -3.5 vs -7.5?)
    - Price alternative totals (e.g., over/under 215.5, 225.5, etc.)
    - Get win/loss/draw probabilities
    - Identify key numbers (margins and totals with outsized probability mass)
    - Estimate half-time and quarter score distributions

    Args:
        game_id: Filter to a single game. Optional -- returns all games if omitted.
        max_score: Maximum score to model per team. Auto-detected per sport if omitted
                   (NBA: 160, NFL: 60, NHL/Soccer: 10, default: 80).
        top_n_scores: How many top exact scores to return per game (default 15).
        alt_spreads: Comma-separated alternative spreads to price, e.g. "-3.5,-7.5,-10.5,1.5".
                     Defaults to a sport-appropriate set if omitted.
        alt_totals: Comma-separated alternative totals to price, e.g. "210.5,215.5,220.5,225.5".
                    Auto-generated around the consensus total if omitted.
        filename: Data file to load. Optional.

    Returns Poisson-modeled score predictions per game with exact scores, alt lines, and key numbers.
    """
    cache_key = (
        f"poisson:{game_id or 'all'}:{max_score}:{top_n_scores}:"
        f"{alt_spreads or ''}:{alt_totals or ''}"
    )
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    games = _cache.load_by_game(filename)
    consensus = _cache.load_consensus(filename)

    # Parse user-supplied alternative lines -----------------------------------
    user_alt_spreads = None
    if alt_spreads:
        try:
            user_alt_spreads = [float(x.strip()) for x in alt_spreads.split(",")]
        except ValueError:
            pass

    user_alt_totals = None
    if alt_totals:
        try:
            user_alt_totals = [float(x.strip()) for x in alt_totals.split(",")]
        except ValueError:
            pass

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
        sport = (first.get("sport") or "").upper()

        # Expected points (lambda) per team
        home_lambda = (avg_total + avg_home_line) / 2
        away_lambda = (avg_total - avg_home_line) / 2

        if home_lambda <= 0 or away_lambda <= 0:
            continue

        # Sport-adaptive max score
        if max_score is not None:
            ms = max_score
        elif sport in ("NBA", "WNBA", "CBB", "NCAAB"):
            ms = 160
        elif sport in ("NFL", "NCAAF", "CFB"):
            ms = 70
        elif sport in ("NHL",):
            ms = 12
        elif sport in ("SOCCER", "MLS", "EPL", "UEFA", "FIFA"):
            ms = 10
        elif sport in ("MLB",):
            ms = 20
        else:
            ms = max(int(max(home_lambda, away_lambda) * 2.5), 30)

        # Build Poisson matrix -------------------------------------------------
        matrix, home_pmf, away_pmf = _build_score_matrix(home_lambda, away_lambda, ms)

        # Win / loss / draw probabilities
        home_win_prob = 0.0
        away_win_prob = 0.0
        draw_prob = 0.0
        for h in range(ms + 1):
            for a in range(ms + 1):
                p = matrix[h][a]
                if h > a:
                    home_win_prob += p
                elif a > h:
                    away_win_prob += p
                else:
                    draw_prob += p

        # Top exact scores -----------------------------------------------------
        score_probs = []
        for h in range(ms + 1):
            for a in range(ms + 1):
                p = matrix[h][a]
                if p > 1e-6:
                    score_probs.append({"home": h, "away": a, "prob": p})
        score_probs.sort(key=lambda s: s["prob"], reverse=True)
        top_scores = score_probs[:top_n_scores]
        for s in top_scores:
            s["prob_pct"] = f"{s['prob'] * 100:.2f}%"
            s["score"] = f"{home_team} {s['home']} - {away_team} {s['away']}"

        # Alternative spread pricing -------------------------------------------
        if user_alt_spreads is not None:
            spreads_to_price = user_alt_spreads
        elif sport in ("NBA", "WNBA", "CBB", "NCAAB"):
            base = round(avg_home_line)
            spreads_to_price = [base + d for d in [-6, -4, -2, -0.5, 0.5, 2, 4, 6]]
        elif sport in ("NFL", "NCAAF", "CFB"):
            spreads_to_price = [-14.5, -10.5, -7.5, -6.5, -3.5, -2.5, -1.5,
                                1.5, 2.5, 3.5, 6.5, 7.5, 10.5, 14.5]
        elif sport in ("NHL", "SOCCER", "MLS", "EPL", "UEFA", "FIFA"):
            spreads_to_price = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        else:
            base = round(avg_home_line)
            spreads_to_price = [base + d for d in [-4, -2, -0.5, 0.5, 2, 4]]

        alt_spread_results = []
        for sp in sorted(spreads_to_price):
            # Home covers when home_score + spread > away_score
            cover_prob = 0.0
            push_prob = 0.0
            for h in range(ms + 1):
                for a in range(ms + 1):
                    margin = h - a
                    p = matrix[h][a]
                    if margin + sp > 0:
                        cover_prob += p
                    elif margin + sp == 0:
                        push_prob += p

            no_push = 1 - push_prob if push_prob < 1 else 1
            home_cover_adj = cover_prob / no_push if no_push > 0 else 0.5
            away_cover_adj = 1 - home_cover_adj

            alt_spread_results.append({
                "spread": sp,
                "home_cover_prob": round(cover_prob, 4),
                "home_cover_prob_pct": f"{cover_prob * 100:.1f}%",
                "away_cover_prob": round(1 - cover_prob - push_prob, 4),
                "push_prob": round(push_prob, 4),
                "home_fair_odds": fair_odds_to_american(home_cover_adj) if 0.01 < home_cover_adj < 0.99 else None,
                "away_fair_odds": fair_odds_to_american(away_cover_adj) if 0.01 < away_cover_adj < 0.99 else None,
                "is_consensus": abs(sp - avg_home_line) < 0.25,
            })

        # Alternative total pricing --------------------------------------------
        if user_alt_totals is not None:
            totals_to_price = user_alt_totals
        else:
            base_total = round(avg_total)
            if sport in ("NBA", "WNBA", "CBB", "NCAAB"):
                offsets = [-15, -10, -7, -5, -3, -0.5, 0.5, 3, 5, 7, 10, 15]
            elif sport in ("NFL", "NCAAF", "CFB"):
                offsets = [-10, -7, -3.5, -3, -0.5, 0.5, 3, 3.5, 7, 10]
            elif sport in ("NHL", "SOCCER", "MLS", "EPL", "MLB"):
                offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
            else:
                offsets = [-8, -5, -3, -0.5, 0.5, 3, 5, 8]
            totals_to_price = [base_total + o for o in offsets]

        alt_total_results = []
        for t in sorted(totals_to_price):
            over_prob = 0.0
            push_prob = 0.0
            for h in range(ms + 1):
                for a in range(ms + 1):
                    total_score = h + a
                    p = matrix[h][a]
                    if total_score > t:
                        over_prob += p
                    elif total_score == t:
                        push_prob += p

            under_prob = 1 - over_prob - push_prob
            no_push = 1 - push_prob if push_prob < 1 else 1
            over_adj = over_prob / no_push if no_push > 0 else 0.5
            under_adj = 1 - over_adj

            alt_total_results.append({
                "total_line": t,
                "over_prob": round(over_prob, 4),
                "over_prob_pct": f"{over_prob * 100:.1f}%",
                "under_prob": round(under_prob, 4),
                "push_prob": round(push_prob, 4),
                "over_fair_odds": fair_odds_to_american(over_adj) if 0.01 < over_adj < 0.99 else None,
                "under_fair_odds": fair_odds_to_american(under_adj) if 0.01 < under_adj < 0.99 else None,
                "is_consensus": abs(t - avg_total) < 0.25,
            })

        # Key numbers -- margins and totals with outsized probability mass -----
        margin_probs: dict[int, float] = {}
        total_probs: dict[int, float] = {}
        for h in range(ms + 1):
            for a in range(ms + 1):
                p = matrix[h][a]
                if p < 1e-9:
                    continue
                m = h - a
                margin_probs[m] = margin_probs.get(m, 0) + p
                t = h + a
                total_probs[t] = total_probs.get(t, 0) + p

        key_margins = sorted(margin_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        key_totals = sorted(total_probs.items(), key=lambda x: x[1], reverse=True)[:10]

        # Half / quarter estimates (simple Poisson scaling) --------------------
        half_home = home_lambda / 2
        half_away = away_lambda / 2
        quarter_home = home_lambda / 4
        quarter_away = away_lambda / 4

        half_matrix, _, _ = _build_score_matrix(half_home, half_away, ms // 2)
        half_scores = []
        for h in range(ms // 2 + 1):
            for a in range(ms // 2 + 1):
                p = half_matrix[h][a]
                if p > 1e-5:
                    half_scores.append({"home": h, "away": a, "prob": p})
        half_scores.sort(key=lambda s: s["prob"], reverse=True)
        top_half_scores = half_scores[:5]
        for s in top_half_scores:
            s["prob_pct"] = f"{s['prob'] * 100:.2f}%"
            s["score"] = f"{home_team} {s['home']} - {away_team} {s['away']}"

        context_str = (
            f"Poisson model for {home_team} vs {away_team}: "
            f"lambda_home={round(home_lambda, 1)}, lambda_away={round(away_lambda, 1)} "
            f"(spread {avg_home_line}, total {avg_total}). "
            f"Home win {home_win_prob * 100:.1f}%, Away win {away_win_prob * 100:.1f}%, "
            f"Draw {draw_prob * 100:.1f}%."
        )
        if top_scores:
            context_str += f" Most likely score: {top_scores[0]['score']} ({top_scores[0]['prob_pct']})."

        results.append({
            "game_id": gid,
            "sport": sport,
            "home_team": home_team,
            "away_team": away_team,
            "consensus_spread": avg_home_line,
            "consensus_total": avg_total,
            "model_params": {
                "home_lambda": round(home_lambda, 2),
                "away_lambda": round(away_lambda, 2),
                "max_score_modeled": ms,
            },
            "win_probabilities": {
                "home_win": round(home_win_prob, 4),
                "home_win_pct": f"{home_win_prob * 100:.1f}%",
                "away_win": round(away_win_prob, 4),
                "away_win_pct": f"{away_win_prob * 100:.1f}%",
                "draw": round(draw_prob, 4),
                "draw_pct": f"{draw_prob * 100:.1f}%",
                "home_ml_fair_odds": fair_odds_to_american(home_win_prob) if 0.01 < home_win_prob < 0.99 else None,
                "away_ml_fair_odds": fair_odds_to_american(away_win_prob) if 0.01 < away_win_prob < 0.99 else None,
            },
            "top_exact_scores": top_scores,
            "alternative_spreads": alt_spread_results,
            "alternative_totals": alt_total_results,
            "key_numbers": {
                "margins": [
                    {"margin": m, "prob": round(p, 4), "prob_pct": f"{p * 100:.2f}%",
                     "label": f"{home_team} by {abs(m)}" if m > 0 else (f"{away_team} by {abs(m)}" if m < 0 else "Tie")}
                    for m, p in key_margins
                ],
                "totals": [
                    {"total": t, "prob": round(p, 4), "prob_pct": f"{p * 100:.2f}%"}
                    for t, p in key_totals
                ],
            },
            "half_time_estimates": {
                "home_expected_half": round(half_home, 2),
                "away_expected_half": round(half_away, 2),
                "expected_half_total": round(half_home + half_away, 2),
                "top_half_scores": top_half_scores,
                "note": "Assumes scoring is uniformly distributed across halves (simple 50/50 split of full-game lambda).",
            },
            "quarter_estimates": {
                "home_expected_quarter": round(quarter_home, 2),
                "away_expected_quarter": round(quarter_away, 2),
                "expected_quarter_total": round(quarter_home + quarter_away, 2),
                "note": "Assumes scoring is uniformly distributed across quarters (25% split of full-game lambda).",
            },
            "context": context_str,
        })

    results.sort(key=lambda r: r["consensus_total"], reverse=True)

    overall_context = f"Poisson score predictions for {len(results)} game(s)."
    if results:
        overall_context += (
            f" Highest-scoring game expected: "
            f"{results[0]['home_team']} vs {results[0]['away_team']} "
            f"(total {results[0]['consensus_total']})."
        )

    result = {
        "poisson_predictions": results,
        "count": len(results),
        "methodology": (
            "Poisson Score Prediction Model\n"
            "================================\n"
            "1. Derive expected points per team from consensus spread & total:\n"
            "   home_lambda = (total + home_spread) / 2\n"
            "   away_lambda = (total - home_spread) / 2\n"
            "2. Apply independent Poisson distributions: P(score=k) = (lambda^k * e^-lambda) / k!\n"
            "3. Build joint score matrix: P(H=h, A=a) = P_home(h) * P_away(a)\n"
            "4. From the matrix derive:\n"
            "   - Exact score probabilities (most likely final scores)\n"
            "   - Win/loss/draw probabilities\n"
            "   - Alternative spread pricing (fair odds for any spread)\n"
            "   - Alternative total pricing (fair over/under for any total)\n"
            "   - Key numbers (margins & totals with outsized probability mass)\n"
            "   - Half/quarter estimates (lambda scaled proportionally)\n\n"
            "Assumptions: Team scores are independent Poisson random variables. "
            "This works best for lower-scoring sports (soccer, hockey, baseball). "
            "For high-scoring sports (NBA, NFL), the model still provides useful "
            "relative pricing but the normal approximation is more accurate for "
            "tail probabilities."
        ),
        "context": overall_context,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _cache.set_analysis(filename, cache_key, result)
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# TOOL — Information Flow Analysis via Timestamps
# ═══════════════════════════════════════════════════════════════════════════


def _compute_information_flow(
    by_game: dict[str, list[dict]],
) -> dict:
    """Analyze last_updated timestamps across sportsbooks to infer information
    flow — which books move first (market leaders) and which follow (laggers).

    Returns per-game update order, aggregate leader scores, and exploitable
    windows where slow followers haven't caught up.
    """
    # --- Per-game analysis ---------------------------------------------------
    game_analyses = {}

    for game_id, records in by_game.items():
        # Parse timestamps for each book
        book_times: list[dict] = []
        for r in records:
            ts = _parse_ts_safe(r.get("last_updated"))
            if ts is None:
                continue
            book_times.append({
                "sportsbook": r.get("sportsbook", "Unknown"),
                "ts": ts,
                "record": r,
            })

        if len(book_times) < 2:
            continue

        # Sort by timestamp ascending (earliest update first = leader)
        book_times.sort(key=lambda x: x["ts"])

        earliest_ts = book_times[0]["ts"]
        latest_ts = book_times[-1]["ts"]
        total_span_seconds = (latest_ts - earliest_ts).total_seconds()
        total_span_minutes = total_span_seconds / 60

        update_order = []
        for rank, bt in enumerate(book_times, start=1):
            lag_seconds = (bt["ts"] - earliest_ts).total_seconds()
            lag_minutes = lag_seconds / 60
            update_order.append({
                "rank": rank,
                "sportsbook": bt["sportsbook"],
                "last_updated": bt["ts"].isoformat(),
                "lag_from_leader_seconds": round(lag_seconds, 1),
                "lag_from_leader_minutes": round(lag_minutes, 1),
            })

        leader = book_times[0]["sportsbook"]
        laggards = [bt["sportsbook"] for bt in book_times if (bt["ts"] - earliest_ts).total_seconds() > 60]

        game_analyses[game_id] = {
            "game_id": game_id,
            "sport": records[0].get("sport"),
            "home_team": records[0].get("home_team"),
            "away_team": records[0].get("away_team"),
            "leader": leader,
            "leader_updated_at": earliest_ts.isoformat(),
            "total_span_minutes": round(total_span_minutes, 1),
            "books_analyzed": len(book_times),
            "update_order": update_order,
            "laggards": laggards,
        }

    # --- Aggregate: score each book across all games -------------------------
    book_stats: dict[str, dict] = {}

    for ga in game_analyses.values():
        for entry in ga["update_order"]:
            book = entry["sportsbook"]
            if book not in book_stats:
                book_stats[book] = {
                    "first_count": 0,
                    "last_count": 0,
                    "total_games": 0,
                    "sum_rank": 0,
                    "lags_seconds": [],
                }
            stats = book_stats[book]
            stats["total_games"] += 1
            stats["sum_rank"] += entry["rank"]
            stats["lags_seconds"].append(entry["lag_from_leader_seconds"])
            if entry["rank"] == 1:
                stats["first_count"] += 1
            if entry["rank"] == ga["books_analyzed"]:
                stats["last_count"] += 1

    # Build ranked sportsbook list
    book_rankings = []
    for book, stats in book_stats.items():
        n = stats["total_games"]
        avg_rank = stats["sum_rank"] / n if n else 0
        avg_lag = sum(stats["lags_seconds"]) / n if n else 0
        max_lag = max(stats["lags_seconds"]) if stats["lags_seconds"] else 0
        leader_pct = (stats["first_count"] / n * 100) if n else 0
        laggard_pct = (stats["last_count"] / n * 100) if n else 0

        # Leader score: 0-100 where 100 = always first with no lag
        # Weighted: 60% leader frequency, 40% inverse average rank
        max_possible_rank = max(ga["books_analyzed"] for ga in game_analyses.values()) if game_analyses else 1
        rank_component = max(0, (1 - (avg_rank - 1) / max(max_possible_rank - 1, 1))) * 100
        leader_score = round(leader_pct * 0.6 + rank_component * 0.4, 1)

        book_rankings.append({
            "sportsbook": book,
            "leader_score": leader_score,
            "leader_pct": round(leader_pct, 1),
            "laggard_pct": round(laggard_pct, 1),
            "avg_rank": round(avg_rank, 2),
            "avg_lag_seconds": round(avg_lag, 1),
            "avg_lag_minutes": round(avg_lag / 60, 1),
            "max_lag_seconds": round(max_lag, 1),
            "max_lag_minutes": round(max_lag / 60, 1),
            "first_count": stats["first_count"],
            "last_count": stats["last_count"],
            "games_analyzed": n,
        })

    book_rankings.sort(key=lambda x: x["leader_score"], reverse=True)

    # --- Exploitable windows: games where laggards differ significantly ------
    exploitable_windows = []
    for ga in game_analyses.values():
        if ga["total_span_minutes"] < 1:
            continue  # All books roughly in sync — nothing to exploit

        game_records = {r.get("sportsbook"): r for r in by_game[ga["game_id"]]}
        leader_record = game_records.get(ga["leader"])
        if not leader_record:
            continue

        for laggard in ga["laggards"]:
            laggard_record = game_records.get(laggard)
            if not laggard_record:
                continue

            lag_entry = next(
                (e for e in ga["update_order"] if e["sportsbook"] == laggard), None
            )
            lag_minutes = lag_entry["lag_from_leader_minutes"] if lag_entry else 0

            # Compare markets to detect stale pricing
            diffs = []
            for mkt_type in ("spread", "moneyline", "total"):
                leader_mkt = leader_record.get("markets", {}).get(mkt_type, {})
                laggard_mkt = laggard_record.get("markets", {}).get(mkt_type, {})
                if not leader_mkt or not laggard_mkt:
                    continue

                if mkt_type in ("spread", "moneyline"):
                    keys = ["home_odds", "away_odds"]
                    line_key = "spread" if mkt_type == "spread" else None
                else:
                    keys = ["over_odds", "under_odds"]
                    line_key = "total"

                for k in keys:
                    l_val = leader_mkt.get(k)
                    g_val = laggard_mkt.get(k)
                    if l_val is not None and g_val is not None:
                        l_prob = implied_probability(l_val)
                        g_prob = implied_probability(g_val)
                        diff_pct = abs(l_prob - g_prob) * 100
                        if diff_pct > 1.0:  # >1% implied prob difference
                            diffs.append({
                                "market": mkt_type,
                                "odds_field": k,
                                "leader_odds": l_val,
                                "laggard_odds": g_val,
                                "leader_implied_prob": round(l_prob * 100, 2),
                                "laggard_implied_prob": round(g_prob * 100, 2),
                                "implied_prob_diff_pct": round(diff_pct, 2),
                            })

                # Check line differences (spread/total line value)
                if line_key:
                    l_line = leader_mkt.get(line_key)
                    g_line = laggard_mkt.get(line_key)
                    if l_line is not None and g_line is not None and l_line != g_line:
                        diffs.append({
                            "market": mkt_type,
                            "field": f"{line_key}_line",
                            "leader_line": l_line,
                            "laggard_line": g_line,
                            "line_diff": round(abs(l_line - g_line), 1),
                        })

            if diffs:
                exploitable_windows.append({
                    "game_id": ga["game_id"],
                    "sport": ga.get("sport"),
                    "home_team": ga.get("home_team"),
                    "away_team": ga.get("away_team"),
                    "leader": ga["leader"],
                    "laggard": laggard,
                    "lag_minutes": lag_minutes,
                    "pricing_differences": diffs,
                    "context": (
                        f"{laggard} is {lag_minutes:.1f} min behind {ga['leader']} "
                        f"for {ga.get('away_team')} @ {ga.get('home_team')} — "
                        f"{len(diffs)} market(s) show stale pricing that may be exploitable."
                    ),
                })

    exploitable_windows.sort(key=lambda w: w["lag_minutes"], reverse=True)

    return {
        "game_analyses": game_analyses,
        "book_rankings": book_rankings,
        "exploitable_windows": exploitable_windows,
    }


@mcp.tool()
def get_information_flow(
    game_id: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """Map information flow across sportsbooks using last_updated timestamps.

    Determines which sportsbook updates first (market leader / sharpest) and
    which follow with a delay (laggards).  The leader is typically the sharpest
    book — its line moves first in response to new information.  Laggards that
    haven't yet caught up create exploitable windows where their stale odds
    may offer value.

    Analysis includes:
    - **Update order per game**: Chronological ranking of when each book last
      refreshed its odds, with lag from the leader in seconds/minutes.
    - **Aggregate leader scores**: Across all games, which books move first most
      often (0-100 composite score; 100 = always first).
    - **Exploitable windows**: Games where a laggard's odds differ meaningfully
      from the leader's — indicating stale pricing ripe for +EV bets.

    Use this to:
    1. Identify the sharpest book(s) in the current data snapshot.
    2. Find slow-to-update books whose stale lines can be beaten.
    3. Quantify how large the delay windows are in minutes/seconds.

    Args:
        game_id: Optional — limit analysis to a specific game.
        filename: Data file to load. Optional.

    Returns update order per game, aggregate sportsbook leader rankings,
    and exploitable laggard windows with pricing differences.
    """
    cache_key = f"info_flow:{game_id or 'all'}"
    cached = _cache.get_analysis(filename, cache_key)
    if cached is not None:
        return json.dumps(cached, indent=2)

    odds = _load_odds(filename)
    by_game = _cache.load_by_game(filename)

    if game_id:
        by_game = {k: v for k, v in by_game.items() if k == game_id}

    flow = _compute_information_flow(by_game)

    # Build per-game list for output
    games_list = list(flow["game_analyses"].values())
    games_list.sort(key=lambda g: g["total_span_minutes"], reverse=True)

    result = {
        "information_flow": {
            "games": games_list,
            "games_analyzed": len(games_list),
        },
        "sportsbook_leader_rankings": flow["book_rankings"],
        "exploitable_windows": flow["exploitable_windows"],
        "exploitable_count": len(flow["exploitable_windows"]),
        "methodology": (
            "Timestamps (last_updated) from each sportsbook are compared per game "
            "to establish a chronological update order.  The book that updates first "
            "is the market leader — typically the sharpest, as sharp action drives "
            "price discovery.  Books that update later are followers/laggards.  "
            "When a laggard's odds diverge from the leader's by >1% implied "
            "probability, that gap is flagged as an exploitable window — the "
            "laggard hasn't yet adjusted to the leader's new information.  "
            "Leader scores (0-100) aggregate each book's first-mover frequency "
            "and average rank across all games."
        ),
        "context": (
            f"Analyzed {len(games_list)} game(s) across {len(flow['book_rankings'])} sportsbooks. "
            + (
                f"Top leader: {flow['book_rankings'][0]['sportsbook']} "
                f"(score {flow['book_rankings'][0]['leader_score']}, "
                f"first in {flow['book_rankings'][0]['leader_pct']}% of games). "
                if flow["book_rankings"]
                else "No sportsbook data available. "
            )
            + f"Found {len(flow['exploitable_windows'])} exploitable laggard window(s)."
            + (
                f" Largest gap: {flow['exploitable_windows'][0]['context']}"
                if flow["exploitable_windows"]
                else ""
            )
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
