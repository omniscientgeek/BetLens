"""
Detection phase – compute implied probabilities, vig, fair odds, and stale
line detection for every record in a sample odds file.

This module is called by the processing pipeline in app.py during the
"detect" phase.
"""

import json
import os
import aiofiles
from collections import defaultdict
from datetime import datetime
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, fair_odds_to_american, expected_value

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _enrich_market_spread(market: dict) -> dict:
    """Add probability / vig / fair-odds columns to a spread market."""
    home_odds = market["home_odds"]
    away_odds = market["away_odds"]

    vig_info = calculate_vig(home_odds, away_odds)
    fair = no_vig_probabilities(home_odds, away_odds)

    return {
        **market,
        "home_implied_prob": vig_info["implied_a"],
        "away_implied_prob": vig_info["implied_b"],
        "vig": vig_info["vig"],
        "vig_pct": vig_info["vig_pct"],
        "home_fair_prob": fair["fair_a"],
        "away_fair_prob": fair["fair_b"],
        "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
        "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
    }


def _enrich_market_moneyline(market: dict) -> dict:
    """Add probability / vig / fair-odds columns to a moneyline market."""
    home_odds = market["home_odds"]
    away_odds = market["away_odds"]

    vig_info = calculate_vig(home_odds, away_odds)
    fair = no_vig_probabilities(home_odds, away_odds)

    return {
        **market,
        "home_implied_prob": vig_info["implied_a"],
        "away_implied_prob": vig_info["implied_b"],
        "vig": vig_info["vig"],
        "vig_pct": vig_info["vig_pct"],
        "home_fair_prob": fair["fair_a"],
        "away_fair_prob": fair["fair_b"],
        "home_fair_odds": fair_odds_to_american(fair["fair_a"]),
        "away_fair_odds": fair_odds_to_american(fair["fair_b"]),
    }


def _enrich_market_total(market: dict) -> dict:
    """Add probability / vig / fair-odds columns to a totals (over/under) market."""
    over_odds = market["over_odds"]
    under_odds = market["under_odds"]

    vig_info = calculate_vig(over_odds, under_odds)
    fair = no_vig_probabilities(over_odds, under_odds)

    return {
        **market,
        "over_implied_prob": vig_info["implied_a"],
        "under_implied_prob": vig_info["implied_b"],
        "vig": vig_info["vig"],
        "vig_pct": vig_info["vig_pct"],
        "over_fair_prob": fair["fair_a"],
        "under_fair_prob": fair["fair_b"],
        "over_fair_odds": fair_odds_to_american(fair["fair_a"]),
        "under_fair_odds": fair_odds_to_american(fair["fair_b"]),
    }


async def run_detection(filename: str) -> dict:
    """Run the detection phase on a data file.

    For every odds record, enriches each market (spread, moneyline, total)
    with:
      - implied probability for each side
      - vig (vigorish / overround)
      - no-vig fair probability and fair American odds
      - expected value (EV) edge vs consensus fair probability

    Also computes cross-sportsbook summaries.

    Returns
    -------
    dict with:
        enriched_odds  – list of odds records with computed fields added
        vig_summary    – per-game, per-market vig comparison across books
        ev_summary     – per-game, per-market +EV opportunities (sorted by edge)
        consensus      – consensus fair probabilities per game/market
    """
    filepath = os.path.join(DATA_DIR, filename)
    async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
        content = await f.read()
    data = json.loads(content)

    odds_list = data.get("odds", [])
    enriched = []

    for record in odds_list:
        markets = record.get("markets", {})
        enriched_markets = {}

        if "spread" in markets:
            enriched_markets["spread"] = _enrich_market_spread(markets["spread"])

        if "moneyline" in markets:
            enriched_markets["moneyline"] = _enrich_market_moneyline(markets["moneyline"])

        if "total" in markets:
            enriched_markets["total"] = _enrich_market_total(markets["total"])

        enriched.append({
            **{k: v for k, v in record.items() if k != "markets"},
            "markets": enriched_markets,
        })

    # ---------- Consensus fair probabilities per game/market/side ----------
    consensus = _compute_consensus(enriched)

    # ---------- Enrich each record with EV fields ----------
    for record in enriched:
        game_id = record["game_id"]
        markets = record.get("markets", {})

        for market_name in ("spread", "moneyline", "total"):
            market = markets.get(market_name)
            if not market:
                continue

            game_consensus = consensus.get(game_id, {}).get(market_name)
            if not game_consensus:
                continue

            if market_name in ("spread", "moneyline"):
                home_odds = market.get("home_odds")
                away_odds = market.get("away_odds")
                if home_odds is not None and "home_fair_prob" in game_consensus:
                    ev_home = expected_value(home_odds, game_consensus["home_fair_prob"])
                    market["home_ev"] = ev_home["ev_edge"]
                    market["home_ev_pct"] = ev_home["ev_edge_pct"]
                    market["home_ev_dollar"] = ev_home["ev_dollar"]
                    market["home_is_positive_ev"] = ev_home["is_positive_ev"]
                if away_odds is not None and "away_fair_prob" in game_consensus:
                    ev_away = expected_value(away_odds, game_consensus["away_fair_prob"])
                    market["away_ev"] = ev_away["ev_edge"]
                    market["away_ev_pct"] = ev_away["ev_edge_pct"]
                    market["away_ev_dollar"] = ev_away["ev_dollar"]
                    market["away_is_positive_ev"] = ev_away["is_positive_ev"]
            else:  # total
                over_odds = market.get("over_odds")
                under_odds = market.get("under_odds")
                if over_odds is not None and "over_fair_prob" in game_consensus:
                    ev_over = expected_value(over_odds, game_consensus["over_fair_prob"])
                    market["over_ev"] = ev_over["ev_edge"]
                    market["over_ev_pct"] = ev_over["ev_edge_pct"]
                    market["over_ev_dollar"] = ev_over["ev_dollar"]
                    market["over_is_positive_ev"] = ev_over["is_positive_ev"]
                if under_odds is not None and "under_fair_prob" in game_consensus:
                    ev_under = expected_value(under_odds, game_consensus["under_fair_prob"])
                    market["under_ev"] = ev_under["ev_edge"]
                    market["under_ev_pct"] = ev_under["ev_edge_pct"]
                    market["under_ev_dollar"] = ev_under["ev_dollar"]
                    market["under_is_positive_ev"] = ev_under["is_positive_ev"]

    # ---------- Stale line detection ----------
    stale_summary = _detect_stale_lines(enriched)

    # Stamp each enriched record with staleness info
    _enrich_staleness(enriched, stale_summary)

    # Build a vig summary: game_id → market → list of {sportsbook, vig, vig_pct}
    vig_summary = _build_vig_summary(enriched)

    # Build an EV summary: game_id → market → list of +EV opportunities sorted by edge
    ev_summary = _build_ev_summary(enriched)

    return {
        "enriched_odds": enriched,
        "vig_summary": vig_summary,
        "ev_summary": ev_summary,
        "consensus": consensus,
        "stale_summary": stale_summary,
    }


def _compute_consensus(enriched_odds: list) -> dict:
    """Compute consensus fair probabilities across all sportsbooks per game/market.

    For each game and market type, averages the no-vig fair probabilities
    from every sportsbook to establish a "true" consensus baseline.

    Returns
    -------
    dict structured as:
        {
            "game_id": {
                "spread": {"home_fair_prob": 0.55, "away_fair_prob": 0.45, "book_count": 8},
                "moneyline": {"home_fair_prob": ..., "away_fair_prob": ..., "book_count": ...},
                "total": {"over_fair_prob": ..., "under_fair_prob": ..., "book_count": ...},
            },
            ...
        }
    """
    # Accumulate fair probs per game/market/side
    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for record in enriched_odds:
        game_id = record["game_id"]
        markets = record.get("markets", {})

        for market_name in ("spread", "moneyline"):
            market = markets.get(market_name)
            if not market:
                continue
            if "home_fair_prob" in market:
                accum[game_id][market_name]["home"].append(market["home_fair_prob"])
            if "away_fair_prob" in market:
                accum[game_id][market_name]["away"].append(market["away_fair_prob"])

        total_market = markets.get("total")
        if total_market:
            if "over_fair_prob" in total_market:
                accum[game_id]["total"]["over"].append(total_market["over_fair_prob"])
            if "under_fair_prob" in total_market:
                accum[game_id]["total"]["under"].append(total_market["under_fair_prob"])

    # Average them out
    consensus = {}
    for game_id, market_map in accum.items():
        consensus[game_id] = {}
        for market_name, sides in market_map.items():
            entry = {"book_count": 0}
            if market_name in ("spread", "moneyline"):
                home_probs = sides.get("home", [])
                away_probs = sides.get("away", [])
                count = max(len(home_probs), len(away_probs))
                entry["book_count"] = count
                if home_probs:
                    entry["home_fair_prob"] = round(sum(home_probs) / len(home_probs), 6)
                if away_probs:
                    entry["away_fair_prob"] = round(sum(away_probs) / len(away_probs), 6)
            else:  # total
                over_probs = sides.get("over", [])
                under_probs = sides.get("under", [])
                count = max(len(over_probs), len(under_probs))
                entry["book_count"] = count
                if over_probs:
                    entry["over_fair_prob"] = round(sum(over_probs) / len(over_probs), 6)
                if under_probs:
                    entry["under_fair_prob"] = round(sum(under_probs) / len(under_probs), 6)
            consensus[game_id][market_name] = entry

    return consensus


def _build_ev_summary(enriched_odds: list) -> dict:
    """Create a per-game, per-market EV leaderboard of +EV opportunities.

    Structure:
        {
            "game_id": {
                "spread": [
                    {"sportsbook": "Pinnacle", "side": "home", "ev_edge": 0.023,
                     "ev_edge_pct": "+2.3%", "ev_dollar": 0.048, "odds": -105},
                    ...
                ],
                ...
            },
            ...
        }

    Only includes +EV entries (ev_edge > 0), sorted by ev_edge descending
    (biggest edge first).
    """
    summary = {}

    for record in enriched_odds:
        game_id = record["game_id"]
        sportsbook = record["sportsbook"]

        if game_id not in summary:
            summary[game_id] = {}

        for market_name in ("spread", "moneyline", "total"):
            market = record.get("markets", {}).get(market_name)
            if not market:
                continue

            if market_name not in summary[game_id]:
                summary[game_id][market_name] = []

            if market_name in ("spread", "moneyline"):
                sides = [
                    ("home", "home_ev", "home_ev_pct", "home_ev_dollar", "home_odds"),
                    ("away", "away_ev", "away_ev_pct", "away_ev_dollar", "away_odds"),
                ]
            else:
                sides = [
                    ("over", "over_ev", "over_ev_pct", "over_ev_dollar", "over_odds"),
                    ("under", "under_ev", "under_ev_pct", "under_ev_dollar", "under_odds"),
                ]

            for side_name, ev_key, pct_key, dollar_key, odds_key in sides:
                ev_edge = market.get(ev_key)
                if ev_edge is not None and ev_edge > 0:
                    summary[game_id][market_name].append({
                        "sportsbook": sportsbook,
                        "side": side_name,
                        "ev_edge": ev_edge,
                        "ev_edge_pct": market.get(pct_key, ""),
                        "ev_dollar": market.get(dollar_key, 0),
                        "odds": market.get(odds_key),
                    })

    # Sort each list by ev_edge descending (biggest edge first)
    for game_id in summary:
        for market_name in summary[game_id]:
            summary[game_id][market_name].sort(key=lambda x: x["ev_edge"], reverse=True)

    return summary


def _build_vig_summary(enriched_odds: list) -> dict:
    """Create a per-game, per-market vig leaderboard across sportsbooks.

    Structure:
        {
            "nba_20260320_lal_bos": {
                "spread": [
                    {"sportsbook": "Pinnacle", "vig": 0.019, "vig_pct": "1.9%"},
                    {"sportsbook": "DraftKings", "vig": 0.052, "vig_pct": "5.2%"},
                    ...
                ],
                "moneyline": [...],
                "total": [...]
            },
            ...
        }
    """
    summary = {}

    for record in enriched_odds:
        game_id = record["game_id"]
        sportsbook = record["sportsbook"]

        if game_id not in summary:
            summary[game_id] = {}

        for market_name in ("spread", "moneyline", "total"):
            market = record.get("markets", {}).get(market_name)
            if not market:
                continue

            if market_name not in summary[game_id]:
                summary[game_id][market_name] = []

            summary[game_id][market_name].append({
                "sportsbook": sportsbook,
                "vig": market["vig"],
                "vig_pct": market["vig_pct"],
            })

    # Sort each list by vig ascending (tightest margin first)
    for game_id in summary:
        for market_name in summary[game_id]:
            summary[game_id][market_name].sort(key=lambda x: x["vig"])

    return summary


# ---------------------------------------------------------------------------
# Stale line detection
# ---------------------------------------------------------------------------

STALE_THRESHOLD_MINUTES = 30


def _parse_timestamp(ts_str: str):
    """Parse an ISO-8601 timestamp string, returning a datetime or None."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _detect_stale_lines(
    enriched_odds: list,
    stale_threshold_minutes: int = STALE_THRESHOLD_MINUTES,
) -> dict:
    """Detect sportsbook lines that are significantly older than their peers.

    For each game, finds the newest ``last_updated`` timestamp across all
    sportsbooks and flags any record whose timestamp falls behind by more
    than *stale_threshold_minutes*.

    Returns
    -------
    dict with:
        stale_lines        – list of stale-line detail dicts, sorted most stale first
        count              – number of stale lines detected
        threshold_minutes  – the threshold that was used
        newest_per_game    – {game_id: newest_iso_timestamp} for reference
        context            – human-readable summary string
    """
    # Group records by game
    games: dict[str, list[dict]] = defaultdict(list)
    for record in enriched_odds:
        games[record["game_id"]].append(record)

    stale_lines: list[dict] = []
    newest_per_game: dict[str, str] = {}

    for game_id, records in games.items():
        timed = []
        for r in records:
            ts = _parse_timestamp(r.get("last_updated", ""))
            if ts is not None:
                timed.append({"record": r, "ts": ts})

        if len(timed) < 2:
            continue

        newest = max(t["ts"] for t in timed)
        newest_per_game[game_id] = newest.isoformat()

        for t in timed:
            age_minutes = (newest - t["ts"]).total_seconds() / 60
            if age_minutes >= stale_threshold_minutes:
                r = t["record"]
                stale_lines.append({
                    "game_id": game_id,
                    "sportsbook": r["sportsbook"],
                    "last_updated": r.get("last_updated"),
                    "newest_update": newest.isoformat(),
                    "staleness_minutes": round(age_minutes, 1),
                    "sport": r.get("sport"),
                    "home_team": r.get("home_team"),
                    "away_team": r.get("away_team"),
                    "context": (
                        f"STALE: {r['sportsbook']} line for "
                        f"{r.get('away_team')} @ {r.get('home_team')} is "
                        f"{round(age_minutes)} min behind. "
                        f"May represent value if line has moved elsewhere."
                    ),
                })

    stale_lines.sort(key=lambda s: s["staleness_minutes"], reverse=True)

    return {
        "stale_lines": stale_lines,
        "count": len(stale_lines),
        "threshold_minutes": stale_threshold_minutes,
        "newest_per_game": newest_per_game,
        "context": (
            f"Found {len(stale_lines)} stale lines "
            f"(>{stale_threshold_minutes} min old)."
            + (f" Most stale: {stale_lines[0]['context']}" if stale_lines else "")
        ),
    }


def _enrich_staleness(
    enriched_odds: list,
    stale_summary: dict,
) -> None:
    """Stamp each enriched record in-place with staleness fields.

    Adds to every record:
        staleness_minutes  – how many minutes behind the newest line for
                             the same game (0 = freshest)
        is_stale           – True if the record exceeds the threshold
    """
    # Build a quick lookup: (game_id, sportsbook) → staleness_minutes
    stale_lookup: dict[tuple[str, str], float] = {}
    for entry in stale_summary.get("stale_lines", []):
        stale_lookup[(entry["game_id"], entry["sportsbook"])] = entry["staleness_minutes"]

    newest_per_game = stale_summary.get("newest_per_game", {})
    threshold = stale_summary.get("threshold_minutes", STALE_THRESHOLD_MINUTES)

    for record in enriched_odds:
        game_id = record["game_id"]
        sportsbook = record["sportsbook"]
        key = (game_id, sportsbook)

        if key in stale_lookup:
            record["staleness_minutes"] = stale_lookup[key]
            record["is_stale"] = True
        else:
            # Compute actual staleness (may be below threshold)
            newest_iso = newest_per_game.get(game_id)
            if newest_iso:
                newest_ts = _parse_timestamp(newest_iso)
                record_ts = _parse_timestamp(record.get("last_updated", ""))
                if newest_ts and record_ts:
                    age = (newest_ts - record_ts).total_seconds() / 60
                    record["staleness_minutes"] = round(age, 1)
                else:
                    record["staleness_minutes"] = 0.0
            else:
                record["staleness_minutes"] = 0.0
            record["is_stale"] = False
