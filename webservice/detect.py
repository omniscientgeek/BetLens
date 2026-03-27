"""
Detection phase – compute implied probabilities, vig, and fair odds for every
record in a sample odds file.

This module is called by the processing pipeline in app.py during the
"detect" phase.
"""

import json
import os
import aiofiles
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, fair_odds_to_american

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

    Also computes a per-game vig comparison across sportsbooks.

    Returns
    -------
    dict with:
        enriched_odds  – list of odds records with computed fields added
        vig_summary    – per-game, per-market vig comparison across books
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

    # Build a vig summary: game_id → market → list of {sportsbook, vig, vig_pct}
    vig_summary = _build_vig_summary(enriched)

    return {
        "enriched_odds": enriched,
        "vig_summary": vig_summary,
    }


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
