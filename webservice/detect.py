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
from itertools import combinations
from odds_math import implied_probability, calculate_vig, no_vig_probabilities, fair_odds_to_american, expected_value, arbitrage_profit, kelly_criterion

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
        enriched_odds        – list of odds records with computed fields added
        vig_summary          – per-game, per-market vig comparison across books
        ev_summary           – per-game, per-market +EV opportunities (sorted by edge)
        consensus            – consensus fair probabilities per game/market
        stale_summary        – stale line flags per sportsbook/game
        arb_profit_curves    – all book-pair arbitrage combinations
        synthetic_perfect_book – synthetic "perfect book" combining the best
                                 line from each sportsbook for every side of
                                 every market; negative total hold signals
                                 guaranteed cross-book arbitrage
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

    # Build arbitrage profit curves: all book-pair combinations per game/market
    arb_profit_curves = _build_arb_profit_curves(enriched)

    # ---------- Synthetic "Perfect Book" construction ----------
    synthetic_perfect_book = _build_synthetic_perfect_book(enriched, consensus)

    # ---------- Step 2: Cross-book analysis (moved from analyze phase) ----------
    analysis = _build_cross_book_analysis(
        enriched, consensus, stale_summary, arb_profit_curves,
    )

    return {
        "enriched_odds": enriched,
        "vig_summary": vig_summary,
        "ev_summary": ev_summary,
        "consensus": consensus,
        "stale_summary": stale_summary,
        "arb_profit_curves": arb_profit_curves,
        "synthetic_perfect_book": synthetic_perfect_book,
        "analysis": analysis,
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


# ---------------------------------------------------------------------------
# Arbitrage profit curves – all book-pair combinations
# ---------------------------------------------------------------------------


def _build_arb_profit_curves(enriched_odds: list) -> dict:
    """Compute arb profit for every sportsbook pairing per game/market.

    For each game and market type, enumerates all ordered pairs of sportsbooks
    (book_a supplies side A odds, book_b supplies side B odds) and calculates
    the arbitrage profit (or loss) for that combination.

    The result is sorted by profit descending so the most profitable pairings
    appear first — forming a "profit curve" from best to worst pairing.

    Structure
    ---------
    {
        "game_id": {
            "moneyline": {
                "curve": [
                    {
                        "book_a": "Pinnacle", "book_b": "DraftKings",
                        "side_a": "home",     "side_b": "away",
                        "odds_a": -210,       "odds_b": 205,
                        "combined_implied": 0.9534,
                        "is_arb": true,
                        "profit_pct": 4.89,
                        "stake_a_pct": 67.2,
                        "stake_b_pct": 32.8,
                    },
                    ...
                ],
                "best_pairing": { ... },        # top entry (or null)
                "arb_count": 3,                 # how many pairings are actual arbs
                "total_pairings": 56,
            },
            ...
        },
        ...
    }

    Also includes a top-level ``best_pairings`` list (across all games/markets)
    sorted by profit_pct descending.
    """
    # Group enriched records by game_id
    games: dict[str, list[dict]] = defaultdict(list)
    for record in enriched_odds:
        games[record["game_id"]].append(record)

    result: dict = {}
    all_best: list[dict] = []

    for game_id, records in games.items():
        if len(records) < 2:
            continue

        first = records[0]
        game_entry: dict = {}

        for market_type in ("spread", "moneyline", "total"):
            if market_type in ("spread", "moneyline"):
                side_a_name, side_b_name = "home", "away"
                key_a, key_b = "home_odds", "away_odds"
            else:
                side_a_name, side_b_name = "over", "under"
                key_a, key_b = "over_odds", "under_odds"

            # Collect (sportsbook, odds_a, odds_b) for books that have this market
            book_odds: list[dict] = []
            for r in records:
                market = r.get("markets", {}).get(market_type)
                if market and key_a in market and key_b in market:
                    book_odds.append({
                        "sportsbook": r["sportsbook"],
                        "odds_a": market[key_a],
                        "odds_b": market[key_b],
                    })

            if len(book_odds) < 2:
                continue

            curve: list[dict] = []

            # Enumerate all *ordered* pairs: book_i supplies side A, book_j supplies side B
            # We want both (i→A, j→B) and (j→A, i→B) since they may yield different results
            for i, j in combinations(range(len(book_odds)), 2):
                for ba, bb in [(book_odds[i], book_odds[j]),
                               (book_odds[j], book_odds[i])]:
                    # Skip same-book pairing (already covered by single-book vig)
                    if ba["sportsbook"] == bb["sportsbook"]:
                        continue

                    arb = arbitrage_profit(ba["odds_a"], bb["odds_b"])
                    entry = {
                        "book_a": ba["sportsbook"],
                        "book_b": bb["sportsbook"],
                        "side_a": side_a_name,
                        "side_b": side_b_name,
                        "odds_a": ba["odds_a"],
                        "odds_b": bb["odds_b"],
                        "combined_implied": arb["combined_implied"],
                        "is_arb": arb["is_arb"],
                        "profit_pct": arb["profit_pct"],
                        "stake_a_pct": arb["stake_a_pct"],
                        "stake_b_pct": arb["stake_b_pct"],
                    }
                    curve.append(entry)

            # Sort by profit descending (best arb first)
            curve.sort(key=lambda x: x["profit_pct"], reverse=True)
            arb_count = sum(1 for c in curve if c["is_arb"])

            best = curve[0] if curve else None
            game_entry[market_type] = {
                "curve": curve,
                "best_pairing": best,
                "arb_count": arb_count,
                "total_pairings": len(curve),
            }

            if best and best["is_arb"]:
                all_best.append({
                    "game_id": game_id,
                    "sport": first.get("sport"),
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market_type": market_type,
                    **best,
                })

        if game_entry:
            result[game_id] = game_entry

    # Sort global best pairings by profit
    all_best.sort(key=lambda x: x["profit_pct"], reverse=True)

    return {
        "games": result,
        "best_pairings": all_best,
        "total_arb_pairings": sum(
            v[mt]["arb_count"]
            for v in result.values()
            for mt in v
        ),
        "context": (
            f"Analyzed book-pair arb curves across {len(result)} games. "
            f"Found {len(all_best)} profitable pairings."
            + (f" Best: {all_best[0]['book_a']}+{all_best[0]['book_b']} on "
               f"{all_best[0]['game_id']} {all_best[0]['market_type']} = "
               f"{all_best[0]['profit_pct']}% profit"
               if all_best else "")
        ),
    }


# ---------------------------------------------------------------------------
# Synthetic "Perfect Book" construction
# ---------------------------------------------------------------------------


def _build_synthetic_perfect_book(enriched_odds: list, consensus: dict) -> dict:
    """Construct a synthetic 'perfect book' by combining the best line from
    each sportsbook for every side of every market.

    For each game and market type:
    1. Finds the best (highest / most bettor-friendly) odds on each side
       across all available sportsbooks.
    2. Calculates the combined implied probability (synthetic hold) of those
       best odds.  If the synthetic hold is **negative**, guaranteed arbitrage
       exists across those books.
    3. Compares the synthetic best-available odds against consensus fair odds
       to measure the edge a disciplined line-shopper captures.

    Structure
    ---------
    {
        "games": {
            "<game_id>": {
                "sport": "...",
                "home_team": "...",
                "away_team": "...",
                "book_count": 8,
                "markets": {
                    "spread": {
                        "best_home": { "odds", "sportsbook", "line",
                                       "implied_prob", "fair_prob",
                                       "edge_vs_fair" },
                        "best_away": { ... },
                        "combined_implied": 0.9823,
                        "synthetic_hold_pct": "-1.77%",
                        "is_arb": true,
                        "arb_profit_pct": 1.80,
                        "arb_stakes": { "side_a_pct", "side_b_pct" },
                    },
                    ...
                },
                "avg_synthetic_hold_pct": "-0.5%",
                "arb_markets": 1,
            },
            ...
        },
        "aggregate": { ... },
        "arb_alerts": [ ... ],
    }
    """
    # Group enriched records by game_id
    games: dict[str, list[dict]] = defaultdict(list)
    for record in enriched_odds:
        games[record["game_id"]].append(record)

    game_results: dict = {}
    all_holds: list[float] = []
    market_type_holds: dict[str, list[float]] = {
        "spread": [], "moneyline": [], "total": [],
    }
    arb_alerts: list[dict] = []

    for game_id, records in games.items():
        if len(records) < 2:
            continue

        first = records[0]
        game_entry: dict = {
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
                line_key = "home_line" if market_type == "spread" else None
            else:
                side_a_key, side_b_key = "over_odds", "under_odds"
                side_a_label, side_b_label = "over", "under"
                line_key = "line"

            # ---- Find the best (highest) odds on each side ----
            best_a: dict | None = None
            best_b: dict | None = None

            for r in records:
                market = r.get("markets", {}).get(market_type)
                if not market:
                    continue

                odds_a = market.get(side_a_key)
                odds_b = market.get(side_b_key)

                if odds_a is not None:
                    if best_a is None or odds_a > best_a["odds"]:
                        best_a = {
                            "odds": odds_a,
                            "sportsbook": r["sportsbook"],
                            "line": market.get(line_key) if line_key else None,
                        }
                if odds_b is not None:
                    if best_b is None or odds_b > best_b["odds"]:
                        best_b = {
                            "odds": odds_b,
                            "sportsbook": r["sportsbook"],
                            "line": market.get(line_key) if line_key else None,
                        }

            if best_a is None or best_b is None:
                continue

            # ---- Calculate synthetic hold ----
            prob_a = implied_probability(best_a["odds"])
            prob_b = implied_probability(best_b["odds"])
            combined_implied = prob_a + prob_b
            synthetic_hold = combined_implied - 1.0  # negative = bettor edge

            # ---- Compare against consensus fair odds ----
            game_con = consensus.get(game_id, {}).get(market_type, {})
            if market_type in ("spread", "moneyline"):
                fair_a = game_con.get("home_fair_prob", 0.5)
                fair_b = game_con.get("away_fair_prob", 0.5)
            else:
                fair_a = game_con.get("over_fair_prob", 0.5)
                fair_b = game_con.get("under_fair_prob", 0.5)

            edge_a = fair_a - prob_a  # positive = bettor value on this side
            edge_b = fair_b - prob_b

            # ---- Arbitrage details (when hold < 0) ----
            is_arb = combined_implied < 1.0
            if is_arb:
                arb_profit_pct = round((1.0 - combined_implied) / combined_implied * 100, 4)
                stake_a_pct = round(prob_a / combined_implied * 100, 2)
                stake_b_pct = round(prob_b / combined_implied * 100, 2)
            else:
                arb_profit_pct = 0.0
                stake_a_pct = 50.0
                stake_b_pct = 50.0

            market_entry = {
                f"best_{side_a_label}": {
                    "odds": best_a["odds"],
                    "sportsbook": best_a["sportsbook"],
                    "line": best_a["line"],
                    "implied_prob": round(prob_a, 6),
                    "implied_prob_pct": f"{round(prob_a * 100, 2)}%",
                    "fair_prob": round(fair_a, 6),
                    "edge_vs_fair": round(edge_a * 100, 3),
                    "edge_vs_fair_pct": f"{round(edge_a * 100, 3)}%",
                },
                f"best_{side_b_label}": {
                    "odds": best_b["odds"],
                    "sportsbook": best_b["sportsbook"],
                    "line": best_b["line"],
                    "implied_prob": round(prob_b, 6),
                    "implied_prob_pct": f"{round(prob_b * 100, 2)}%",
                    "fair_prob": round(fair_b, 6),
                    "edge_vs_fair": round(edge_b * 100, 3),
                    "edge_vs_fair_pct": f"{round(edge_b * 100, 3)}%",
                },
                "combined_implied": round(combined_implied, 6),
                "combined_implied_pct": f"{round(combined_implied * 100, 2)}%",
                "synthetic_hold": round(synthetic_hold, 6),
                "synthetic_hold_pct": f"{round(synthetic_hold * 100, 3)}%",
                "is_arb": is_arb,
                "arb_profit_pct": arb_profit_pct,
                "arb_stakes": {
                    f"{side_a_label}_pct": stake_a_pct,
                    f"{side_b_label}_pct": stake_b_pct,
                },
            }

            # Add consensus line for spread / total markets
            if market_type == "spread":
                lines = [
                    r["markets"][market_type].get("home_line", 0)
                    for r in records
                    if market_type in r.get("markets", {})
                    and "home_line" in r["markets"][market_type]
                ]
                if lines:
                    market_entry["consensus_line"] = round(
                        sum(lines) / len(lines), 1,
                    )
            elif market_type == "total":
                lines = [
                    r["markets"][market_type].get("line", 0)
                    for r in records
                    if market_type in r.get("markets", {})
                    and "line" in r["markets"][market_type]
                ]
                if lines:
                    market_entry["consensus_line"] = round(
                        sum(lines) / len(lines), 1,
                    )

            game_entry["markets"][market_type] = market_entry
            all_holds.append(synthetic_hold)
            market_type_holds[market_type].append(synthetic_hold)

            # Collect arb alerts
            if is_arb:
                arb_alerts.append({
                    "game_id": game_id,
                    "sport": first.get("sport"),
                    "home_team": first.get("home_team"),
                    "away_team": first.get("away_team"),
                    "market_type": market_type,
                    "synthetic_hold_pct": f"{round(synthetic_hold * 100, 3)}%",
                    "arb_profit_pct": arb_profit_pct,
                    f"best_{side_a_label}_book": best_a["sportsbook"],
                    f"best_{side_a_label}_odds": best_a["odds"],
                    f"best_{side_b_label}_book": best_b["sportsbook"],
                    f"best_{side_b_label}_odds": best_b["odds"],
                    "arb_stakes": {
                        f"{side_a_label}_pct": stake_a_pct,
                        f"{side_b_label}_pct": stake_b_pct,
                    },
                })

        # Per-game summary
        if game_entry["markets"]:
            game_holds = [
                m["synthetic_hold"]
                for m in game_entry["markets"].values()
            ]
            game_entry["avg_synthetic_hold_pct"] = (
                f"{round(sum(game_holds) / len(game_holds) * 100, 3)}%"
            )
            game_entry["arb_markets"] = sum(
                1 for m in game_entry["markets"].values() if m["is_arb"]
            )
            game_results[game_id] = game_entry

    # ---- Aggregate statistics ----
    avg_hold = sum(all_holds) / len(all_holds) if all_holds else 0
    arb_market_count = sum(1 for h in all_holds if h < 0)

    by_market_summary: dict = {}
    for mkt in ("spread", "moneyline", "total"):
        holds = market_type_holds[mkt]
        if holds:
            by_market_summary[mkt] = {
                "avg_synthetic_hold_pct": f"{round(sum(holds) / len(holds) * 100, 3)}%",
                "min_hold_pct": f"{round(min(holds) * 100, 3)}%",
                "max_hold_pct": f"{round(max(holds) * 100, 3)}%",
                "arb_markets": sum(1 for h in holds if h < 0),
                "market_count": len(holds),
            }

    # Sort arb alerts by profit descending
    arb_alerts.sort(key=lambda a: a["arb_profit_pct"], reverse=True)

    return {
        "games": game_results,
        "game_count": len(game_results),
        "aggregate": {
            "total_markets_analyzed": len(all_holds),
            "avg_synthetic_hold_pct": f"{round(avg_hold * 100, 3)}%",
            "avg_sharp_edge_pct": f"{round(-avg_hold * 100, 3)}%",
            "arb_market_count": arb_market_count,
            "guaranteed_arb_exists": arb_market_count > 0,
            "by_market_type": by_market_summary,
        },
        "arb_alerts": arb_alerts,
        "context": (
            f"Synthetic perfect book constructed across {len(game_results)} games "
            f"and {len(all_holds)} markets. "
            f"Average synthetic hold: {round(avg_hold * 100, 3)}% "
            f"(sharp edge: {round(-avg_hold * 100, 3)}%). "
            f"{arb_market_count} market(s) with negative hold "
            f"(guaranteed arbitrage exists across books)."
        ),
    }


# ---------------------------------------------------------------------------
# Step 2: Cross-book analysis (efficiency, best lines, middles, outliers)
# ---------------------------------------------------------------------------
# This logic was previously in ai_service.run_analyze_phase().  It is pure
# Python (no AI calls) and belongs in the detect phase so the pipeline's
# "analyze" phase is kept free for future AI-powered analysis.
# ---------------------------------------------------------------------------

def _build_cross_book_analysis(
    enriched_odds: list,
    consensus: dict,
    stale_summary: dict,
    arb_profit_curves: dict,
) -> dict:
    """Cross-sportsbook analysis — efficiency, best lines, middles & outliers.

    This is the second step of the detect phase.  It reuses the enriched data
    from step 1 and layers on additional cross-book insights:

    1. Efficiency Ranking — books ranked by average vig
    2. Best Line Shopping — best odds for each game+market+side
    3. Arbitrage — flattened from step-1 arb_profit_curves
    4. Middles Detection — spread/total line gaps allowing both-side wins
    5. Outlier / Anomaly Detection — lines deviating from consensus

    Returns a dict with the same ``analysis`` structure that
    ``run_analyze_phase`` previously produced.
    """

    odds = enriched_odds

    # ── Group odds by game ──────────────────────────────────────────────
    games: dict[str, dict] = {}
    all_books: set[str] = set()
    for row in odds:
        gid = row.get("game_id", "")
        book = row.get("sportsbook", "")
        all_books.add(book)
        if gid not in games:
            games[gid] = {
                "home": row.get("home_team", ""),
                "away": row.get("away_team", ""),
                "rows": [],
            }
        games[gid]["rows"].append(row)

    # ── 1. Vig / Efficiency Ranking ─────────────────────────────────────
    book_vigs: dict[str, list[float]] = {}
    for row in odds:
        book = row.get("sportsbook", "")
        markets = row.get("markets", {})
        for mkt in markets.values():
            v = mkt.get("vig")
            if v is not None:
                book_vigs.setdefault(book, []).append(v)

    efficiency = []
    for book, vigs in book_vigs.items():
        avg = sum(vigs) / len(vigs) if vigs else 0
        efficiency.append({
            "book": book,
            "avg_vig": round(avg, 6),
            "avg_vig_pct": f"{round(avg * 100, 2)}%",
            "markets_counted": len(vigs),
        })
    efficiency.sort(key=lambda x: x["avg_vig"])

    # ── 2. Best Line Shopping ───────────────────────────────────────────
    best_lines = []
    for gid, gdata in games.items():
        rows = gdata["rows"]
        # Spread: home side & away side
        spread_rows = [(r["sportsbook"], r["markets"]["spread"]) for r in rows if "spread" in r.get("markets", {})]
        if spread_rows:
            best_home = max(spread_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(spread_rows, key=lambda x: x[1]["away_odds"])
            best_lines.append({
                "game_id": gid, "market": "spread", "side": "home",
                "line": best_home[1].get("home_line"),
                "best_odds": best_home[1]["home_odds"],
                "best_book": best_home[0],
                "home_team": gdata["home"],
            })
            best_lines.append({
                "game_id": gid, "market": "spread", "side": "away",
                "line": best_away[1].get("away_line"),
                "best_odds": best_away[1]["away_odds"],
                "best_book": best_away[0],
                "away_team": gdata["away"],
            })

        # Moneyline: home & away
        ml_rows = [(r["sportsbook"], r["markets"]["moneyline"]) for r in rows if "moneyline" in r.get("markets", {})]
        if ml_rows:
            best_home = max(ml_rows, key=lambda x: x[1]["home_odds"])
            best_away = max(ml_rows, key=lambda x: x[1]["away_odds"])
            best_lines.append({
                "game_id": gid, "market": "moneyline", "side": "home",
                "best_odds": best_home[1]["home_odds"],
                "best_book": best_home[0],
                "home_team": gdata["home"],
            })
            best_lines.append({
                "game_id": gid, "market": "moneyline", "side": "away",
                "best_odds": best_away[1]["away_odds"],
                "best_book": best_away[0],
                "away_team": gdata["away"],
            })

        # Totals: over & under
        total_rows = [(r["sportsbook"], r["markets"]["total"]) for r in rows if "total" in r.get("markets", {})]
        if total_rows:
            best_over = max(total_rows, key=lambda x: x[1]["over_odds"])
            best_under = max(total_rows, key=lambda x: x[1]["under_odds"])
            best_lines.append({
                "game_id": gid, "market": "total", "side": "over",
                "line": best_over[1].get("line"),
                "best_odds": best_over[1]["over_odds"],
                "best_book": best_over[0],
            })
            best_lines.append({
                "game_id": gid, "market": "total", "side": "under",
                "line": best_under[1].get("line"),
                "best_odds": best_under[1]["under_odds"],
                "best_book": best_under[0],
            })

    # ── 3. Arbitrage — flatten arb_profit_curves ────────────────────────
    arbitrage = []
    arb_games = arb_profit_curves.get("games", arb_profit_curves) if isinstance(arb_profit_curves, dict) else {}
    best_pairings = arb_profit_curves.get("best_pairings", []) if isinstance(arb_profit_curves, dict) else []

    if best_pairings:
        for bp in best_pairings:
            gid = bp.get("game_id", "")
            gdata = games.get(gid, {})
            arbitrage.append({
                "game_id": gid,
                "market": bp.get("market_type", ""),
                "home_team": bp.get("home_team", gdata.get("home", "")),
                "away_team": bp.get("away_team", gdata.get("away", "")),
                "leg_1": {"side": bp.get("side_a", ""),
                          "odds": bp.get("odds_a"),
                          "book": bp.get("book_a", ""),
                          "implied_prob": round(implied_probability(bp["odds_a"]), 4) if bp.get("odds_a") else None},
                "leg_2": {"side": bp.get("side_b", ""),
                          "odds": bp.get("odds_b"),
                          "book": bp.get("book_b", ""),
                          "implied_prob": round(implied_probability(bp["odds_b"]), 4) if bp.get("odds_b") else None},
                "combined_implied": bp.get("combined_implied"),
                "profit_pct": bp.get("profit_pct", 0),
            })
    else:
        for gid, market_map in arb_games.items():
            gdata = games.get(gid, {})
            for market_name, mdata in market_map.items():
                if isinstance(mdata, dict):
                    for entry in mdata.get("curve", []):
                        if entry.get("is_arb"):
                            arbitrage.append({
                                "game_id": gid, "market": market_name,
                                "home_team": gdata.get("home", ""),
                                "away_team": gdata.get("away", ""),
                                "leg_1": {"side": entry.get("side_a", ""),
                                          "odds": entry.get("odds_a"),
                                          "book": entry.get("book_a", ""),
                                          "implied_prob": round(implied_probability(entry["odds_a"]), 4) if entry.get("odds_a") else None},
                                "leg_2": {"side": entry.get("side_b", ""),
                                          "odds": entry.get("odds_b"),
                                          "book": entry.get("book_b", ""),
                                          "implied_prob": round(implied_probability(entry["odds_b"]), 4) if entry.get("odds_b") else None},
                                "combined_implied": entry.get("combined_implied"),
                                "profit_pct": entry.get("profit_pct", 0),
                            })
        seen_arb = set()
        unique_arb = []
        for a in arbitrage:
            key = (a["game_id"], a["market"],
                   frozenset([a["leg_1"]["book"], a["leg_2"]["book"]]))
            if key not in seen_arb:
                seen_arb.add(key)
                unique_arb.append(a)
        arbitrage = unique_arb

    arbitrage.sort(key=lambda x: x["profit_pct"], reverse=True)

    # ── 4. Middles Detection ────────────────────────────────────────────
    middles = []
    for gid, gdata in games.items():
        rows = gdata["rows"]

        # Spread middles
        spread_rows = [(r["sportsbook"], r["markets"]["spread"]) for r in rows if "spread" in r.get("markets", {})]
        if len(spread_rows) >= 2:
            for i in range(len(spread_rows)):
                for j in range(i + 1, len(spread_rows)):
                    book_a, mkt_a = spread_rows[i]
                    book_b, mkt_b = spread_rows[j]
                    home_line_a = mkt_a.get("home_line", 0)
                    away_line_b = mkt_b.get("away_line", 0)
                    home_line_b = mkt_b.get("home_line", 0)
                    away_line_a = mkt_a.get("away_line", 0)

                    if abs(home_line_a) != abs(home_line_b):
                        spread_gap = abs(home_line_a) - abs(home_line_b)
                        if abs(spread_gap) >= 1.0:
                            if abs(home_line_a) < abs(home_line_b):
                                middles.append({
                                    "game_id": gid, "market": "spread",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "home", "line": home_line_a,
                                              "odds": mkt_a["home_odds"], "book": book_a},
                                    "leg_2": {"side": "away", "line": away_line_b,
                                              "odds": mkt_b["away_odds"], "book": book_b},
                                    "middle_gap": abs(spread_gap),
                                    "middle_range": f"Result lands between {home_line_a} and {home_line_b}",
                                })
                            else:
                                middles.append({
                                    "game_id": gid, "market": "spread",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "home", "line": home_line_b,
                                              "odds": mkt_b["home_odds"], "book": book_b},
                                    "leg_2": {"side": "away", "line": away_line_a,
                                              "odds": mkt_a["away_odds"], "book": book_a},
                                    "middle_gap": abs(spread_gap),
                                    "middle_range": f"Result lands between {home_line_b} and {home_line_a}",
                                })

        # Total middles
        total_rows = [(r["sportsbook"], r["markets"]["total"]) for r in rows if "total" in r.get("markets", {})]
        if len(total_rows) >= 2:
            for i in range(len(total_rows)):
                for j in range(i + 1, len(total_rows)):
                    book_a, mkt_a = total_rows[i]
                    book_b, mkt_b = total_rows[j]
                    line_a = mkt_a.get("line", 0)
                    line_b = mkt_b.get("line", 0)
                    if line_a != line_b:
                        gap = abs(line_a - line_b)
                        if gap >= 1.0:
                            if line_a < line_b:
                                middles.append({
                                    "game_id": gid, "market": "total",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "over", "line": line_a,
                                              "odds": mkt_a["over_odds"], "book": book_a},
                                    "leg_2": {"side": "under", "line": line_b,
                                              "odds": mkt_b["under_odds"], "book": book_b},
                                    "middle_gap": gap,
                                    "middle_range": f"Total lands between {line_a} and {line_b}",
                                })
                            else:
                                middles.append({
                                    "game_id": gid, "market": "total",
                                    "home_team": gdata["home"], "away_team": gdata["away"],
                                    "leg_1": {"side": "over", "line": line_b,
                                              "odds": mkt_b["over_odds"], "book": book_b},
                                    "leg_2": {"side": "under", "line": line_a,
                                              "odds": mkt_a["under_odds"], "book": book_a},
                                    "middle_gap": gap,
                                    "middle_range": f"Total lands between {line_b} and {line_a}",
                                })

    # Deduplicate middles
    seen_middles = set()
    unique_middles = []
    for m in middles:
        key = (m["game_id"], m["market"], m["leg_1"]["book"], m["leg_2"]["book"])
        rev_key = (m["game_id"], m["market"], m["leg_2"]["book"], m["leg_1"]["book"])
        if key not in seen_middles and rev_key not in seen_middles:
            seen_middles.add(key)
            unique_middles.append(m)
    middles = sorted(unique_middles, key=lambda x: x["middle_gap"], reverse=True)

    # ── 5. Outlier / Anomaly Detection ──────────────────────────────────
    outliers = []
    OUTLIER_THRESHOLD = 15  # American odds points deviation from consensus

    for gid, gdata in games.items():
        rows = gdata["rows"]
        for market_name in ("spread", "moneyline", "total"):
            market_rows = [(r["sportsbook"], r["markets"][market_name])
                           for r in rows if market_name in r.get("markets", {})]
            if len(market_rows) < 3:
                continue

            if market_name in ("spread", "moneyline"):
                odds_keys = [("home_odds", "home"), ("away_odds", "away")]
            else:
                odds_keys = [("over_odds", "over"), ("under_odds", "under")]

            for odds_key, side_label in odds_keys:
                values = [m[1][odds_key] for m in market_rows]
                avg_odds = sum(values) / len(values)

                for book, mkt in market_rows:
                    deviation = abs(mkt[odds_key] - avg_odds)
                    if deviation >= OUTLIER_THRESHOLD:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": market_name,
                            "side": side_label,
                            "sportsbook": book,
                            "odds": mkt[odds_key],
                            "consensus_avg": round(avg_odds, 1),
                            "deviation": round(deviation, 1),
                            "type": "odds_outlier",
                        })

            # Line outliers for spread and total
            if market_name == "spread":
                lines = [m[1].get("home_line", 0) for m in market_rows]
                avg_line = sum(lines) / len(lines)
                for book, mkt in market_rows:
                    line_dev = abs(mkt.get("home_line", 0) - avg_line)
                    if line_dev >= 1.0:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": "spread",
                            "sportsbook": book,
                            "line": mkt.get("home_line"),
                            "consensus_line": round(avg_line, 1),
                            "deviation": round(line_dev, 1),
                            "type": "line_outlier",
                        })
            elif market_name == "total":
                lines = [m[1].get("line", 0) for m in market_rows]
                avg_line = sum(lines) / len(lines)
                for book, mkt in market_rows:
                    line_dev = abs(mkt.get("line", 0) - avg_line)
                    if line_dev >= 1.0:
                        outliers.append({
                            "game_id": gid,
                            "home_team": gdata["home"],
                            "away_team": gdata["away"],
                            "market": "total",
                            "sportsbook": book,
                            "line": mkt.get("line"),
                            "consensus_line": round(avg_line, 1),
                            "deviation": round(line_dev, 1),
                            "type": "line_outlier",
                        })

    outliers.sort(key=lambda x: x["deviation"], reverse=True)

    # ── 6. Stale Lines — from step 1 ───────────────────────────────────
    stale_lines = stale_summary.get("stale_lines", [])

    # ── 7. Fair Odds — from step 1 consensus ────────────────────────────
    fair_odds_summary = []
    for gid, market_map in consensus.items():
        gdata = games.get(gid, {})
        game_fair: dict = {"game_id": gid, "home_team": gdata.get("home", ""), "away_team": gdata.get("away", "")}
        for market_name, probs in market_map.items():
            if market_name in ("spread", "moneyline"):
                if "home_fair_prob" in probs:
                    game_fair[f"{market_name}_home_fair_prob"] = probs["home_fair_prob"]
                if "away_fair_prob" in probs:
                    game_fair[f"{market_name}_away_fair_prob"] = probs["away_fair_prob"]
            else:
                if "over_fair_prob" in probs:
                    game_fair["total_over_fair_prob"] = probs["over_fair_prob"]
                if "under_fair_prob" in probs:
                    game_fair["total_under_fair_prob"] = probs["under_fair_prob"]
        fair_odds_summary.append(game_fair)

    # ── Build summary ───────────────────────────────────────────────────
    summary_parts = [f"Analyzed {len(games)} games across {len(all_books)} sportsbooks."]
    if arbitrage:
        summary_parts.append(f"Found {len(arbitrage)} arbitrage opportunity(ies).")
    if middles:
        summary_parts.append(f"Found {len(middles)} middle opportunity(ies).")
    if outliers:
        summary_parts.append(f"Detected {len(outliers)} outlier(s).")
    if stale_lines:
        summary_parts.append(f"Detected {len(stale_lines)} stale line(s).")

    return {
        "games_count": len(games),
        "books_count": len(all_books),
        "efficiency_ranking": efficiency,
        "best_lines": best_lines,
        "arbitrage": arbitrage,
        "middles": middles,
        "outliers": outliers,
        "stale_lines": stale_lines,
        "fair_odds_summary": fair_odds_summary,
        "summary": " ".join(summary_parts),
    }
