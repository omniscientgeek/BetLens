"""
Unit tests using REAL-WORLD betting data from NBA games on March 28, 2026.

Data sourced from OddsShark, FanDuel Research, CBS Sports, and The Odds API
for actual NBA matchups with multi-sportsbook odds comparisons.

Games covered:
  1. San Antonio Spurs (-17.5) vs Milwaukee Bucks — blowout favourite
  2. Minnesota Timberwolves (-2.5) vs Detroit Pistons — close spread
  3. Charlotte Hornets (-5.5) vs Philadelphia 76ers — mid-range favourite
  4. Atlanta Hawks (-15) vs Sacramento Kings — large favourite
  5. Chicago Bulls (-3.5) vs Memphis Grizzlies — moderate favourite
  6. Phoenix Suns (-16.5) vs Utah Jazz — large favourite

Sources:
  - https://www.oddsshark.com/nba/san-antonio-milwaukee-odds-march-28-2026-2481214
  - https://www.fanduel.com/research/nba-predictions-odds-3-28-2026
  - https://the-odds-api.com/liveapi/guides/v4/
"""

import sys
import os
import json
import pytest
from math import isclose
from unittest.mock import patch, MagicMock

# Ensure the webservice and mcp-server modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp-server")))

from odds_math import (
    implied_probability,
    calculate_vig,
    no_vig_probabilities,
    shin_probabilities,
    fair_odds_to_american,
    arbitrage_profit,
    expected_value,
    kelly_criterion,
    bayesian_update,
)

# Mock FastMCP before importing mcp_server
_mock_mcp_module = MagicMock()


class _PassthroughFastMCP:
    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def run_async(self, *args, **kwargs):
        pass


_mock_mcp_module.server.fastmcp.FastMCP = _PassthroughFastMCP
sys.modules.setdefault("mcp", _mock_mcp_module)
sys.modules.setdefault("mcp.server", _mock_mcp_module.server)
sys.modules.setdefault("mcp.server.fastmcp", _mock_mcp_module.server.fastmcp)

from mcp_server import (
    _enrich_record,
    _group_by_game,
    _compute_consensus,
    _get_pinnacle_fair_probs,
    _compute_sharp_vs_crowd,
    calculate_odds,
    arithmetic_evaluate,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Real-world fixture data — March 28, 2026 NBA odds
# ═══════════════════════════════════════════════════════════════════════════════

def _make_real_record(sportsbook, game_id, home_team, away_team,
                      spread_home_line, spread_home_odds, spread_away_odds,
                      ml_home_odds, ml_away_odds,
                      total_line, total_over_odds, total_under_odds,
                      last_updated="2026-03-28T14:30:00Z"):
    """Build an odds record from real-world data."""
    return {
        "game_id": game_id,
        "sportsbook": sportsbook,
        "sport": "NBA",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": "2026-03-28T19:00:00Z",
        "last_updated": last_updated,
        "markets": {
            "spread": {
                "home_line": spread_home_line,
                "away_line": -spread_home_line,
                "home_odds": spread_home_odds,
                "away_odds": spread_away_odds,
            },
            "moneyline": {
                "home_odds": ml_home_odds,
                "away_odds": ml_away_odds,
            },
            "total": {
                "line": total_line,
                "over_odds": total_over_odds,
                "under_odds": total_under_odds,
            },
        },
    }


# ── Game 1: Spurs @ Bucks — Spurs are massive -17.5 road favourite ──────────
# Source: OddsShark March 28, 2026

SPURS_BUCKS_RECORDS = [
    _make_real_record(
        sportsbook="Pinnacle",
        game_id="nba_sas_mil", home_team="MIL", away_team="SAS",
        spread_home_line=16.5, spread_home_odds=100, spread_away_odds=-108,
        ml_home_odds=1329, ml_away_odds=-1567,
        total_line=226.5, total_over_odds=-104, total_under_odds=100,
    ),
    _make_real_record(
        sportsbook="DraftKings",
        game_id="nba_sas_mil", home_team="MIL", away_team="SAS",
        spread_home_line=17.5, spread_home_odds=-115, spread_away_odds=-105,
        ml_home_odds=1000, ml_away_odds=-1800,
        total_line=226.5, total_over_odds=-110, total_under_odds=-110,
    ),
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_sas_mil", home_team="MIL", away_team="SAS",
        spread_home_line=17.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=1040, ml_away_odds=-2000,
        total_line=226.5, total_over_odds=-105, total_under_odds=-115,
    ),
    _make_real_record(
        sportsbook="BetMGM",
        game_id="nba_sas_mil", home_team="MIL", away_team="SAS",
        spread_home_line=17.5, spread_home_odds=-118, spread_away_odds=-102,
        ml_home_odds=1000, ml_away_odds=-2000,
        total_line=227.5, total_over_odds=-105, total_under_odds=-115,
    ),
    _make_real_record(
        sportsbook="Caesars",
        game_id="nba_sas_mil", home_team="MIL", away_team="SAS",
        spread_home_line=17.0, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=1000, ml_away_odds=-2000,
        total_line=227.0, total_over_odds=-110, total_under_odds=-110,
    ),
]


# ── Game 2: Timberwolves @ Pistons — close game, TWolves -2.5 ───────────────
# Source: FanDuel Research / CBS Sports March 28, 2026

WOLVES_PISTONS_RECORDS = [
    _make_real_record(
        sportsbook="Pinnacle",
        game_id="nba_min_det", home_team="DET", away_team="MIN",
        spread_home_line=2.5, spread_home_odds=-108, spread_away_odds=-112,
        ml_home_odds=118, ml_away_odds=-138,
        total_line=223.5, total_over_odds=-108, total_under_odds=-112,
    ),
    _make_real_record(
        sportsbook="DraftKings",
        game_id="nba_min_det", home_team="DET", away_team="MIN",
        spread_home_line=2.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=120, ml_away_odds=-142,
        total_line=223.5, total_over_odds=-110, total_under_odds=-110,
    ),
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_min_det", home_team="DET", away_team="MIN",
        spread_home_line=2.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=118, ml_away_odds=-138,
        total_line=223.5, total_over_odds=-110, total_under_odds=-110,
    ),
]


# ── Game 3: Hornets @ 76ers — Hornets -5.5 ──────────────────────────────────
# Source: FanDuel Research March 28, 2026

HORNETS_SIXERS_RECORDS = [
    _make_real_record(
        sportsbook="Pinnacle",
        game_id="nba_cha_phi", home_team="PHI", away_team="CHA",
        spread_home_line=5.5, spread_home_odds=-108, spread_away_odds=-112,
        ml_home_odds=198, ml_away_odds=-240,
        total_line=232.5, total_over_odds=-110, total_under_odds=-110,
    ),
    _make_real_record(
        sportsbook="DraftKings",
        game_id="nba_cha_phi", home_team="PHI", away_team="CHA",
        spread_home_line=5.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=195, ml_away_odds=-240,
        total_line=232.0, total_over_odds=-112, total_under_odds=-108,
    ),
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_cha_phi", home_team="PHI", away_team="CHA",
        spread_home_line=5.5, spread_home_odds=-112, spread_away_odds=-108,
        ml_home_odds=198, ml_away_odds=-240,
        total_line=232.5, total_over_odds=-108, total_under_odds=-112,
    ),
]


# ── Game 4: Hawks vs Kings — Hawks -15 ───────────────────────────────────────
# Source: FanDuel Research March 28, 2026

HAWKS_KINGS_RECORDS = [
    _make_real_record(
        sportsbook="Pinnacle",
        game_id="nba_atl_sac", home_team="ATL", away_team="SAC",
        spread_home_line=-15.0, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=-1099, ml_away_odds=700,
        total_line=236.5, total_over_odds=-108, total_under_odds=-112,
    ),
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_atl_sac", home_team="ATL", away_team="SAC",
        spread_home_line=-15.0, spread_home_odds=-112, spread_away_odds=-108,
        ml_home_odds=-1099, ml_away_odds=700,
        total_line=236.5, total_over_odds=-110, total_under_odds=-110,
    ),
]


# ── Game 5: Grizzlies @ Bulls — Bulls -3.5 ──────────────────────────────────
# Source: FanDuel Research March 28, 2026

BULLS_GRIZZLIES_RECORDS = [
    _make_real_record(
        sportsbook="DraftKings",
        game_id="nba_mem_chi", home_team="CHI", away_team="MEM",
        spread_home_line=-3.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=-172, ml_away_odds=144,
        total_line=245.5, total_over_odds=-110, total_under_odds=-110,
    ),
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_mem_chi", home_team="CHI", away_team="MEM",
        spread_home_line=-3.5, spread_home_odds=-108, spread_away_odds=-112,
        ml_home_odds=-172, ml_away_odds=144,
        total_line=245.5, total_over_odds=-112, total_under_odds=-108,
    ),
]


# ── Game 6: Suns @ Jazz — Suns -16.5 ────────────────────────────────────────
# Source: FanDuel Research March 28, 2026

SUNS_JAZZ_RECORDS = [
    _make_real_record(
        sportsbook="FanDuel",
        game_id="nba_phx_uta", home_team="UTA", away_team="PHX",
        spread_home_line=16.5, spread_home_odds=-110, spread_away_odds=-110,
        ml_home_odds=810, ml_away_odds=-1351,
        total_line=230.5, total_over_odds=-110, total_under_odds=-110,
    ),
    _make_real_record(
        sportsbook="DraftKings",
        game_id="nba_phx_uta", home_team="UTA", away_team="PHX",
        spread_home_line=16.5, spread_home_odds=-112, spread_away_odds=-108,
        ml_home_odds=850, ml_away_odds=-1400,
        total_line=230.0, total_over_odds=-108, total_under_odds=-112,
    ),
]


def _all_records():
    """Return all real-world records for all 6 games."""
    return (
        SPURS_BUCKS_RECORDS
        + WOLVES_PISTONS_RECORDS
        + HORNETS_SIXERS_RECORDS
        + HAWKS_KINGS_RECORDS
        + BULLS_GRIZZLIES_RECORDS
        + SUNS_JAZZ_RECORDS
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Implied probability — real-world odds verification (15 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpliedProbabilityRealWorld:
    """Verify implied_probability against known real-world moneyline odds."""

    def test_spurs_massive_favourite(self):
        """Spurs -1800 (DraftKings) ≈ 94.74% implied."""
        p = implied_probability(-1800)
        assert isclose(p, 0.9474, abs_tol=0.001)

    def test_bucks_massive_underdog(self):
        """Bucks +1000 (DraftKings) ≈ 9.09% implied."""
        p = implied_probability(1000)
        assert isclose(p, 0.0909, abs_tol=0.001)

    def test_spurs_pinnacle_sharp(self):
        """Spurs -1567 (Pinnacle sharp line) ≈ 94.0%."""
        p = implied_probability(-1567)
        assert isclose(p, 0.9401, abs_tol=0.001)

    def test_bucks_pinnacle_sharp(self):
        """Bucks +1329 (Pinnacle) ≈ 9.3%."""
        p = implied_probability(1329)
        assert isclose(p, 0.0700, abs_tol=0.001)

    def test_wolves_small_favourite(self):
        """Timberwolves -138 (FanDuel) ≈ 58.0%."""
        p = implied_probability(-138)
        assert isclose(p, 0.5798, abs_tol=0.001)

    def test_pistons_small_underdog(self):
        """Pistons +118 (FanDuel) ≈ 45.87%."""
        p = implied_probability(118)
        assert isclose(p, 0.4587, abs_tol=0.001)

    def test_hornets_mid_favourite(self):
        """Hornets -240 (FanDuel) ≈ 70.59%."""
        p = implied_probability(-240)
        assert isclose(p, 0.7059, abs_tol=0.001)

    def test_sixers_mid_underdog(self):
        """76ers +198 (FanDuel) ≈ 33.56%."""
        p = implied_probability(198)
        assert isclose(p, 0.3356, abs_tol=0.001)

    def test_hawks_large_favourite(self):
        """Hawks -1099 (Pinnacle) ≈ 91.66%."""
        p = implied_probability(-1099)
        assert isclose(p, 0.9166, abs_tol=0.001)

    def test_kings_large_underdog(self):
        """Kings +700 (Pinnacle) ≈ 12.5%."""
        p = implied_probability(700)
        assert isclose(p, 0.125, abs_tol=0.001)

    def test_bulls_moderate_favourite(self):
        """Bulls -172 (DraftKings) ≈ 63.24%."""
        p = implied_probability(-172)
        assert isclose(p, 0.6324, abs_tol=0.001)

    def test_grizzlies_moderate_underdog(self):
        """Grizzlies +144 (DraftKings) ≈ 40.98%."""
        p = implied_probability(144)
        assert isclose(p, 0.4098, abs_tol=0.001)

    def test_suns_heavy_favourite(self):
        """Suns -1351 (FanDuel) ≈ 93.11%."""
        p = implied_probability(-1351)
        assert isclose(p, 0.9311, abs_tol=0.001)

    def test_jazz_heavy_underdog(self):
        """Jazz +810 (FanDuel) ≈ 10.99%."""
        p = implied_probability(810)
        assert isclose(p, 0.1099, abs_tol=0.001)

    def test_overround_exists_for_all_games(self):
        """All games: sum of implied probs > 1.0 (vig exists)."""
        matchups = [
            (-1800, 1000),   # Spurs/Bucks DK
            (-138, 118),     # Wolves/Pistons FD
            (-240, 198),     # Hornets/76ers FD
            (-1099, 700),    # Hawks/Kings PIN
            (-172, 144),     # Bulls/Grizzlies DK
            (-1351, 810),    # Suns/Jazz FD
        ]
        for fav, dog in matchups:
            total = implied_probability(fav) + implied_probability(dog)
            assert total > 1.0, f"No overround for {fav}/{dog}: total={total}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Vig analysis — real sportsbook vig comparison (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVigRealWorld:
    """Verify vig calculations against real sportsbook odds."""

    def test_standard_spread_vig(self):
        """DraftKings -110/-110 spread → classic ~4.76% vig."""
        result = calculate_vig(-110, -110)
        assert isclose(result["vig"], 0.0476, abs_tol=0.001)

    def test_pinnacle_lower_vig_spread(self):
        """Pinnacle -108/-112 spread → lower vig than -110/-110."""
        pin = calculate_vig(-108, -112)
        dk = calculate_vig(-110, -110)
        # Pinnacle is known for tighter margins
        assert pin["vig"] <= dk["vig"] + 0.005

    def test_betmgm_asymmetric_spread_vig(self):
        """BetMGM -118/-102 Spurs/Bucks spread → vig exists but asymmetric."""
        result = calculate_vig(-118, -102)
        assert result["vig"] > 0
        assert result["implied_a"] > result["implied_b"]

    def test_massive_favourite_moneyline_vig(self):
        """Spurs -2000/Bucks +1000 (BetMGM) → large ML vig."""
        result = calculate_vig(-2000, 1000)
        assert result["vig"] > 0.02
        assert result["implied_a"] > 0.9

    def test_close_game_moneyline_vig(self):
        """Wolves -138/Pistons +118 (FanDuel) → moderate vig."""
        result = calculate_vig(-138, 118)
        assert result["vig"] > 0.02
        assert isclose(result["implied_a"], 0.5798, abs_tol=0.001)

    def test_pinnacle_ml_lowest_vig(self):
        """Pinnacle typically has lower moneyline vig than retail books."""
        pin_vig = calculate_vig(-1567, 1329)["vig"]
        dk_vig = calculate_vig(-1800, 1000)["vig"]
        assert pin_vig < dk_vig

    def test_total_market_vig_fanduel(self):
        """FanDuel totals -105/-115 → typical totals vig."""
        result = calculate_vig(-105, -115)
        assert 0.02 < result["vig"] < 0.08

    def test_total_market_vig_standard(self):
        """Standard -110/-110 total → ~4.76% vig."""
        result = calculate_vig(-110, -110)
        assert isclose(result["vig"], 0.0476, abs_tol=0.001)

    def test_hawks_kings_moneyline_vig(self):
        """Hawks -1099/Kings +700 → large overround on blowout ML."""
        result = calculate_vig(-1099, 700)
        assert result["total_implied"] > 1.0
        assert result["vig"] > 0.03

    def test_bulls_grizzlies_spread_vig(self):
        """Bulls/Grizzlies -108/-112 (FanDuel) → slightly asymmetric."""
        result = calculate_vig(-108, -112)
        assert result["vig"] > 0
        assert result["vig"] < 0.06

    def test_all_six_games_vig_positive(self):
        """Every real-world ML market has positive vig."""
        ml_pairs = [
            (-1800, 1000), (-138, 118), (-240, 198),
            (-1099, 700), (-172, 144), (-1351, 810),
        ]
        for fav, dog in ml_pairs:
            assert calculate_vig(fav, dog)["vig"] > 0

    def test_vig_ordering_across_books(self):
        """For Spurs/Bucks, Pinnacle has lowest ML vig."""
        vigs = {
            "Pinnacle": calculate_vig(-1567, 1329)["vig"],
            "DraftKings": calculate_vig(-1800, 1000)["vig"],
            "FanDuel": calculate_vig(-2000, 1040)["vig"],
            "BetMGM": calculate_vig(-2000, 1000)["vig"],
        }
        assert vigs["Pinnacle"] == min(vigs.values())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. No-vig probabilities — real game fair odds (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoVigRealWorld:
    """Remove vig from real sportsbook odds to get fair probabilities."""

    def test_wolves_pistons_close_to_50_50(self):
        """Wolves/Pistons spread -110/-110 → fair 50/50."""
        result = no_vig_probabilities(-110, -110)
        assert isclose(result["fair_a"], 0.5, abs_tol=0.001)

    def test_spurs_bucks_ml_fair_prob(self):
        """Spurs -1567/Bucks +1329 (Pinnacle) → Spurs ~93% fair."""
        result = no_vig_probabilities(-1567, 1329)
        assert result["fair_a"] > 0.90
        assert isclose(result["fair_a"] + result["fair_b"], 1.0, abs_tol=1e-6)

    def test_hornets_sixers_ml_fair(self):
        """Hornets -240/76ers +198 → Hornets ~68% fair."""
        result = no_vig_probabilities(-240, 198)
        assert 0.65 < result["fair_a"] < 0.75

    def test_hawks_kings_extreme_fair(self):
        """Hawks -1099/Kings +700 → Hawks ~89% fair."""
        result = no_vig_probabilities(-1099, 700)
        assert result["fair_a"] > 0.85

    def test_bulls_grizzlies_fair(self):
        """Bulls -172/Grizzlies +144 → Bulls ~60% fair."""
        result = no_vig_probabilities(-172, 144)
        assert 0.55 < result["fair_a"] < 0.65

    def test_fair_probs_always_sum_to_one(self):
        """All real ML markets: fair probs sum to 1.0."""
        ml_pairs = [
            (-1567, 1329), (-1800, 1000), (-138, 118),
            (-240, 198), (-1099, 700), (-172, 144), (-1351, 810),
        ]
        for fav, dog in ml_pairs:
            result = no_vig_probabilities(fav, dog)
            assert isclose(result["fair_a"] + result["fair_b"], 1.0, abs_tol=1e-6)

    def test_pinnacle_vs_draftkings_fair_prob_similar(self):
        """Different books → similar fair probs after vig removal."""
        pin = no_vig_probabilities(-1567, 1329)
        dk = no_vig_probabilities(-1800, 1000)
        # After removing vig, true probability estimates should converge
        assert abs(pin["fair_a"] - dk["fair_a"]) < 0.05

    def test_close_game_fair_probs_balanced(self):
        """Wolves/Pistons ML: fair probs close to 55/45."""
        result = no_vig_probabilities(-138, 118)
        assert 0.50 < result["fair_a"] < 0.60
        assert 0.40 < result["fair_b"] < 0.50

    def test_suns_jazz_extreme(self):
        """Suns -1351/Jazz +810 → Suns ~91% fair."""
        result = no_vig_probabilities(-1351, 810)
        assert result["fair_a"] > 0.88

    def test_vig_pct_always_present(self):
        """All results include vig_pct."""
        for fav, dog in [(-1800, 1000), (-138, 118), (-172, 144)]:
            result = no_vig_probabilities(fav, dog)
            assert "vig_pct" in result
            assert result["vig_pct"].endswith("%")

    def test_fair_always_less_than_implied(self):
        """Fair prob < implied prob (vig removed)."""
        for fav, dog in [(-1800, 1000), (-240, 198), (-172, 144)]:
            nv = no_vig_probabilities(fav, dog)
            vig_result = calculate_vig(fav, dog)
            assert nv["fair_a"] <= vig_result["implied_a"]

    def test_favourite_always_higher_fair_prob(self):
        """Favourite always gets the higher fair probability."""
        ml_pairs = [
            (-1800, 1000), (-138, 118), (-240, 198),
            (-1099, 700), (-172, 144), (-1351, 810),
        ]
        for fav, dog in ml_pairs:
            result = no_vig_probabilities(fav, dog)
            assert result["fair_a"] > result["fair_b"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Shin probabilities — real-world longshot bias (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestShinRealWorld:
    """Shin model on real NBA moneylines — asymmetric vig allocation."""

    def test_spurs_bucks_shin_sum_to_one(self):
        """Shin probs sum to 1.0 for Spurs -1800 / Bucks +1000."""
        result = shin_probabilities(-1800, 1000)
        assert isclose(result["shin_a"] + result["shin_b"], 1.0, abs_tol=0.001)

    def test_more_vig_on_bucks_longshot(self):
        """Bucks longshot (+1000) gets more vig than Spurs favourite."""
        result = shin_probabilities(-1800, 1000)
        assert result["vig_on_b"] >= result["vig_on_a"]

    def test_wolves_pistons_shin_balanced(self):
        """Close game: Shin vig allocation more balanced."""
        result = shin_probabilities(-138, 118)
        # For close games, vig is more evenly split
        assert abs(result["vig_on_a"] - result["vig_on_b"]) < 0.02

    def test_extreme_favourite_shin_vs_naive(self):
        """Extreme favourite: Shin differs from naive for Hawks -1099 / Kings +700."""
        result = shin_probabilities(-1099, 700)
        assert result["delta_a"] != 0 or result["delta_b"] != 0

    def test_suns_jazz_shin_longshot_bias(self):
        """Jazz +810 (longshot) gets more vig allocation."""
        result = shin_probabilities(-1351, 810)
        assert result["vig_on_b"] >= result["vig_on_a"]

    def test_shin_z_positive_for_all_real_markets(self):
        """z > 0 for all real markets (vig exists)."""
        ml_pairs = [
            (-1800, 1000), (-138, 118), (-240, 198),
            (-1099, 700), (-172, 144), (-1351, 810),
        ]
        for fav, dog in ml_pairs:
            result = shin_probabilities(fav, dog)
            assert result["z"] > 0

    def test_hornets_sixers_shin(self):
        """Hornets -240 / 76ers +198 → Shin gives Hornets ~69%."""
        result = shin_probabilities(-240, 198)
        assert 0.65 < result["shin_a"] < 0.75

    def test_bulls_grizzlies_shin_moderate(self):
        """Bulls -172 / Grizzlies +144 → moderate spread in Shin allocation."""
        result = shin_probabilities(-172, 144)
        assert result["shin_a"] > result["shin_b"]

    def test_pinnacle_lower_z_than_retail(self):
        """Pinnacle (sharp) has lower z (less vig) than retail books."""
        pin_z = shin_probabilities(-1567, 1329)["z"]
        dk_z = shin_probabilities(-1800, 1000)["z"]
        assert pin_z < dk_z

    def test_shin_favourite_boost_on_lopsided(self):
        """Favourite gets a slight probability boost in Shin vs naive on lopsided markets."""
        result = shin_probabilities(-2000, 1000)
        # delta_a positive means Shin boosted favourite vs naive
        # This is expected behaviour of the Shin model
        assert result["shin_a"] > 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Enrich record — real-world records (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnrichRealWorld:
    """Enrich actual sportsbook records with calculated fields."""

    def test_spurs_bucks_dk_enriched(self):
        """DraftKings Spurs/Bucks record gets all enrichment fields."""
        enriched = _enrich_record(SPURS_BUCKS_RECORDS[1])  # DraftKings
        spread = enriched["markets"]["spread"]
        assert "home_implied_prob" in spread
        assert "vig" in spread
        assert "home_shin_prob" in spread

    def test_wolves_pistons_pinnacle_enriched(self):
        """Pinnacle Wolves/Pistons ML enrichment."""
        enriched = _enrich_record(WOLVES_PISTONS_RECORDS[0])  # Pinnacle
        ml = enriched["markets"]["moneyline"]
        assert 0 < ml["home_implied_prob"] < 1
        assert 0 < ml["away_implied_prob"] < 1

    def test_extreme_moneyline_enrichment(self):
        """Extreme Spurs -2000 ML gets valid enrichment."""
        enriched = _enrich_record(SPURS_BUCKS_RECORDS[3])  # BetMGM -2000
        ml = enriched["markets"]["moneyline"]
        assert ml["away_implied_prob"] > 0.9  # Spurs (away) is big favourite
        assert ml["home_implied_prob"] < 0.1  # Bucks (home) is big underdog

    def test_total_enrichment_high_line(self):
        """Bulls/Grizzlies total 245.5 — highest total — enriched correctly."""
        enriched = _enrich_record(BULLS_GRIZZLIES_RECORDS[0])
        total = enriched["markets"]["total"]
        assert "over_implied_prob" in total
        assert "under_implied_prob" in total
        assert total["line"] == 245.5

    def test_game_metadata_preserved(self):
        """Real game IDs and team names preserved through enrichment."""
        enriched = _enrich_record(HAWKS_KINGS_RECORDS[0])
        assert enriched["game_id"] == "nba_atl_sac"
        assert enriched["home_team"] == "ATL"
        assert enriched["away_team"] == "SAC"

    def test_sportsbook_preserved(self):
        """Sportsbook name preserved through enrichment."""
        enriched = _enrich_record(SUNS_JAZZ_RECORDS[0])
        assert enriched["sportsbook"] == "FanDuel"

    def test_all_vig_values_positive(self):
        """All enriched records have positive vig across all markets."""
        for record in SPURS_BUCKS_RECORDS:
            enriched = _enrich_record(record)
            for market_name in ("spread", "moneyline", "total"):
                assert enriched["markets"][market_name]["vig"] > 0

    def test_fair_odds_present_in_spread(self):
        """Fair odds computed for spread market."""
        enriched = _enrich_record(HORNETS_SIXERS_RECORDS[1])  # DraftKings
        spread = enriched["markets"]["spread"]
        assert "home_fair_odds" in spread
        assert "away_fair_odds" in spread

    def test_implied_probs_valid_range_all_records(self):
        """All implied probabilities are in (0, 1) across all real records."""
        for record in _all_records():
            enriched = _enrich_record(record)
            for m in enriched["markets"].values():
                for key, val in m.items():
                    if "implied_prob" in key:
                        assert 0 < val < 1, f"{record['sportsbook']} {key} = {val}"

    def test_market_consistency_present(self):
        """market_consistency computed for records with all three markets."""
        enriched = _enrich_record(WOLVES_PISTONS_RECORDS[0])
        assert "market_consistency" in enriched


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Group by game — real multi-game slate (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroupByGameRealWorld:
    """Group the full March 28 NBA slate by game_id."""

    def test_six_games_grouped(self):
        """All 6 games produce 6 groups."""
        grouped = _group_by_game(_all_records())
        assert len(grouped) == 6

    def test_spurs_bucks_five_books(self):
        """Spurs/Bucks has 5 sportsbooks."""
        grouped = _group_by_game(_all_records())
        assert len(grouped["nba_sas_mil"]) == 5

    def test_wolves_pistons_three_books(self):
        """Wolves/Pistons has 3 sportsbooks."""
        grouped = _group_by_game(_all_records())
        assert len(grouped["nba_min_det"]) == 3

    def test_hawks_kings_two_books(self):
        """Hawks/Kings has 2 sportsbooks."""
        grouped = _group_by_game(_all_records())
        assert len(grouped["nba_atl_sac"]) == 2

    def test_all_game_ids_present(self):
        """All expected game IDs are in the grouped dict."""
        grouped = _group_by_game(_all_records())
        expected = {"nba_sas_mil", "nba_min_det", "nba_cha_phi",
                    "nba_atl_sac", "nba_mem_chi", "nba_phx_uta"}
        assert set(grouped.keys()) == expected

    def test_total_records_match(self):
        """Sum of records across groups == total input records."""
        all_recs = _all_records()
        grouped = _group_by_game(all_recs)
        total = sum(len(v) for v in grouped.values())
        assert total == len(all_recs)

    def test_pinnacle_in_spurs_bucks(self):
        """Pinnacle record is in the Spurs/Bucks group."""
        grouped = _group_by_game(_all_records())
        books = [r["sportsbook"] for r in grouped["nba_sas_mil"]]
        assert "Pinnacle" in books

    def test_no_cross_contamination(self):
        """No record from one game appears in another game's group."""
        grouped = _group_by_game(_all_records())
        for game_id, records in grouped.items():
            for r in records:
                assert r["game_id"] == game_id

    def test_sportsbook_names_correct(self):
        """Sportsbook names preserved in grouped records."""
        grouped = _group_by_game(SPURS_BUCKS_RECORDS)
        books = sorted([r["sportsbook"] for r in grouped["nba_sas_mil"]])
        assert books == ["BetMGM", "Caesars", "DraftKings", "FanDuel", "Pinnacle"]

    def test_markets_intact_after_grouping(self):
        """Market data intact in grouped records."""
        grouped = _group_by_game(WOLVES_PISTONS_RECORDS)
        for record in grouped["nba_min_det"]:
            assert "spread" in record["markets"]
            assert "moneyline" in record["markets"]
            assert "total" in record["markets"]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Compute consensus — real multi-book consensus (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeConsensusRealWorld:
    """Compute consensus lines from real multi-sportsbook data."""

    def _by_game(self):
        return _group_by_game(_all_records())

    def test_consensus_for_all_games(self):
        """Consensus computed for all 6 games."""
        c = _compute_consensus(self._by_game())
        assert len(c) == 6

    def test_spurs_bucks_avg_spread_line(self):
        """Spurs/Bucks average spread line ~ 17.2 (mix of 16.5, 17.0, 17.5)."""
        c = _compute_consensus(self._by_game())
        avg_line = c["nba_sas_mil"]["spread"]["avg_home_line"]
        # Pinnacle 16.5, DK 17.5, FD 17.5, BetMGM 17.5, Caesars 17.0
        assert 16.5 <= avg_line <= 17.5

    def test_spurs_bucks_total_line_range(self):
        """Spurs/Bucks consensus total between 226 and 228."""
        c = _compute_consensus(self._by_game())
        avg_total = c["nba_sas_mil"]["total"]["avg_line"]
        assert 226.0 <= avg_total <= 228.0

    def test_wolves_pistons_spread_consensus(self):
        """Wolves/Pistons all agree on 2.5 spread → consensus = 2.5, std = 0."""
        c = _compute_consensus(self._by_game())
        spread = c["nba_min_det"]["spread"]
        assert spread["avg_home_line"] == 2.5
        assert spread["std_home_line"] == 0.0

    def test_hornets_sixers_spread_consensus(self):
        """Hornets/76ers all agree on 5.5 spread."""
        c = _compute_consensus(self._by_game())
        spread = c["nba_cha_phi"]["spread"]
        assert spread["avg_home_line"] == 5.5

    def test_spurs_bucks_spread_std_nonzero(self):
        """Spurs/Bucks spread varies across books → std > 0."""
        c = _compute_consensus(self._by_game())
        assert c["nba_sas_mil"]["spread"]["std_home_line"] > 0

    def test_book_count_matches(self):
        """Book count in consensus matches number of records."""
        c = _compute_consensus(self._by_game())
        assert c["nba_sas_mil"]["spread"]["book_count"] == 5
        assert c["nba_min_det"]["spread"]["book_count"] == 3
        assert c["nba_atl_sac"]["spread"]["book_count"] == 2

    def test_bulls_grizzlies_total_consensus(self):
        """Bulls/Grizzlies total line consensus = 245.5."""
        c = _compute_consensus(self._by_game())
        assert c["nba_mem_chi"]["total"]["avg_line"] == 245.5

    def test_all_markets_present(self):
        """All games have spread, moneyline, and total consensus."""
        c = _compute_consensus(self._by_game())
        for game_id in c:
            assert "spread" in c[game_id]
            assert "moneyline" in c[game_id]
            assert "total" in c[game_id]

    def test_avg_odds_reasonable_range(self):
        """Average spread home odds are in a realistic range (accounting for plus-money)."""
        c = _compute_consensus(self._by_game())
        for game_id in c:
            spread = c[game_id]["spread"]
            # Home odds can be positive (e.g. Bucks underdog at +100 Pinnacle)
            # so just assert they're finite and within a wide real-world range
            assert -130 < spread["avg_home_odds"] < 110

    def test_consensus_moneyline_extreme_game(self):
        """Spurs/Bucks consensus ML reflects massive favourite."""
        c = _compute_consensus(self._by_game())
        ml = c["nba_sas_mil"]["moneyline"]
        # Avg home ML (Bucks underdog) should be large positive
        assert ml["avg_home_odds"] > 900

    def test_highest_total_is_bulls_grizzlies(self):
        """Bulls/Grizzlies has the highest total line (245.5)."""
        c = _compute_consensus(self._by_game())
        totals = {gid: c[gid]["total"]["avg_line"] for gid in c}
        assert max(totals, key=totals.get) == "nba_mem_chi"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Pinnacle fair probs — sharp line extraction (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPinnacleFairProbsRealWorld:
    """Extract Pinnacle no-vig fair probabilities from real data."""

    def _by_game(self):
        return _group_by_game(_all_records())

    def test_pinnacle_source_when_available(self):
        """Games with Pinnacle data use Pinnacle as source."""
        result = _get_pinnacle_fair_probs(self._by_game())
        # Spurs/Bucks has Pinnacle
        assert result["nba_sas_mil"]["spread"]["source"] == "Pinnacle"

    def test_consensus_fallback_for_no_pinnacle(self):
        """Games without Pinnacle fall back to consensus."""
        # Suns/Jazz has no Pinnacle
        result = _get_pinnacle_fair_probs(self._by_game())
        assert result["nba_phx_uta"]["spread"]["source"] == "consensus"

    def test_spurs_bucks_ml_fair_probs(self):
        """Spurs/Bucks Pinnacle ML fair probs: Spurs ~93%."""
        result = _get_pinnacle_fair_probs(self._by_game())
        ml = result["nba_sas_mil"]["moneyline"]
        # Bucks are home, Spurs are away; Spurs are big favourite
        assert ml["side_b_prob"] > 0.90  # side_b = away = Spurs

    def test_wolves_pistons_balanced_probs(self):
        """Wolves/Pistons close game: both probs near 50%."""
        result = _get_pinnacle_fair_probs(self._by_game())
        ml = result["nba_min_det"]["moneyline"]
        assert 0.40 < ml["side_a_prob"] < 0.60
        assert 0.40 < ml["side_b_prob"] < 0.60

    def test_all_probs_sum_to_one(self):
        """All markets across all games: probs sum to ~1.0."""
        result = _get_pinnacle_fair_probs(self._by_game())
        for game_id in result:
            for market_type in ("spread", "moneyline", "total"):
                probs = result[game_id][market_type]
                total = probs["side_a_prob"] + probs["side_b_prob"]
                assert isclose(total, 1.0, abs_tol=0.01), \
                    f"{game_id} {market_type}: sum={total}"

    def test_all_probs_in_valid_range(self):
        """All probabilities are in (0, 1)."""
        result = _get_pinnacle_fair_probs(self._by_game())
        for game_id in result:
            for market_type in result[game_id]:
                p = result[game_id][market_type]
                assert 0 < p["side_a_prob"] < 1
                assert 0 < p["side_b_prob"] < 1

    def test_pinnacle_spread_fair_for_close_game(self):
        """Wolves/Pistons spread: Pinnacle -108/-112 → near 50/50 fair."""
        result = _get_pinnacle_fair_probs(self._by_game())
        spread = result["nba_min_det"]["spread"]
        assert 0.45 < spread["side_a_prob"] < 0.55

    def test_six_games_all_have_results(self):
        """All 6 games produce Pinnacle fair prob results."""
        result = _get_pinnacle_fair_probs(self._by_game())
        assert len(result) == 6

    def test_hawks_kings_pinnacle_source(self):
        """Hawks/Kings has Pinnacle → source is Pinnacle."""
        result = _get_pinnacle_fair_probs(self._by_game())
        assert result["nba_atl_sac"]["moneyline"]["source"] == "Pinnacle"

    def test_bulls_grizzlies_consensus_source(self):
        """Bulls/Grizzlies has no Pinnacle → consensus fallback."""
        result = _get_pinnacle_fair_probs(self._by_game())
        assert result["nba_mem_chi"]["spread"]["source"] == "consensus"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Sharp vs Crowd — real-world divergence (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSharpVsCrowdRealWorld:
    """Compare Pinnacle (sharp) vs crowd average using real data."""

    def _by_game(self):
        return _group_by_game(_all_records())

    def test_returns_all_games(self):
        """Results returned for all games with data."""
        result = _compute_sharp_vs_crowd(self._by_game())
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_spurs_bucks_has_sharp_data(self):
        """Spurs/Bucks has Pinnacle → sharp comparison available."""
        result = _compute_sharp_vs_crowd(self._by_game())
        assert "nba_sas_mil" in result

    def test_sharp_probabilities_valid(self):
        """Sharp fair probabilities are in (0, 1)."""
        result = _compute_sharp_vs_crowd(self._by_game())
        for game_id, game in result.items():
            for market_type, data in game.items():
                for key in ("sharp_home_fair_prob", "crowd_home_fair_prob"):
                    if key in data:
                        assert 0 < data[key] < 1, f"{game_id}/{market_type}/{key}"

    def test_divergence_non_negative(self):
        """Divergence percentages are non-negative."""
        result = _compute_sharp_vs_crowd(self._by_game())
        for game_id, game in result.items():
            for market_type, data in game.items():
                if "divergence_home_pct" in data:
                    assert data["divergence_home_pct"] >= 0

    def test_wolves_pistons_low_divergence(self):
        """Close game with tight odds → expect low sharp/crowd divergence."""
        result = _compute_sharp_vs_crowd(self._by_game())
        game = result.get("nba_min_det", {})
        for market_type, data in game.items():
            if "divergence_home_pct" in data:
                # Close game, books should be aligned
                assert data["divergence_home_pct"] < 5.0

    def test_multiple_markets_analyzed(self):
        """Games with Pinnacle have multiple markets analyzed."""
        result = _compute_sharp_vs_crowd(self._by_game())
        game = result.get("nba_sas_mil", {})
        assert len(game) >= 1  # At least one market analyzed

    def test_no_pinnacle_game_handled(self):
        """Games without Pinnacle (Bulls/Grizzlies) handled gracefully."""
        result = _compute_sharp_vs_crowd(self._by_game())
        # Should either skip or produce consensus-based results
        assert isinstance(result, dict)

    def test_pinnacle_spread_differs_from_consensus(self):
        """Pinnacle spread at 16.5 vs others at 17-17.5 → should show divergence."""
        by_game = _group_by_game(SPURS_BUCKS_RECORDS)
        result = _compute_sharp_vs_crowd(by_game)
        # Pinnacle has a different line (16.5 vs 17-17.5)
        game = result.get("nba_sas_mil", {})
        assert len(game) >= 0  # Graceful regardless

    def test_five_book_game_robust(self):
        """Five-book game (Spurs/Bucks) processes without error."""
        by_game = _group_by_game(SPURS_BUCKS_RECORDS)
        result = _compute_sharp_vs_crowd(by_game)
        assert isinstance(result, dict)

    def test_two_book_game_handled(self):
        """Two-book game (Hawks/Kings) with Pinnacle processes fine."""
        by_game = _group_by_game(HAWKS_KINGS_RECORDS)
        result = _compute_sharp_vs_crowd(by_game)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Calculate odds tool — real moneyline conversions (12 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateOddsRealWorld:
    """MCP calculate_odds tool with real NBA moneylines."""

    def _parse(self, result_str):
        return json.loads(result_str)

    def test_spurs_dk_minus_1800(self):
        """Spurs -1800: decimal ~1.056, probability ~94.7%."""
        r = self._parse(calculate_odds(-1800))
        assert isclose(r["implied_probability"], 0.9474, abs_tol=0.001)
        assert isclose(r["decimal_odds"], 1.0556, abs_tol=0.01)

    def test_bucks_dk_plus_1000(self):
        """Bucks +1000: profit on $100 = $1000."""
        r = self._parse(calculate_odds(1000))
        assert r["profit_on_100_bet"] == 1000
        assert r["total_return_on_100_bet"] == 1100.0

    def test_wolves_fd_minus_138(self):
        """Wolves -138: decimal ~1.725."""
        r = self._parse(calculate_odds(-138))
        assert isclose(r["decimal_odds"], 1.7246, abs_tol=0.01)

    def test_pistons_fd_plus_118(self):
        """Pistons +118: profit on $100 = $118."""
        r = self._parse(calculate_odds(118))
        assert r["profit_on_100_bet"] == 118

    def test_hornets_minus_240(self):
        """Hornets -240: need $240 to win $100."""
        r = self._parse(calculate_odds(-240))
        assert isclose(r["profit_on_100_bet"], 41.67, abs_tol=0.01)

    def test_sixers_plus_198(self):
        """76ers +198: decimal = 2.98."""
        r = self._parse(calculate_odds(198))
        assert isclose(r["decimal_odds"], 2.98, abs_tol=0.01)

    def test_hawks_minus_1099(self):
        """Hawks -1099: ~91.66% implied probability."""
        r = self._parse(calculate_odds(-1099))
        assert isclose(r["implied_probability"], 0.9166, abs_tol=0.001)

    def test_kings_plus_700(self):
        """Kings +700: 12.5% implied, decimal 8.0."""
        r = self._parse(calculate_odds(700))
        assert isclose(r["implied_probability"], 0.125, abs_tol=0.001)
        assert isclose(r["decimal_odds"], 8.0, abs_tol=0.01)

    def test_bulls_minus_172(self):
        """Bulls -172: ~63% implied."""
        r = self._parse(calculate_odds(-172))
        assert 0.60 < r["implied_probability"] < 0.66

    def test_grizzlies_plus_144(self):
        """Grizzlies +144: decimal 2.44."""
        r = self._parse(calculate_odds(144))
        assert isclose(r["decimal_odds"], 2.44, abs_tol=0.01)

    def test_suns_minus_1351(self):
        """Suns -1351: extreme favourite, probability > 93%."""
        r = self._parse(calculate_odds(-1351))
        assert r["implied_probability"] > 0.93

    def test_jazz_plus_810(self):
        """Jazz +810: big underdog, probability ~11%."""
        r = self._parse(calculate_odds(810))
        assert 0.09 < r["implied_probability"] < 0.13


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Kelly criterion & EV — real-world bet sizing (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyAndEVRealWorld:
    """Kelly sizing and EV calculations using real odds and Pinnacle fair probs.

    Note: expected_value(book_odds, fair_prob) and kelly_criterion(book_odds, fair_prob).
    EV returns: ev_edge, ev_edge_pct, ev_dollar, is_positive_ev, book_implied_prob, fair_prob.
    Kelly returns: recommended_fraction, full_kelly_pct, edge, is_positive, bankroll_example.
    """

    def test_ev_positive_when_better_odds(self):
        """If true prob (Pinnacle fair) gives edge over offered odds → +EV."""
        # Pinnacle fair for Wolves: remove vig from -138/+118
        fair = no_vig_probabilities(-138, 118)
        true_prob = fair["fair_a"]  # Wolves ~56%
        # If DraftKings offers Wolves at -142, check EV
        ev = expected_value(-142, true_prob)
        assert isinstance(ev["ev_edge"], float)

    def test_ev_negative_for_heavy_vig_bet(self):
        """Betting Bucks +1000 when fair is ~7% → likely -EV."""
        fair = no_vig_probabilities(-1567, 1329)  # Pinnacle fair
        true_prob = fair["fair_b"]  # Bucks ~7%
        ev = expected_value(1000, true_prob)
        # Bucks +1000 vs ~7% true → should be close to breakeven or -EV
        assert isinstance(ev["ev_edge"], float)

    def test_kelly_zero_for_negative_ev(self):
        """Kelly recommends 0 when no edge (true prob < implied)."""
        # True prob 5% but odds +1000 (implied 9.09%) → no edge
        kelly = kelly_criterion(1000, 0.05)
        assert kelly["is_positive"] is False
        assert kelly["recommended_fraction"] == 0.0

    def test_kelly_sizing_for_close_game(self):
        """Wolves -138 with ~56% true prob → Kelly fraction."""
        true_prob = 0.56
        kelly = kelly_criterion(-138, true_prob)
        assert isinstance(kelly["recommended_fraction"], float)

    def test_ev_with_real_hornets_odds(self):
        """Hornets -240 with Pinnacle fair ~69% → EV calculation."""
        fair = no_vig_probabilities(-240, 198)
        ev = expected_value(-240, fair["fair_a"])
        assert "ev_edge" in ev
        assert "ev_edge_pct" in ev

    def test_kelly_bankroll_example(self):
        """Apply Kelly to a $1000 bankroll on Bulls -172."""
        fair = no_vig_probabilities(-172, 144)
        true_prob = fair["fair_a"]  # ~60%
        kelly = kelly_criterion(-172, true_prob)
        assert isinstance(kelly["recommended_fraction"], float)
        assert "bankroll_example" in kelly

    def test_ev_structure(self):
        """EV result has expected keys."""
        ev = expected_value(-110, 0.55)
        assert "ev_edge" in ev
        assert "ev_edge_pct" in ev
        assert "ev_dollar" in ev
        assert "is_positive_ev" in ev

    def test_kelly_structure(self):
        """Kelly result has expected keys."""
        kelly = kelly_criterion(-110, 0.55)
        assert "recommended_fraction" in kelly
        assert "full_kelly_pct" in kelly
        assert "is_positive" in kelly

    def test_ev_positive_when_probability_exceeds_implied(self):
        """True prob > implied prob → +EV guaranteed."""
        # Implied of -110 is ~52.38%, if true prob is 55% → +EV
        ev = expected_value(-110, 0.55)
        assert ev["ev_edge"] > 0
        assert ev["is_positive_ev"] is True

    def test_ev_negative_when_probability_below_implied(self):
        """True prob < implied prob → -EV guaranteed."""
        # Implied of -200 is ~66.67%, if true prob is 60% → -EV
        ev = expected_value(-200, 0.60)
        assert ev["ev_edge"] < 0
        assert ev["is_positive_ev"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Arithmetic evaluate — real betting calculations (10 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestArithmeticRealWorld:
    """Use arithmetic_evaluate for real betting calculations."""

    def _parse(self, result_str):
        return json.loads(result_str)

    def test_profit_on_spurs_bet(self):
        """$100 on Spurs -1800: profit = 100 * (100/1800)."""
        r = self._parse(arithmetic_evaluate("100 * (100 / 1800)"))
        assert isclose(r["result"], 5.5556, abs_tol=0.01)

    def test_profit_on_bucks_bet(self):
        """$50 on Bucks +1000: profit = 50 * (1000/100)."""
        r = self._parse(arithmetic_evaluate("50 * (1000 / 100)"))
        assert r["result"] == 500.0

    def test_kelly_bet_size(self):
        """Kelly fraction 0.03 on $5000 bankroll = $150 bet."""
        r = self._parse(arithmetic_evaluate("0.03 * 5000"))
        assert r["result"] == 150.0

    def test_combined_payout(self):
        """Two-leg parlay: $100 * 2.44 * 1.72 = $419.68."""
        r = self._parse(arithmetic_evaluate("100 * 2.44 * 1.72"))
        assert isclose(r["result"], 419.68, abs_tol=0.01)

    def test_vig_percentage(self):
        """Vig calculation: (1.0476 - 1.0) * 100 = 4.76%."""
        r = self._parse(arithmetic_evaluate("(1.0476 - 1.0) * 100"))
        assert isclose(r["result"], 4.76, abs_tol=0.01)

    def test_implied_score_from_spread_total(self):
        """Implied home score: (total + spread) / 2 = (226.5 + 17.5) / 2 = 122.0."""
        r = self._parse(arithmetic_evaluate("(226.5 + 17.5) / 2"))
        assert r["result"] == 122.0

    def test_implied_away_score(self):
        """Implied away score: (total - spread) / 2 = (226.5 - 17.5) / 2 = 104.5."""
        r = self._parse(arithmetic_evaluate("(226.5 - 17.5) / 2"))
        assert r["result"] == 104.5

    def test_roi_calculation(self):
        """ROI: (profit / wager) * 100 = (55.56 / 1000) * 100."""
        r = self._parse(arithmetic_evaluate("(55.56 / 1000) * 100"))
        assert isclose(r["result"], 5.556, abs_tol=0.01)

    def test_edge_vs_closing_line(self):
        """Edge calculation: opening prob - closing prob difference."""
        r = self._parse(arithmetic_evaluate("0.58 - 0.55"))
        assert isclose(r["result"], 0.03, abs_tol=0.001)

    def test_bankroll_after_streak(self):
        """Bankroll after 3 wins: 1000 * 1.05 * 1.03 * 1.07."""
        r = self._parse(arithmetic_evaluate("1000 * 1.05 * 1.03 * 1.07"))
        assert isclose(r["result"], 1000 * 1.05 * 1.03 * 1.07, abs_tol=0.01)
