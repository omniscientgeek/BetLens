"""
Unit tests covering the specific formulas and anomaly detection features
described in Requirements.pdf.

PDF Page 3 — "Odds Math You'll Need":
  1. American → Implied Probability (exact examples: -150→60%, +200→33.3%)
  2. Vig/Margin (-110/-110 → 4.76% vig)
  3. No-vig fair odds (normalize to 100%)
  4. Best line (highest payout / lowest implied probability)

PDF Page 1 — Seeded anomalies the agent must detect:
  - 2–3 stale lines (last_updated significantly older)
  - 1–2 outlier prices (odds way off-market)
  - At least 1 arbitrage opportunity

MCP tools tested (previously 0 tests each):
  - get_best_odds
  - get_worst_odds
  - find_arbitrage_opportunities
  - detect_stale_lines
  - detect_line_outliers
"""

import sys
import os
import json
import pytest
from math import isclose
from unittest.mock import patch, MagicMock

# Ensure modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp-server")))

from odds_math import implied_probability, calculate_vig, no_vig_probabilities

# ── Mock FastMCP before importing mcp_server ─────────────────────────────────
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

import mcp_server
from mcp_server import (
    _group_by_game,
    get_best_odds,
    get_worst_odds,
    find_arbitrage_opportunities,
    detect_stale_lines,
    detect_line_outliers,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _rec(sportsbook, game_id="nba_lal_bos",
         home_team="BOS", away_team="LAL",
         spread_home_line=-5.5, spread_home_odds=-110, spread_away_odds=-110,
         ml_home_odds=-200, ml_away_odds=170,
         total_line=220.0, total_over_odds=-110, total_under_odds=-110,
         last_updated="2026-03-28T18:00:00Z"):
    """Build a minimal odds record."""
    return {
        "game_id": game_id,
        "sportsbook": sportsbook,
        "sport": "NBA",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": "2026-03-28T20:00:00Z",
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


def _mock_cache(records):
    """Return a MagicMock that replaces mcp_server._cache."""
    cache = MagicMock()
    grouped = _group_by_game(records)
    cache.load_odds.return_value = records
    cache.load_by_game.return_value = grouped
    cache.get_analysis.return_value = None   # bypass cache
    cache.set_analysis.return_value = None
    return cache


# ── Fixture A: Best / Worst Odds (3 books, 1 game) ──────────────────────────

BEST_WORST_DATA = [
    _rec("Pinnacle",   ml_home_odds=-195, ml_away_odds=175,
         spread_home_odds=-108, spread_away_odds=-112,
         total_over_odds=-108, total_under_odds=-112),
    _rec("DraftKings", ml_home_odds=-200, ml_away_odds=170,
         spread_home_odds=-110, spread_away_odds=-110,
         total_over_odds=-110, total_under_odds=-110),
    _rec("FanDuel",    ml_home_odds=-180, ml_away_odds=160,
         spread_home_odds=-105, spread_away_odds=-115,
         total_over_odds=-112, total_under_odds=-108),
]


# ── Fixture B: Arbitrage (engineered arb on ML, normal spread) ───────────────

ARB_DATA = [
    _rec("BookA", game_id="nba_arb_game",
         ml_home_odds=120, ml_away_odds=110,
         spread_home_odds=-110, spread_away_odds=-110),
    _rec("BookB", game_id="nba_arb_game",
         ml_home_odds=130, ml_away_odds=105,
         spread_home_odds=-110, spread_away_odds=-110),
    _rec("BookC", game_id="nba_arb_game",
         ml_home_odds=115, ml_away_odds=108,
         spread_home_odds=-110, spread_away_odds=-110),
]


# ── Fixture C: Stale Lines (4 books, varied timestamps) ─────────────────────

STALE_DATA = [
    _rec("Fresh1",     game_id="nba_stale_game",
         last_updated="2026-03-28T18:30:00Z"),
    _rec("Fresh2",     game_id="nba_stale_game",
         last_updated="2026-03-28T18:20:00Z"),
    _rec("StaleBook",  game_id="nba_stale_game",
         last_updated="2026-03-28T08:00:00Z"),
    _rec("StaleBook2", game_id="nba_stale_game",
         last_updated="2026-03-28T11:15:00Z"),
]


# ── Fixture D: Outlier Odds (4 books, 1 outlier book) ────────────────────────

OUTLIER_DATA = [
    _rec("BookA",       game_id="nba_outlier_game",
         ml_home_odds=-140, ml_away_odds=120),
    _rec("BookB",       game_id="nba_outlier_game",
         ml_home_odds=-135, ml_away_odds=115),
    _rec("BookC",       game_id="nba_outlier_game",
         ml_home_odds=-140, ml_away_odds=120),
    _rec("OutlierBook", game_id="nba_outlier_game",
         ml_home_odds=-195, ml_away_odds=165),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestPdfOddsFormulas — pin the exact PDF page-3 examples (6 tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPdfOddsFormulas:
    """Pin the exact formulas and examples from Requirements.pdf page 3."""

    def test_negative_odds_implied_prob_pdf_example(self):
        """PDF: '-150 → 60%'.  Formula: |odds| / (|odds| + 100) = 150/250 = 0.60."""
        assert isclose(implied_probability(-150), 0.60, abs_tol=0.001)

    def test_positive_odds_implied_prob_pdf_example(self):
        """PDF: '+200 → 33.3%'.  Formula: 100 / (odds + 100) = 100/300 = 0.3333."""
        assert isclose(implied_probability(200), 0.3333, abs_tol=0.001)

    def test_vig_calculation_pdf_example(self):
        """PDF: '-110/-110 → 52.38% + 52.38% = 104.76% → 4.76% vig'."""
        # Verify each step of the PDF's worked example
        p_a = implied_probability(-110)
        p_b = implied_probability(-110)
        assert isclose(p_a, 0.5238, abs_tol=0.001)
        assert isclose(p_b, 0.5238, abs_tol=0.001)
        assert isclose(p_a + p_b, 1.0476, abs_tol=0.001)
        result = calculate_vig(-110, -110)
        assert isclose(result["vig"], 0.0476, abs_tol=0.001)

    def test_no_vig_normalize_to_100_pct(self):
        """PDF: 'Normalize implied probabilities to sum to 100%'."""
        result = no_vig_probabilities(-110, -110)
        # After normalization: 52.38/104.76 = 50%, 52.38/104.76 = 50%
        assert isclose(result["fair_a"], 0.50, abs_tol=0.001)
        assert isclose(result["fair_b"], 0.50, abs_tol=0.001)
        assert isclose(result["fair_a"] + result["fair_b"], 1.0, abs_tol=1e-6)

    def test_best_line_is_highest_payout(self):
        """PDF: 'Best line = highest payout (lowest implied probability)'.
        -180 is better than -200 because it has lower implied prob (= higher payout)."""
        p_180 = implied_probability(-180)
        p_200 = implied_probability(-200)
        assert p_180 < p_200  # lower implied prob = better for bettor
        assert -180 > -200    # higher American odds = better

    def test_vig_with_asymmetric_odds(self):
        """Additional vig example: -150/+130 → vig = (0.60 + 0.4348) - 1 ≈ 3.48%."""
        result = calculate_vig(-150, 130)
        expected_vig = (150 / 250) + (100 / 230) - 1.0  # 0.6 + 0.43478 - 1 = 0.03478
        assert isclose(result["vig"], expected_vig, abs_tol=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestGetBestOdds — 8 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetBestOdds:
    """Test get_best_odds MCP tool — 'which book offers the best payout?'"""

    def _parse(self, result_str):
        return json.loads(result_str)

    @pytest.fixture(autouse=True)
    def _patch(self):
        with patch.object(mcp_server, "_load_odds", return_value=BEST_WORST_DATA):
            yield

    def test_best_home_ml_is_least_negative(self):
        """Best home ML = -180 (FanDuel) — least negative = best payout."""
        r = self._parse(get_best_odds("nba_lal_bos", "moneyline", "home"))
        assert r["best"]["odds"] == -180
        assert r["best"]["sportsbook"] == "FanDuel"

    def test_best_away_ml_is_most_positive(self):
        """Best away ML = +175 (Pinnacle) — most positive = best payout."""
        r = self._parse(get_best_odds("nba_lal_bos", "moneyline", "away"))
        assert r["best"]["odds"] == 175
        assert r["best"]["sportsbook"] == "Pinnacle"

    def test_best_spread_home_odds(self):
        """Best home spread odds = -105 (FanDuel)."""
        r = self._parse(get_best_odds("nba_lal_bos", "spread", "home"))
        assert r["best"]["odds"] == -105
        assert r["best"]["sportsbook"] == "FanDuel"

    def test_best_total_over_odds(self):
        """Best over odds = -108 (Pinnacle)."""
        r = self._parse(get_best_odds("nba_lal_bos", "total", "over"))
        assert r["best"]["odds"] == -108
        assert r["best"]["sportsbook"] == "Pinnacle"

    def test_worst_also_reported(self):
        """Result includes worst odds alongside best."""
        r = self._parse(get_best_odds("nba_lal_bos", "moneyline", "home"))
        assert r["worst"]["odds"] == -200
        assert r["worst"]["sportsbook"] == "DraftKings"

    def test_all_books_sorted_best_first(self):
        """all_books list is sorted descending by odds (best first)."""
        r = self._parse(get_best_odds("nba_lal_bos", "moneyline", "home"))
        odds_list = [c["odds"] for c in r["all_books"]]
        assert odds_list == sorted(odds_list, reverse=True)

    def test_invalid_game_id_returns_error(self):
        """Unknown game_id produces error response."""
        r = self._parse(get_best_odds("nonexistent", "moneyline", "home"))
        assert "error" in r

    def test_edge_vs_average_string(self):
        """edge_vs_average is a percentage string."""
        r = self._parse(get_best_odds("nba_lal_bos", "moneyline", "home"))
        assert r["edge_vs_average"].endswith("%")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestGetWorstOdds — 7 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetWorstOdds:
    """Test get_worst_odds MCP tool — 'which book to avoid?'"""

    def _parse(self, result_str):
        return json.loads(result_str)

    @pytest.fixture(autouse=True)
    def _patch(self):
        with patch.object(mcp_server, "_load_odds", return_value=BEST_WORST_DATA):
            yield

    def test_worst_home_ml_is_most_negative(self):
        """Worst home ML = -200 (DraftKings) — most negative = worst payout."""
        r = self._parse(get_worst_odds("nba_lal_bos", "moneyline", "home"))
        assert r["worst"]["odds"] == -200
        assert r["worst"]["sportsbook"] == "DraftKings"

    def test_worst_away_ml_is_least_positive(self):
        """Worst away ML = +160 (FanDuel) — least positive = worst payout."""
        r = self._parse(get_worst_odds("nba_lal_bos", "moneyline", "away"))
        assert r["worst"]["odds"] == 160
        assert r["worst"]["sportsbook"] == "FanDuel"

    def test_worst_spread_away_odds(self):
        """Worst away spread odds = -115 (FanDuel)."""
        r = self._parse(get_worst_odds("nba_lal_bos", "spread", "away"))
        assert r["worst"]["odds"] == -115

    def test_best_also_reported(self):
        """Result includes best odds alongside worst."""
        r = self._parse(get_worst_odds("nba_lal_bos", "moneyline", "home"))
        assert r["best"]["odds"] == -180

    def test_spread_between_best_worst(self):
        """spread_between_best_worst = best - worst."""
        r = self._parse(get_worst_odds("nba_lal_bos", "moneyline", "home"))
        assert r["spread_between_best_worst"] == r["best"]["odds"] - r["worst"]["odds"]

    def test_all_books_sorted_worst_first(self):
        """all_books_worst_to_best sorted ascending."""
        r = self._parse(get_worst_odds("nba_lal_bos", "moneyline", "home"))
        odds_list = [c["odds"] for c in r["all_books_worst_to_best"]]
        assert odds_list == sorted(odds_list)

    def test_invalid_game_id_returns_error(self):
        """Unknown game_id produces error."""
        r = self._parse(get_worst_odds("nonexistent", "moneyline", "home"))
        assert "error" in r


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestFindArbitrageOpportunities — 8 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindArbitrageOpportunities:
    """Test find_arbitrage_opportunities MCP tool.

    Fixture B has engineered arb on moneyline:
      Best home = BookB +130 → implied = 100/230 = 43.48%
      Best away = BookA +110 → implied = 100/210 = 47.62%
      Combined = 91.10% < 100% → arb profit ≈ 8.9%
    """

    def _parse(self, result_str):
        return json.loads(result_str)

    @pytest.fixture(autouse=True)
    def _patch(self):
        cache = _mock_cache(ARB_DATA)
        with patch.object(mcp_server, "_load_odds", return_value=ARB_DATA), \
             patch.object(mcp_server, "_cache", cache):
            yield

    def test_arb_detected(self):
        """At least one arbitrage opportunity found."""
        r = self._parse(find_arbitrage_opportunities())
        assert r["count"] >= 1

    def test_arb_profit_pct_correct(self):
        """Profit percentage matches manual calculation."""
        r = self._parse(find_arbitrage_opportunities())
        # Find the moneyline arb
        ml_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "moneyline"]
        assert len(ml_arbs) >= 1
        arb = ml_arbs[0]
        # Best home +130: implied = 100/230 = 0.43478
        # Best away +110: implied = 100/210 = 0.47619
        # Combined = 0.91097, profit = 8.903%
        assert isclose(arb["profit_pct"], 8.903, abs_tol=0.1)

    def test_arb_combined_implied_under_one(self):
        """combined_implied < 1.0 (definition of arbitrage)."""
        r = self._parse(find_arbitrage_opportunities())
        ml_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "moneyline"]
        assert ml_arbs[0]["combined_implied"] < 1.0

    def test_arb_sides_different_books(self):
        """Arb sides reference different sportsbooks."""
        r = self._parse(find_arbitrage_opportunities())
        ml_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "moneyline"]
        arb = ml_arbs[0]
        assert arb["side_a"]["sportsbook"] != arb["side_b"]["sportsbook"]

    def test_arb_best_odds_selected(self):
        """Arb picks the best odds from each side."""
        r = self._parse(find_arbitrage_opportunities())
        ml_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "moneyline"]
        arb = ml_arbs[0]
        # Best home = BookB +130
        assert arb["side_a"]["odds"] == 130
        assert arb["side_a"]["sportsbook"] == "BookB"
        # Best away = BookA +110
        assert arb["side_b"]["odds"] == 110
        assert arb["side_b"]["sportsbook"] == "BookA"

    def test_no_arb_on_normal_spread(self):
        """Standard -110/-110 spread has no arbitrage."""
        r = self._parse(find_arbitrage_opportunities())
        spread_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "spread"]
        # -110/-110 combined = 52.38+52.38 = 104.76% > 100%, no arb
        assert len(spread_arbs) == 0

    def test_min_profit_filter(self):
        """min_profit_pct=10.0 filters out the ~8.9% arb."""
        r = self._parse(find_arbitrage_opportunities(min_profit_pct=10.0))
        assert r["count"] == 0

    def test_arb_context_string(self):
        """Context string contains 'ARB:' marker."""
        r = self._parse(find_arbitrage_opportunities())
        ml_arbs = [a for a in r["arbitrage_opportunities"] if a["market_type"] == "moneyline"]
        assert "ARB:" in ml_arbs[0]["context"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestDetectStaleLines — 7 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectStaleLines:
    """Test detect_stale_lines MCP tool.

    Fixture C timestamps:
      Fresh1:     18:30 (newest)
      Fresh2:     18:20 (10 min behind)
      StaleBook:  08:00 (630 min behind)
      StaleBook2: 11:15 (435 min behind)
    """

    def _parse(self, result_str):
        return json.loads(result_str)

    @pytest.fixture(autouse=True)
    def _patch(self):
        cache = _mock_cache(STALE_DATA)
        with patch.object(mcp_server, "_load_odds", return_value=STALE_DATA), \
             patch.object(mcp_server, "_cache", cache):
            yield

    def test_stale_lines_detected(self):
        """Default 30-min threshold detects 2 stale lines."""
        r = self._parse(detect_stale_lines())
        assert r["count"] == 2

    def test_stalest_line_first(self):
        """Results sorted by staleness descending — StaleBook (630 min) first."""
        r = self._parse(detect_stale_lines())
        assert r["stale_lines"][0]["sportsbook"] == "StaleBook"
        assert r["stale_lines"][1]["sportsbook"] == "StaleBook2"

    def test_staleness_minutes_correct(self):
        """StaleBook is 630 minutes behind newest (18:30 - 08:00)."""
        r = self._parse(detect_stale_lines())
        assert isclose(r["stale_lines"][0]["staleness_minutes"], 630, abs_tol=1)

    def test_stalebook2_minutes_correct(self):
        """StaleBook2 is 435 minutes behind newest (18:30 - 11:15)."""
        r = self._parse(detect_stale_lines())
        assert isclose(r["stale_lines"][1]["staleness_minutes"], 435, abs_tol=1)

    def test_fresh_lines_not_flagged(self):
        """Fresh1 and Fresh2 do not appear in stale list."""
        r = self._parse(detect_stale_lines())
        stale_books = [s["sportsbook"] for s in r["stale_lines"]]
        assert "Fresh1" not in stale_books
        assert "Fresh2" not in stale_books

    def test_custom_threshold_500(self):
        """threshold=500 only flags StaleBook (630 min), not StaleBook2 (435 min)."""
        r = self._parse(detect_stale_lines(stale_threshold_minutes=500))
        assert r["count"] == 1
        assert r["stale_lines"][0]["sportsbook"] == "StaleBook"

    def test_stale_record_contains_markets(self):
        """Each stale entry includes the full markets data."""
        r = self._parse(detect_stale_lines())
        for entry in r["stale_lines"]:
            assert "markets" in entry
            assert "spread" in entry["markets"]
            assert "moneyline" in entry["markets"]
            assert "total" in entry["markets"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestDetectLineOutliers — 6 tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectLineOutliers:
    """Test detect_line_outliers MCP tool.

    Fixture D: 4 books, OutlierBook has ML home=-195 (avg≈-152.5, dev≈42.5)
    and ML away=+165 (avg≈130, dev≈35).
    """

    def _parse(self, result_str):
        return json.loads(result_str)

    @pytest.fixture(autouse=True)
    def _patch(self):
        cache = _mock_cache(OUTLIER_DATA)
        with patch.object(mcp_server, "_load_odds", return_value=OUTLIER_DATA), \
             patch.object(mcp_server, "_cache", cache):
            yield

    def test_outlier_detected(self):
        """At least one outlier found with default threshold=15."""
        r = self._parse(detect_line_outliers())
        assert r["count"] >= 1

    def test_outlier_sportsbook_correct(self):
        """OutlierBook is flagged as the outlier."""
        r = self._parse(detect_line_outliers())
        flagged_books = [o["sportsbook"] for o in r["outliers"]]
        assert "OutlierBook" in flagged_books

    def test_home_ml_outlier_deviation(self):
        """OutlierBook home ML: -195 vs avg -152.5 → deviation ≈ 42.5."""
        r = self._parse(detect_line_outliers())
        home_outliers = [o for o in r["outliers"]
                         if o["sportsbook"] == "OutlierBook"
                         and o.get("side") == "home"
                         and o.get("market_type") == "moneyline"]
        assert len(home_outliers) >= 1
        assert isclose(home_outliers[0]["deviation"], 42.5, abs_tol=1)

    def test_outlier_direction_worse_for_bettor(self):
        """OutlierBook home ML -195 < avg -152.5 → worse_for_bettor."""
        r = self._parse(detect_line_outliers())
        home_outliers = [o for o in r["outliers"]
                         if o["sportsbook"] == "OutlierBook"
                         and o.get("side") == "home"
                         and o.get("market_type") == "moneyline"]
        assert home_outliers[0]["direction"] == "worse_for_bettor"

    def test_high_threshold_no_outliers(self):
        """threshold=100 → no outliers (max dev ≈ 42.5)."""
        r = self._parse(detect_line_outliers(threshold_odds=100))
        # Only odds outliers should be filtered; line outliers use a separate >= 1.0 check
        odds_outliers = [o for o in r["outliers"] if o.get("type") != "line_outlier"]
        assert len(odds_outliers) == 0

    def test_both_home_and_away_ml_flagged(self):
        """OutlierBook is an outlier on BOTH home (-195) and away (+165) ML."""
        r = self._parse(detect_line_outliers())
        outlier_entries = [o for o in r["outliers"]
                           if o["sportsbook"] == "OutlierBook"
                           and o.get("market_type") == "moneyline"]
        sides_flagged = {o["side"] for o in outlier_entries}
        assert "home" in sides_flagged
        assert "away" in sides_flagged
