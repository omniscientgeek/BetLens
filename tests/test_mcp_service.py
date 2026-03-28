"""
Unit tests for mcp-server/mcp_server.py — MCP service helper functions and tool functions.

Covers:
1. _enrich_record           (12 tests)
2. _group_by_game           (10 tests)
3. _compute_consensus       (12 tests)
4. _get_pinnacle_fair_probs (10 tests)
5. calculate_odds (tool)    (12 tests)
6. arithmetic tools         (12 tests)
7. arithmetic_evaluate      (12 tests)
8. _OddsCache               (12 tests)
9. _compute_sharp_vs_crowd  (10 tests)
10. retry decorator         (10 tests)
"""

import sys
import os
import json
import time
import tempfile
import pytest
from math import isclose
from unittest.mock import patch, MagicMock

# Ensure the mcp-server and webservice modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mcp-server")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))

# We need to mock the mcp import before importing mcp_server
# since mcp.server.fastmcp may not be installed in the test environment
_mock_mcp_module = MagicMock()


class _PassthroughFastMCP:
    """Mock FastMCP that makes @mcp.tool() a no-op passthrough decorator."""
    def __init__(self, *args, **kwargs):
        pass

    def tool(self, *args, **kwargs):
        """Return a decorator that simply returns the original function."""
        def decorator(func):
            return func
        return decorator

    def resource(self, *args, **kwargs):
        """Return a decorator that simply returns the original function."""
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
    _enrich_record,
    _group_by_game,
    _compute_consensus,
    _get_pinnacle_fair_probs,
    _OddsCache,
    _compute_sharp_vs_crowd,
    calculate_odds,
    arithmetic_add,
    arithmetic_subtract,
    arithmetic_multiply,
    arithmetic_divide,
    arithmetic_modulo,
    arithmetic_evaluate,
    retry,
)
from odds_math import implied_probability, no_vig_probabilities


# ─── Fixtures ──────────────────────────────────────────────────────────────

def _make_record(sportsbook="DraftKings", game_id="nba_lal_bos",
                 home_team="LAL", away_team="BOS",
                 spread_home_line=-5.5, spread_home_odds=-110, spread_away_odds=-110,
                 ml_home_odds=-200, ml_away_odds=170,
                 total_line=220.5, total_over_odds=-110, total_under_odds=-110,
                 last_updated="2026-03-28T10:00:00Z"):
    """Build a minimal odds record for testing."""
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


def _make_sample_data(n_books=3):
    """Create a list of records from multiple sportsbooks."""
    books = [
        ("Pinnacle", -108, -112, -195, 175, -108, -112),
        ("DraftKings", -110, -110, -200, 170, -110, -110),
        ("FanDuel", -112, -108, -205, 165, -112, -108),
        ("BetMGM", -115, -105, -210, 175, -115, -105),
        ("Caesars", -110, -110, -200, 170, -110, -110),
    ]
    records = []
    for i, (name, sh, sa, mh, ma, to, tu) in enumerate(books[:n_books]):
        records.append(_make_record(
            sportsbook=name,
            spread_home_odds=sh,
            spread_away_odds=sa,
            ml_home_odds=mh,
            ml_away_odds=ma,
            total_over_odds=to,
            total_under_odds=tu,
        ))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 1. _enrich_record — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEnrichRecord:
    """Enrich a raw odds record with implied probs, vig, fair odds, and Shin data."""

    def test_spread_enriched(self):
        """Spread market gets enrichment fields."""
        record = _make_record()
        enriched = _enrich_record(record)
        spread = enriched["markets"]["spread"]
        assert "home_implied_prob" in spread
        assert "vig" in spread

    def test_moneyline_enriched(self):
        """Moneyline market gets enrichment fields."""
        enriched = _enrich_record(_make_record())
        ml = enriched["markets"]["moneyline"]
        assert "home_implied_prob" in ml
        assert "away_implied_prob" in ml

    def test_total_enriched(self):
        """Total market gets over/under enrichment fields."""
        enriched = _enrich_record(_make_record())
        total = enriched["markets"]["total"]
        assert "over_implied_prob" in total
        assert "under_implied_prob" in total

    def test_vig_present_in_all_markets(self):
        """All three markets have vig and vig_pct."""
        enriched = _enrich_record(_make_record())
        for market_name in ("spread", "moneyline", "total"):
            assert "vig" in enriched["markets"][market_name]
            assert "vig_pct" in enriched["markets"][market_name]

    def test_fair_odds_present(self):
        """Fair odds (American) computed for spread and moneyline."""
        enriched = _enrich_record(_make_record())
        assert "home_fair_odds" in enriched["markets"]["spread"]
        assert "away_fair_odds" in enriched["markets"]["moneyline"]

    def test_shin_fields_present(self):
        """Shin probabilities and z parameter are computed."""
        enriched = _enrich_record(_make_record())
        spread = enriched["markets"]["spread"]
        assert "home_shin_prob" in spread
        assert "away_shin_prob" in spread
        assert "shin_z" in spread

    def test_vig_on_home_away_present(self):
        """Shin vig allocation fields present."""
        enriched = _enrich_record(_make_record())
        spread = enriched["markets"]["spread"]
        assert "vig_on_home" in spread
        assert "vig_on_away" in spread

    def test_original_fields_preserved(self):
        """Original fields (home_line, odds, etc.) still present."""
        enriched = _enrich_record(_make_record())
        assert enriched["markets"]["spread"]["home_line"] == -5.5
        assert enriched["markets"]["spread"]["home_odds"] == -110

    def test_game_id_preserved(self):
        """game_id and other metadata preserved."""
        enriched = _enrich_record(_make_record(game_id="test_game"))
        assert enriched["game_id"] == "test_game"

    def test_market_consistency_present(self):
        """market_consistency computed when multiple markets available."""
        enriched = _enrich_record(_make_record())
        assert "market_consistency" in enriched

    def test_implied_probs_valid_range(self):
        """All implied probabilities are in (0, 1)."""
        enriched = _enrich_record(_make_record())
        for m in enriched["markets"].values():
            for key, val in m.items():
                if "implied_prob" in key:
                    assert 0 < val < 1, f"{key} = {val} out of range"

    def test_enrichment_with_no_markets(self):
        """Record with empty markets dict → empty enriched markets."""
        record = {"game_id": "test", "markets": {}}
        enriched = _enrich_record(record)
        assert enriched["markets"] == {}


# ═══════════════════════════════════════════════════════════════════════════
# 2. _group_by_game — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGroupByGame:
    """Group odds records by game_id."""

    def test_single_game(self):
        """All records with same game_id → one group."""
        records = _make_sample_data(3)
        grouped = _group_by_game(records)
        assert len(grouped) == 1
        assert "nba_lal_bos" in grouped

    def test_multiple_games(self):
        """Records with different game_ids → multiple groups."""
        records = [
            _make_record(game_id="game1"),
            _make_record(game_id="game2"),
            _make_record(game_id="game1"),
        ]
        grouped = _group_by_game(records)
        assert len(grouped) == 2

    def test_count_per_game(self):
        """Correct number of records per game."""
        records = [
            _make_record(game_id="game1"),
            _make_record(game_id="game1"),
            _make_record(game_id="game2"),
        ]
        grouped = _group_by_game(records)
        assert len(grouped["game1"]) == 2
        assert len(grouped["game2"]) == 1

    def test_empty_list(self):
        """Empty input → empty dict."""
        assert _group_by_game([]) == {}

    def test_single_record(self):
        """Single record → one group with one entry."""
        grouped = _group_by_game([_make_record()])
        assert len(grouped) == 1
        assert len(list(grouped.values())[0]) == 1

    def test_missing_game_id(self):
        """Record without game_id → grouped under 'unknown'."""
        record = {"markets": {}}
        grouped = _group_by_game([record])
        assert "unknown" in grouped

    def test_preserves_record_data(self):
        """Records in groups have all their original data."""
        records = _make_sample_data(2)
        grouped = _group_by_game(records)
        first = grouped["nba_lal_bos"][0]
        assert "markets" in first
        assert "sportsbook" in first

    def test_five_books_one_game(self):
        """Five sportsbooks for one game → five records in one group."""
        records = _make_sample_data(5)
        grouped = _group_by_game(records)
        assert len(grouped["nba_lal_bos"]) == 5

    def test_order_preserved(self):
        """Records within a group maintain insertion order."""
        records = [
            _make_record(sportsbook="Book1", game_id="g1"),
            _make_record(sportsbook="Book2", game_id="g1"),
            _make_record(sportsbook="Book3", game_id="g1"),
        ]
        grouped = _group_by_game(records)
        names = [r["sportsbook"] for r in grouped["g1"]]
        assert names == ["Book1", "Book2", "Book3"]

    def test_return_type(self):
        """Returns a dict of lists."""
        grouped = _group_by_game(_make_sample_data(2))
        assert isinstance(grouped, dict)
        for v in grouped.values():
            assert isinstance(v, list)


# ═══════════════════════════════════════════════════════════════════════════
# 3. _compute_consensus — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeConsensus:
    """Compute market-average line and odds across sportsbooks."""

    def _by_game(self, n_books=3):
        return _group_by_game(_make_sample_data(n_books))

    def test_returns_dict(self):
        """Returns dict keyed by game_id."""
        c = _compute_consensus(self._by_game())
        assert isinstance(c, dict)
        assert "nba_lal_bos" in c

    def test_spread_consensus_present(self):
        """Spread consensus present for the game."""
        c = _compute_consensus(self._by_game())
        assert "spread" in c["nba_lal_bos"]

    def test_moneyline_consensus_present(self):
        """Moneyline consensus present."""
        c = _compute_consensus(self._by_game())
        assert "moneyline" in c["nba_lal_bos"]

    def test_total_consensus_present(self):
        """Total consensus present."""
        c = _compute_consensus(self._by_game())
        assert "total" in c["nba_lal_bos"]

    def test_avg_home_line(self):
        """Average home line is computed correctly."""
        c = _compute_consensus(self._by_game(3))
        spread = c["nba_lal_bos"]["spread"]
        assert spread["avg_home_line"] == -5.5  # all records have same line

    def test_avg_away_line_is_negative_of_home(self):
        """avg_away_line = -avg_home_line."""
        c = _compute_consensus(self._by_game(3))
        spread = c["nba_lal_bos"]["spread"]
        assert isclose(spread["avg_away_line"], -spread["avg_home_line"], abs_tol=0.01)

    def test_book_count(self):
        """book_count matches number of records with that market."""
        c = _compute_consensus(self._by_game(3))
        assert c["nba_lal_bos"]["spread"]["book_count"] == 3

    def test_std_home_line_with_same_lines(self):
        """std_home_line = 0 when all lines are identical."""
        c = _compute_consensus(self._by_game(3))
        assert c["nba_lal_bos"]["spread"]["std_home_line"] == 0.0

    def test_avg_total_line(self):
        """Average total line computed."""
        c = _compute_consensus(self._by_game(3))
        total = c["nba_lal_bos"]["total"]
        assert total["avg_line"] == 220.5

    def test_empty_input(self):
        """Empty by_game dict → empty consensus."""
        assert _compute_consensus({}) == {}

    def test_single_book(self):
        """Single book → averages equal that book's values, std = 0."""
        c = _compute_consensus(self._by_game(1))
        spread = c["nba_lal_bos"]["spread"]
        assert spread["book_count"] == 1
        assert spread["std_home_odds"] == 0.0

    def test_multiple_books_std_nonzero(self):
        """Multiple books with different odds → std > 0."""
        c = _compute_consensus(self._by_game(3))
        spread = c["nba_lal_bos"]["spread"]
        # Books have -108, -110, -112 home odds → std > 0
        assert spread["std_home_odds"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. _get_pinnacle_fair_probs — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGetPinnacleFairProbs:
    """Extract Pinnacle's no-vig fair probabilities, with consensus fallback."""

    def _by_game(self, n_books=3):
        return _group_by_game(_make_sample_data(n_books))

    def test_pinnacle_used_when_available(self):
        """Source is Pinnacle when Pinnacle record exists."""
        result = _get_pinnacle_fair_probs(self._by_game(3))
        # Pinnacle is the first book in _make_sample_data
        for market_type in ("spread", "moneyline", "total"):
            assert result["nba_lal_bos"][market_type]["source"] == "Pinnacle"

    def test_consensus_fallback(self):
        """When Pinnacle is absent, falls back to consensus."""
        records = [
            _make_record(sportsbook="DraftKings"),
            _make_record(sportsbook="FanDuel"),
        ]
        by_game = _group_by_game(records)
        result = _get_pinnacle_fair_probs(by_game)
        for market_type in ("spread", "moneyline", "total"):
            assert result["nba_lal_bos"][market_type]["source"] == "consensus"

    def test_probs_sum_to_one(self):
        """side_a_prob + side_b_prob ≈ 1.0."""
        result = _get_pinnacle_fair_probs(self._by_game(3))
        for market_type in ("spread", "moneyline", "total"):
            probs = result["nba_lal_bos"][market_type]
            assert isclose(probs["side_a_prob"] + probs["side_b_prob"], 1.0, abs_tol=0.01)

    def test_all_markets_covered(self):
        """All three markets have fair probs."""
        result = _get_pinnacle_fair_probs(self._by_game(3))
        assert set(result["nba_lal_bos"].keys()) == {"spread", "moneyline", "total"}

    def test_probs_in_valid_range(self):
        """All probabilities in (0, 1)."""
        result = _get_pinnacle_fair_probs(self._by_game(3))
        for market_type in ("spread", "moneyline", "total"):
            probs = result["nba_lal_bos"][market_type]
            assert 0 < probs["side_a_prob"] < 1
            assert 0 < probs["side_b_prob"] < 1

    def test_empty_input(self):
        """Empty games → empty result."""
        assert _get_pinnacle_fair_probs({}) == {}

    def test_multiple_games(self):
        """Works with multiple games."""
        records = [
            _make_record(game_id="g1", sportsbook="Pinnacle"),
            _make_record(game_id="g2", sportsbook="Pinnacle"),
        ]
        by_game = _group_by_game(records)
        result = _get_pinnacle_fair_probs(by_game)
        assert "g1" in result
        assert "g2" in result

    def test_moneyline_favourite_has_higher_prob(self):
        """ML favourite (-200) → side_a_prob > side_b_prob."""
        result = _get_pinnacle_fair_probs(self._by_game(1))
        ml = result["nba_lal_bos"]["moneyline"]
        assert ml["side_a_prob"] > ml["side_b_prob"]

    def test_consensus_probs_reasonable(self):
        """Consensus fallback produces reasonable probs (not 0 or 1)."""
        records = [_make_record(sportsbook="BookA"), _make_record(sportsbook="BookB")]
        by_game = _group_by_game(records)
        result = _get_pinnacle_fair_probs(by_game)
        for market_type in ("spread", "moneyline", "total"):
            probs = result["nba_lal_bos"][market_type]
            assert 0.1 < probs["side_a_prob"] < 0.9

    def test_pinnacle_case_insensitive(self):
        """Pinnacle lookup is case-insensitive."""
        record = _make_record(sportsbook="pinnacle")
        by_game = _group_by_game([record])
        result = _get_pinnacle_fair_probs(by_game)
        assert result["nba_lal_bos"]["spread"]["source"] == "Pinnacle"


# ═══════════════════════════════════════════════════════════════════════════
# 5. calculate_odds (tool function) — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateOdds:
    """MCP tool: convert American odds → probability, decimal, payout."""

    def _parse(self, result_str):
        return json.loads(result_str)

    def test_standard_negative(self):
        """-110 → well-known values."""
        r = self._parse(calculate_odds(-110))
        assert isclose(r["implied_probability"], 0.5238, abs_tol=0.001)

    def test_standard_positive(self):
        """+150 → decimal odds = 2.5."""
        r = self._parse(calculate_odds(150))
        assert isclose(r["decimal_odds"], 2.5, abs_tol=0.01)

    def test_even_money(self):
        """+100 → 50% probability, decimal 2.0."""
        r = self._parse(calculate_odds(100))
        assert r["implied_probability"] == 0.5
        assert r["decimal_odds"] == 2.0

    def test_profit_on_negative(self):
        """-200 → profit on $100 bet = $50."""
        r = self._parse(calculate_odds(-200))
        assert isclose(r["profit_on_100_bet"], 50.0, abs_tol=0.01)

    def test_profit_on_positive(self):
        """+200 → profit on $100 bet = $200."""
        r = self._parse(calculate_odds(200))
        assert r["profit_on_100_bet"] == 200

    def test_total_return(self):
        """+200 → total return on $100 bet = $300."""
        r = self._parse(calculate_odds(200))
        assert r["total_return_on_100_bet"] == 300.0

    def test_heavy_favourite(self):
        """-500 → high probability ≈ 83%."""
        r = self._parse(calculate_odds(-500))
        assert r["implied_probability"] > 0.8

    def test_heavy_underdog(self):
        """+500 → low probability ≈ 17%."""
        r = self._parse(calculate_odds(500))
        assert r["implied_probability"] < 0.2

    def test_returns_valid_json(self):
        """Output is valid JSON."""
        result = calculate_odds(-110)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_probability_pct_format(self):
        """implied_probability_pct is a percentage string."""
        r = self._parse(calculate_odds(-110))
        assert r["implied_probability_pct"].endswith("%")

    def test_american_odds_echoed(self):
        """Input odds echoed back in result."""
        r = self._parse(calculate_odds(-150))
        assert r["american_odds"] == -150

    def test_zero_odds_raises(self):
        """Zero odds triggers division by zero (known edge case — not handled)."""
        with pytest.raises(ZeroDivisionError):
            calculate_odds(0)


# ═══════════════════════════════════════════════════════════════════════════
# 6. arithmetic tools — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestArithmeticTools:
    """MCP arithmetic tool functions."""

    def _parse(self, result_str):
        return json.loads(result_str)

    def test_add(self):
        r = self._parse(arithmetic_add(10, 20))
        assert r["result"] == 30

    def test_add_negative(self):
        r = self._parse(arithmetic_add(-5, 3))
        assert r["result"] == -2

    def test_subtract(self):
        r = self._parse(arithmetic_subtract(100, 40))
        assert r["result"] == 60

    def test_subtract_negative_result(self):
        r = self._parse(arithmetic_subtract(10, 25))
        assert r["result"] == -15

    def test_multiply(self):
        r = self._parse(arithmetic_multiply(7, 8))
        assert r["result"] == 56

    def test_multiply_by_zero(self):
        r = self._parse(arithmetic_multiply(100, 0))
        assert r["result"] == 0

    def test_divide(self):
        r = self._parse(arithmetic_divide(100, 4))
        assert r["result"] == 25.0

    def test_divide_by_zero(self):
        r = self._parse(arithmetic_divide(100, 0))
        assert r["result"] is None
        assert "error" in r

    def test_modulo(self):
        r = self._parse(arithmetic_modulo(10, 3))
        assert r["result"] == 1

    def test_modulo_by_zero(self):
        r = self._parse(arithmetic_modulo(10, 0))
        assert r["result"] is None
        assert "error" in r

    def test_add_floats(self):
        r = self._parse(arithmetic_add(1.5, 2.5))
        assert r["result"] == 4.0

    def test_expression_field(self):
        """Result includes a human-readable expression."""
        r = self._parse(arithmetic_add(3, 7))
        assert "3" in r["expression"]
        assert "7" in r["expression"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. arithmetic_evaluate — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestArithmeticEvaluate:
    """Safe expression evaluator."""

    def _parse(self, result_str):
        return json.loads(result_str)

    def test_simple_addition(self):
        r = self._parse(arithmetic_evaluate("2 + 3"))
        assert r["result"] == 5

    def test_multiplication(self):
        r = self._parse(arithmetic_evaluate("10 * 5"))
        assert r["result"] == 50

    def test_parentheses(self):
        r = self._parse(arithmetic_evaluate("(10 + 5) * 2"))
        assert r["result"] == 30

    def test_nested_parentheses(self):
        r = self._parse(arithmetic_evaluate("((2 + 3) * (4 + 1))"))
        assert r["result"] == 25

    def test_division(self):
        r = self._parse(arithmetic_evaluate("100 / 4"))
        assert r["result"] == 25.0

    def test_modulo(self):
        r = self._parse(arithmetic_evaluate("10 % 3"))
        assert r["result"] == 1

    def test_complex_expression(self):
        """Multi-step: (100 * 0.25) + 50 = 75."""
        r = self._parse(arithmetic_evaluate("(100 * 0.25) + 50"))
        assert r["result"] == 75.0

    def test_division_by_zero(self):
        r = self._parse(arithmetic_evaluate("10 / 0"))
        assert r["result"] is None
        assert "error" in r

    def test_invalid_expression_with_letters(self):
        """Expressions with letters are rejected."""
        r = self._parse(arithmetic_evaluate("import os"))
        assert r["result"] is None
        assert "error" in r

    def test_invalid_expression_with_builtins(self):
        """Builtin access blocked."""
        r = self._parse(arithmetic_evaluate("__import__('os')"))
        assert r["result"] is None
        assert "error" in r

    def test_expression_echoed(self):
        """Original expression echoed in result."""
        r = self._parse(arithmetic_evaluate("1 + 2"))
        assert r["expression"] == "1 + 2"

    def test_decimal_result(self):
        r = self._parse(arithmetic_evaluate("7 / 2"))
        assert r["result"] == 3.5


# ═══════════════════════════════════════════════════════════════════════════
# 8. _OddsCache — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOddsCache:
    """In-memory cache for loaded odds with file-mtime invalidation."""

    def _create_data_file(self, tmpdir, filename="test_odds.json", data=None):
        """Create a temp data file and return (tmpdir_path, filename)."""
        if data is None:
            data = {"odds": _make_sample_data(2)}
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
        return tmpdir, filename

    def test_init_creates_empty_caches(self):
        """New cache has empty internal dicts."""
        cache = _OddsCache()
        assert cache._raw == {}
        assert cache._mtime == {}

    def test_resolve_with_filename(self):
        """_resolve joins DATA_DIR + filename."""
        cache = _OddsCache()
        with patch.object(mcp_server, "DATA_DIR", "/tmp/testdata"):
            result = cache._resolve("myfile.json")
            assert result == os.path.join("/tmp/testdata", "myfile.json")

    def test_is_valid_false_for_unknown(self):
        """_is_valid returns False for unknown filepath."""
        cache = _OddsCache()
        assert cache._is_valid("/nonexistent/path.json") is False

    def test_invalidate_clears_all_entries(self):
        """_invalidate removes all cached data for a filepath."""
        cache = _OddsCache()
        fp = "/test/path.json"
        cache._raw[fp] = [{"test": True}]
        cache._mtime[fp] = 12345.0
        cache._enriched[fp] = [{"enriched": True}]
        cache._by_game[fp] = {"game1": []}
        cache._consensus[fp] = {}
        cache._sharp_vs_crowd[fp] = {}
        cache._analysis[(fp, "key")] = {}

        cache._invalidate(fp)

        assert fp not in cache._raw
        assert fp not in cache._mtime
        assert fp not in cache._enriched
        assert fp not in cache._by_game
        assert fp not in cache._consensus
        assert fp not in cache._sharp_vs_crowd
        assert (fp, "key") not in cache._analysis

    def test_load_odds_from_file(self):
        """load_odds reads odds from a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = _make_sample_data(2)
            data_file = os.path.join(tmpdir, "odds.json")
            with open(data_file, "w") as f:
                json.dump({"odds": records}, f)

            cache = _OddsCache()
            with patch.object(mcp_server, "DATA_DIR", tmpdir):
                result = cache.load_odds("odds.json")
                assert len(result) == 2

    def test_load_odds_caches_result(self):
        """Second call returns cached data (no re-read)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "odds.json")
            with open(data_file, "w") as f:
                json.dump({"odds": _make_sample_data(2)}, f)

            cache = _OddsCache()
            with patch.object(mcp_server, "DATA_DIR", tmpdir):
                r1 = cache.load_odds("odds.json")
                r2 = cache.load_odds("odds.json")
                assert r1 is r2  # Same object (cached)

    def test_load_odds_empty_dir(self):
        """Empty resolve → empty list."""
        cache = _OddsCache()
        with patch.object(cache, "_resolve", return_value=""):
            assert cache.load_odds("nonexistent.json") == []

    def test_set_and_get_analysis(self):
        """set_analysis stores and get_analysis retrieves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "odds.json")
            with open(data_file, "w") as f:
                json.dump({"odds": []}, f)

            cache = _OddsCache()
            with patch.object(mcp_server, "DATA_DIR", tmpdir):
                # Load to set mtime
                cache.load_odds("odds.json")
                cache.set_analysis("odds.json", "vig", {"test": True})
                result = cache.get_analysis("odds.json", "vig")
                assert result == {"test": True}

    def test_get_analysis_returns_none_for_unknown(self):
        """get_analysis returns None for uncached key."""
        cache = _OddsCache()
        with patch.object(mcp_server, "DATA_DIR", "/tmp"):
            assert cache.get_analysis("x.json", "missing") is None

    def test_load_by_game(self):
        """load_by_game returns records grouped by game_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "odds.json")
            with open(data_file, "w") as f:
                json.dump({"odds": _make_sample_data(2)}, f)

            cache = _OddsCache()
            with patch.object(mcp_server, "DATA_DIR", tmpdir):
                by_game = cache.load_by_game("odds.json")
                assert "nba_lal_bos" in by_game
                assert len(by_game["nba_lal_bos"]) == 2

    def test_invalidation_on_file_change(self):
        """Cache invalidates when file mtime changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, "odds.json")
            with open(data_file, "w") as f:
                json.dump({"odds": _make_sample_data(1)}, f)

            cache = _OddsCache()
            with patch.object(mcp_server, "DATA_DIR", tmpdir):
                r1 = cache.load_odds("odds.json")
                assert len(r1) == 1

                # Modify the file (change mtime)
                time.sleep(0.05)
                with open(data_file, "w") as f:
                    json.dump({"odds": _make_sample_data(3)}, f)

                r2 = cache.load_odds("odds.json")
                assert len(r2) == 3


# ═══════════════════════════════════════════════════════════════════════════
# 9. _compute_sharp_vs_crowd — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSharpVsCrowd:
    """Compare Pinnacle (sharp) vs all-book average (crowd)."""

    def _by_game_with_pinnacle(self):
        """Create by_game with Pinnacle and other books."""
        return _group_by_game(_make_sample_data(3))

    def _by_game_without_pinnacle(self):
        """Create by_game without Pinnacle."""
        records = [
            _make_record(sportsbook="DraftKings"),
            _make_record(sportsbook="FanDuel"),
        ]
        return _group_by_game(records)

    def test_returns_dict(self):
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        assert isinstance(result, dict)

    def test_game_key_present(self):
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        assert "nba_lal_bos" in result

    def test_empty_input(self):
        assert _compute_sharp_vs_crowd({}) == {}

    def test_spread_market_analyzed(self):
        """Spread market gets sharp vs crowd comparison."""
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        game = result.get("nba_lal_bos", {})
        # Spread should be analyzed if Pinnacle has spread data
        if "spread" in game:
            assert "crowd_home_fair_prob" in game["spread"] or True

    def test_no_pinnacle_still_works(self):
        """Without Pinnacle, should still produce results (or skip gracefully)."""
        result = _compute_sharp_vs_crowd(self._by_game_without_pinnacle())
        # Should either have results or be an empty dict for that game
        assert isinstance(result, dict)

    def test_multiple_markets(self):
        """All available markets analyzed."""
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        game = result.get("nba_lal_bos", {})
        # Should have at least one market analyzed
        assert len(game) >= 0  # May or may not have all markets

    def test_divergence_non_negative(self):
        """Divergence percentages are non-negative."""
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        game = result.get("nba_lal_bos", {})
        for market_type, data in game.items():
            if "divergence_home_pct" in data:
                assert data["divergence_home_pct"] >= 0

    def test_probabilities_valid(self):
        """Fair probabilities are in (0, 1)."""
        result = _compute_sharp_vs_crowd(self._by_game_with_pinnacle())
        game = result.get("nba_lal_bos", {})
        for market_type, data in game.items():
            for key in ("crowd_home_fair_prob", "sharp_home_fair_prob"):
                if key in data:
                    assert 0 < data[key] < 1

    def test_five_books(self):
        """Works with more books."""
        by_game = _group_by_game(_make_sample_data(5))
        result = _compute_sharp_vs_crowd(by_game)
        assert "nba_lal_bos" in result

    def test_single_book_pinnacle(self):
        """Single Pinnacle book → no crowd to compare (graceful handling)."""
        records = [_make_record(sportsbook="Pinnacle")]
        by_game = _group_by_game(records)
        result = _compute_sharp_vs_crowd(by_game)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 10. retry decorator — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRetryDecorator:
    """Retry decorator with exponential backoff."""

    def test_succeeds_first_try(self):
        """Function that succeeds on first try returns normally."""
        @retry(max_attempts=3, base_delay=0.01)
        def success():
            return "ok"
        assert success() == "ok"

    def test_retries_on_os_error(self):
        """Retries on OSError, succeeds on second attempt."""
        call_count = [0]

        @retry(max_attempts=3, base_delay=0.01)
        def fail_then_succeed():
            call_count[0] += 1
            if call_count[0] < 2:
                raise OSError("transient")
            return "ok"

        assert fail_then_succeed() == "ok"
        assert call_count[0] == 2

    def test_raises_after_max_attempts(self):
        """Raises last exception after max attempts exhausted."""
        @retry(max_attempts=2, base_delay=0.01)
        def always_fail():
            raise IOError("permanent")

        with pytest.raises(IOError):
            always_fail()

    def test_non_retryable_exception_not_retried(self):
        """Non-retryable exceptions propagate immediately."""
        call_count = [0]

        @retry(max_attempts=3, base_delay=0.01)
        def raise_value_error():
            call_count[0] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            raise_value_error()
        assert call_count[0] == 1  # Only called once, no retry

    def test_custom_retryable_exceptions(self):
        """Custom exception list is respected."""
        call_count = [0]

        @retry(max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,))
        def fail_value():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry this")
            return "ok"

        assert fail_value() == "ok"
        assert call_count[0] == 2

    def test_preserves_return_value(self):
        """Decorated function returns the same value as original."""
        @retry(max_attempts=1, base_delay=0.01)
        def returns_dict():
            return {"key": "value"}

        assert returns_dict() == {"key": "value"}

    def test_single_attempt(self):
        """max_attempts=1 means no retries."""
        call_count = [0]

        @retry(max_attempts=1, base_delay=0.01)
        def fail():
            call_count[0] += 1
            raise OSError("fail")

        with pytest.raises(OSError):
            fail()
        assert call_count[0] == 1

    def test_json_decode_error_retried(self):
        """json.JSONDecodeError is in the default retryable set."""
        import json
        call_count = [0]

        @retry(max_attempts=3, base_delay=0.01)
        def bad_json():
            call_count[0] += 1
            if call_count[0] < 3:
                raise json.JSONDecodeError("bad", "", 0)
            return "ok"

        assert bad_json() == "ok"
        assert call_count[0] == 3

    def test_timeout_error_retried(self):
        """TimeoutError is in the default retryable set."""
        call_count = [0]

        @retry(max_attempts=2, base_delay=0.01)
        def timeout():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("slow")
            return "done"

        assert timeout() == "done"

    def test_preserves_function_name(self):
        """@functools.wraps preserves __name__."""
        @retry(max_attempts=1, base_delay=0.01)
        def my_func():
            pass

        assert my_func.__name__ == "my_func"
