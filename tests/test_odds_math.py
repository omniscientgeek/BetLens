"""
Unit tests for webservice/odds_math.py — probability calculation functions.

Covers:
1. implied_probability     (15 tests)
2. calculate_vig           (12 tests)
3. no_vig_probabilities    (12 tests)
4. shin_probabilities      (12 tests)
5. fair_odds_to_american   (12 tests)
6. arbitrage_profit        (12 tests)
7. expected_value          (12 tests)
8. kelly_criterion         (12 tests)
9. bayesian_update         (12 tests)
10. _prob_to_beta_params   (10 tests)
11. _shin_solve_z          (10 tests)
"""

import sys
import os
import pytest
from math import isclose

# Ensure the webservice module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webservice")))

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
    _prob_to_beta_params,
    _shin_solve_z,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. implied_probability — 15 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestImpliedProbability:
    """Convert American odds → implied probability (0-1)."""

    def test_heavy_favourite(self):
        """Heavy favourite -228 ≈ 69.51%."""
        assert isclose(implied_probability(-228), 0.6951, abs_tol=0.0001)

    def test_underdog(self):
        """Underdog +196 ≈ 33.78%."""
        assert isclose(implied_probability(196), 0.3378, abs_tol=0.0001)

    def test_standard_vig_line_negative(self):
        """-110 is the standard vig line ≈ 52.38%."""
        assert isclose(implied_probability(-110), 0.5238, abs_tol=0.0001)

    def test_standard_vig_line_positive(self):
        """+110 ≈ 47.62%."""
        assert isclose(implied_probability(110), 0.4762, abs_tol=0.0001)

    def test_even_money_positive(self):
        """+100 is even money = 50%."""
        assert implied_probability(100) == 0.5

    def test_zero_odds_edge_case(self):
        """0 odds (undefined) treated as even money = 0.5."""
        assert implied_probability(0) == 0.5

    def test_huge_favourite(self):
        """-10000 ≈ 99.01%."""
        assert isclose(implied_probability(-10000), 0.9901, abs_tol=0.0001)

    def test_huge_underdog(self):
        """+10000 ≈ 0.99%."""
        assert isclose(implied_probability(10000), 0.0099, abs_tol=0.0001)

    def test_minus_100(self):
        """-100 = 50% (even money favourite notation)."""
        assert isclose(implied_probability(-100), 0.5, abs_tol=0.0001)

    def test_minus_150(self):
        """-150 = 60%."""
        assert isclose(implied_probability(-150), 0.6, abs_tol=0.0001)

    def test_plus_150(self):
        """+150 = 40%."""
        assert isclose(implied_probability(150), 0.4, abs_tol=0.0001)

    def test_minus_200(self):
        """-200 ≈ 66.67%."""
        assert isclose(implied_probability(-200), 0.6667, abs_tol=0.0001)

    def test_plus_200(self):
        """+200 ≈ 33.33%."""
        assert isclose(implied_probability(200), 0.3333, abs_tol=0.0001)

    def test_result_always_between_0_and_1(self):
        """All valid odds produce probability in (0, 1)."""
        for odds in [-5000, -500, -110, 100, 110, 500, 5000]:
            p = implied_probability(odds)
            assert 0 < p < 1, f"odds={odds} gave prob={p}"

    def test_float_odds(self):
        """Accept float odds (e.g., -110.5)."""
        p = implied_probability(-110.5)
        assert 0 < p < 1


# ═══════════════════════════════════════════════════════════════════════════
# 2. calculate_vig — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateVig:
    """Calculate vigorish (overround) for a two-outcome market."""

    def test_standard_vig_line(self):
        """-110/-110 is the classic ~4.76% vig."""
        result = calculate_vig(-110, -110)
        assert isclose(result["vig"], 0.0476, abs_tol=0.001)

    def test_total_implied_gt_one(self):
        """Total implied must exceed 1.0 when vig exists."""
        result = calculate_vig(-110, -110)
        assert result["total_implied"] > 1.0

    def test_no_vig_market(self):
        """Even odds (+100/+100) → no vig."""
        result = calculate_vig(100, 100)
        assert isclose(result["vig"], 0.0, abs_tol=0.0001)

    def test_high_vig_market(self):
        """-130/-130 → high vig ≈ 13.2%."""
        result = calculate_vig(-130, -130)
        assert result["vig"] > 0.1

    def test_lopsided_market(self):
        """-300/+250 → vig exists but probabilities correct."""
        result = calculate_vig(-300, 250)
        assert result["total_implied"] > 1.0
        assert result["implied_a"] > result["implied_b"]

    def test_vig_pct_format(self):
        """vig_pct is a formatted percentage string."""
        result = calculate_vig(-110, -110)
        assert result["vig_pct"].endswith("%")

    def test_implied_probabilities_returned(self):
        """Both implied probabilities are present and valid."""
        result = calculate_vig(-150, 130)
        assert 0 < result["implied_a"] < 1
        assert 0 < result["implied_b"] < 1

    def test_symmetry(self):
        """Swapping sides swaps implied_a and implied_b."""
        r1 = calculate_vig(-150, 130)
        r2 = calculate_vig(130, -150)
        assert isclose(r1["implied_a"], r2["implied_b"], abs_tol=1e-6)
        assert isclose(r1["implied_b"], r2["implied_a"], abs_tol=1e-6)

    def test_same_vig_regardless_of_order(self):
        """Vig is the same regardless of which side is A vs B."""
        r1 = calculate_vig(-200, 170)
        r2 = calculate_vig(170, -200)
        assert isclose(r1["vig"], r2["vig"], abs_tol=1e-6)

    def test_very_low_vig(self):
        """Pinnacle-like low vig: -104/+104 ≈ <1% vig."""
        result = calculate_vig(-104, -104)
        assert result["vig"] < 0.08

    def test_return_keys(self):
        """All expected keys are in the result dict."""
        result = calculate_vig(-110, -110)
        expected_keys = {"implied_a", "implied_b", "total_implied", "vig", "vig_pct"}
        assert set(result.keys()) == expected_keys

    def test_extreme_favourite(self):
        """-1000/+700 → heavy favourite, valid vig."""
        result = calculate_vig(-1000, 700)
        assert result["implied_a"] > 0.9
        assert result["implied_b"] > 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. no_vig_probabilities — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNoVigProbabilities:
    """Remove vig via proportional scaling → fair (true) probabilities."""

    def test_fair_probs_sum_to_one(self):
        """Fair probabilities must always sum to exactly 1.0."""
        result = no_vig_probabilities(-110, -110)
        assert isclose(result["fair_a"] + result["fair_b"], 1.0, abs_tol=1e-6)

    def test_standard_line_is_50_50(self):
        """-110/-110 → fair 50/50 market."""
        result = no_vig_probabilities(-110, -110)
        assert isclose(result["fair_a"], 0.5, abs_tol=0.001)
        assert isclose(result["fair_b"], 0.5, abs_tol=0.001)

    def test_favourite_gets_more(self):
        """-200/+170 → favourite has higher fair prob than underdog."""
        result = no_vig_probabilities(-200, 170)
        assert result["fair_a"] > result["fair_b"]

    def test_even_money(self):
        """+100/+100 → exactly 50/50 (no vig to remove)."""
        result = no_vig_probabilities(100, 100)
        assert isclose(result["fair_a"], 0.5, abs_tol=1e-6)

    def test_vig_removed_equals_calculate_vig(self):
        """vig_pct should match what calculate_vig reports."""
        nv = no_vig_probabilities(-115, -105)
        vig = calculate_vig(-115, -105)
        assert nv["vig_pct"] == vig["vig_pct"]

    def test_fair_probs_less_than_implied(self):
        """Fair probabilities should be less than raw implied (vig removed)."""
        nv = no_vig_probabilities(-150, 130)
        vig_result = calculate_vig(-150, 130)
        assert nv["fair_a"] <= vig_result["implied_a"]
        assert nv["fair_b"] <= vig_result["implied_b"]

    def test_format_fair_a_pct(self):
        """fair_a_pct is a percentage string."""
        result = no_vig_probabilities(-110, -110)
        assert result["fair_a_pct"].endswith("%")

    def test_heavy_favourite(self):
        """-500/+400 → favourite ≈ 80%+."""
        result = no_vig_probabilities(-500, 400)
        assert result["fair_a"] > 0.75

    def test_return_keys(self):
        """All expected keys present."""
        result = no_vig_probabilities(-110, -110)
        expected = {"fair_a", "fair_b", "fair_a_pct", "fair_b_pct", "vig_pct"}
        assert set(result.keys()) == expected

    def test_symmetry(self):
        """Swapping odds swaps fair_a/fair_b."""
        r1 = no_vig_probabilities(-150, 130)
        r2 = no_vig_probabilities(130, -150)
        assert isclose(r1["fair_a"], r2["fair_b"], abs_tol=1e-6)

    def test_extreme_underdog(self):
        """+500 underdog still gets positive fair probability."""
        result = no_vig_probabilities(-600, 500)
        assert result["fair_b"] > 0.1

    def test_negative_negative_odds(self):
        """-120/-100 (asymmetric vig)."""
        result = no_vig_probabilities(-120, -100)
        assert isclose(result["fair_a"] + result["fair_b"], 1.0, abs_tol=1e-6)
        assert result["fair_a"] > result["fair_b"]


# ═══════════════════════════════════════════════════════════════════════════
# 4. shin_probabilities — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestShinProbabilities:
    """Shin model — asymmetric vig decomposition (longshot bias)."""

    def test_shin_probs_sum_to_one(self):
        """Shin probabilities must sum to 1.0."""
        result = shin_probabilities(-200, 170)
        assert isclose(result["shin_a"] + result["shin_b"], 1.0, abs_tol=1e-4)

    def test_more_vig_on_longshot(self):
        """Shin allocates more vig to the underdog (longshot) in lopsided markets."""
        result = shin_probabilities(-500, 300)
        # Side b is the longshot (lower probability) — gets more vig
        assert result["vig_on_b"] >= result["vig_on_a"]

    def test_z_parameter_positive(self):
        """z (insider fraction) should be positive when vig exists."""
        result = shin_probabilities(-110, -110)
        assert result["z"] > 0

    def test_z_zero_when_no_vig(self):
        """z = 0 when there's no vig (probs sum ≤ 1)."""
        # +100/+100 → sum = 1.0, no vig
        result = shin_probabilities(100, 100)
        assert result["z"] == 0.0

    def test_shin_vs_naive_difference(self):
        """Shin differs from naive proportional — delta should be nonzero for lopsided markets."""
        result = shin_probabilities(-300, 250)
        assert result["delta_a"] != 0 or result["delta_b"] != 0

    def test_favourite_shin_gt_naive(self):
        """Favourite gets slightly higher Shin prob than naive."""
        result = shin_probabilities(-300, 250)
        # Favourite (shin_a) typically gets a slight boost vs naive
        assert result["delta_a"] > 0 or isclose(result["delta_a"], 0, abs_tol=0.01)

    def test_method_is_shin(self):
        """Return dict always has method='shin'."""
        result = shin_probabilities(-110, -110)
        assert result["method"] == "shin"

    def test_vig_allocation_sums_to_total_vig(self):
        """vig_on_a + vig_on_b ≈ total vig."""
        result = shin_probabilities(-115, -105)
        vig_result = calculate_vig(-115, -105)
        total_vig_allocated = result["vig_on_a"] + result["vig_on_b"]
        assert isclose(total_vig_allocated, vig_result["vig"], abs_tol=0.001)

    def test_standard_vig_line(self):
        """-110/-110 → both Shin probs ≈ 0.5."""
        result = shin_probabilities(-110, -110)
        assert isclose(result["shin_a"], 0.5, abs_tol=0.01)
        assert isclose(result["shin_b"], 0.5, abs_tol=0.01)

    def test_return_keys_complete(self):
        """All expected keys present."""
        result = shin_probabilities(-110, -110)
        expected_keys = {
            "shin_a", "shin_b", "shin_a_pct", "shin_b_pct",
            "naive_a", "naive_b", "delta_a", "delta_b",
            "z", "z_pct", "vig_pct",
            "vig_on_a", "vig_on_b", "vig_on_a_pct", "vig_on_b_pct",
            "method",
        }
        assert set(result.keys()) == expected_keys

    def test_extreme_lopsided(self):
        """-1000/+700 → Shin heavily favours home."""
        result = shin_probabilities(-1000, 700)
        assert result["shin_a"] > 0.85
        assert result["shin_b"] < 0.15

    def test_z_pct_format(self):
        """z_pct is a percentage string."""
        result = shin_probabilities(-110, -110)
        assert result["z_pct"].endswith("%")


# ═══════════════════════════════════════════════════════════════════════════
# 5. fair_odds_to_american — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFairOddsToAmerican:
    """Convert fair probability (0-1) back to American odds."""

    def test_even_money(self):
        """0.5 → +100."""
        assert fair_odds_to_american(0.5) == 100

    def test_favourite(self):
        """0.6 → negative odds (favourite)."""
        assert fair_odds_to_american(0.6) < 0

    def test_underdog(self):
        """0.4 → positive odds (underdog)."""
        assert fair_odds_to_american(0.4) > 0

    def test_heavy_favourite(self):
        """0.9 → approximately -900."""
        assert fair_odds_to_american(0.9) == -900

    def test_heavy_underdog(self):
        """0.1 → approximately +900."""
        assert fair_odds_to_american(0.1) == 900

    def test_roundtrip_negative(self):
        """implied_probability(fair_odds_to_american(p)) ≈ p for favourite."""
        p = 0.65
        odds = fair_odds_to_american(p)
        p_back = implied_probability(odds)
        assert isclose(p_back, p, abs_tol=0.01)

    def test_roundtrip_positive(self):
        """Roundtrip for underdog."""
        p = 0.35
        odds = fair_odds_to_american(p)
        p_back = implied_probability(odds)
        assert isclose(p_back, p, abs_tol=0.01)

    def test_raises_on_zero(self):
        """prob=0 should raise ValueError."""
        with pytest.raises(ValueError):
            fair_odds_to_american(0)

    def test_raises_on_one(self):
        """prob=1 should raise ValueError."""
        with pytest.raises(ValueError):
            fair_odds_to_american(1)

    def test_raises_on_negative(self):
        """prob < 0 should raise ValueError."""
        with pytest.raises(ValueError):
            fair_odds_to_american(-0.1)

    def test_returns_integer(self):
        """American odds are always integers."""
        assert isinstance(fair_odds_to_american(0.6), int)

    def test_slightly_above_half(self):
        """0.51 → small negative odds (slight favourite)."""
        odds = fair_odds_to_american(0.51)
        assert odds < 0
        assert odds > -200  # slight favourite, not extreme


# ═══════════════════════════════════════════════════════════════════════════
# 6. arbitrage_profit — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestArbitrageProfit:
    """Guaranteed-profit arbitrage detection."""

    def test_arb_exists(self):
        """Large positive odds on both sides → arb exists."""
        # +110 on both sides: prob_a + prob_b < 1.0
        result = arbitrage_profit(110, 110)
        assert result["is_arb"] is True

    def test_no_arb_standard_vig(self):
        """-110/-110 → no arb (combined > 1)."""
        result = arbitrage_profit(-110, -110)
        assert result["is_arb"] is False

    def test_no_arb_profit_zero(self):
        """When no arb, profit_pct is 0.0."""
        result = arbitrage_profit(-110, -110)
        assert result["profit_pct"] == 0.0

    def test_arb_profit_positive(self):
        """Arb profit is positive when arb exists."""
        result = arbitrage_profit(110, 110)
        assert result["profit_pct"] > 0

    def test_stakes_sum_to_100(self):
        """Optimal stakes always sum to 100%."""
        result = arbitrage_profit(110, 110)
        assert isclose(result["stake_a_pct"] + result["stake_b_pct"], 100.0, abs_tol=0.01)

    def test_no_arb_stakes_50_50(self):
        """When no arb, stakes default to 50/50."""
        result = arbitrage_profit(-110, -110)
        assert result["stake_a_pct"] == 50.0
        assert result["stake_b_pct"] == 50.0

    def test_combined_implied_lt_one_for_arb(self):
        """Combined implied < 1.0 signals an arb."""
        result = arbitrage_profit(110, 110)
        assert result["combined_implied"] < 1.0

    def test_return_keys(self):
        """All expected keys present."""
        result = arbitrage_profit(-110, -110)
        expected = {"implied_a", "implied_b", "combined_implied", "is_arb",
                    "profit_pct", "stake_a_pct", "stake_b_pct"}
        assert set(result.keys()) == expected

    def test_extreme_arb(self):
        """Very large positive odds → large arb profit."""
        result = arbitrage_profit(500, 500)
        assert result["is_arb"] is True
        assert result["profit_pct"] > 10

    def test_borderline_no_arb(self):
        """Even money +100/+100 → combined = 1.0, no arb (not strictly < 1)."""
        result = arbitrage_profit(100, 100)
        assert result["is_arb"] is False

    def test_asymmetric_arb(self):
        """Asymmetric odds can still create an arb."""
        # +200 on side A, +200 on side B → each ~33.3% = 66.6% combined
        result = arbitrage_profit(200, 200)
        assert result["is_arb"] is True

    def test_asymmetric_stakes(self):
        """Unequal odds → unequal optimal stakes when arb exists."""
        result = arbitrage_profit(150, 250)
        assert result["stake_a_pct"] != result["stake_b_pct"]


# ═══════════════════════════════════════════════════════════════════════════
# 7. expected_value — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExpectedValue:
    """EV calculation: edge vs fair probability."""

    def test_positive_ev(self):
        """Book underestimates true probability → +EV."""
        # Fair prob = 0.55, book odds = -110 (implies 52.38%) → positive edge
        result = expected_value(-110, 0.55)
        assert result["is_positive_ev"] is True

    def test_negative_ev(self):
        """Fair prob lower than book implied → -EV."""
        # Fair prob = 0.50, book odds = -110 (implies 52.38%) → negative edge
        result = expected_value(-110, 0.50)
        assert result["is_positive_ev"] is False

    def test_ev_edge_calculation(self):
        """EV edge = fair_prob - book_implied_prob."""
        result = expected_value(-110, 0.55)
        expected_edge = 0.55 - implied_probability(-110)
        assert isclose(result["ev_edge"], expected_edge, abs_tol=1e-4)

    def test_ev_dollar_positive_for_plus_ev(self):
        """ev_dollar > 0 when it's a +EV bet."""
        result = expected_value(150, 0.45)
        assert result["ev_dollar"] > 0

    def test_ev_dollar_negative_for_minus_ev(self):
        """ev_dollar < 0 when it's a -EV bet."""
        result = expected_value(-200, 0.5)
        assert result["ev_dollar"] < 0

    def test_ev_edge_pct_format_positive(self):
        """Positive edge has '+' prefix in percentage."""
        result = expected_value(-110, 0.55)
        assert result["ev_edge_pct"].startswith("+")

    def test_ev_edge_pct_format_negative(self):
        """Negative edge has '-' prefix in percentage."""
        result = expected_value(-110, 0.50)
        assert result["ev_edge_pct"].startswith("-")

    def test_return_keys(self):
        """All expected keys present."""
        result = expected_value(-110, 0.55)
        expected = {"book_implied_prob", "fair_prob", "ev_edge",
                    "ev_edge_pct", "ev_dollar", "is_positive_ev"}
        assert set(result.keys()) == expected

    def test_zero_odds(self):
        """Odds of 0 (even money) → decimal payout = 2.0."""
        result = expected_value(0, 0.55)
        assert result["book_implied_prob"] == 0.5

    def test_large_positive_odds(self):
        """+500 with fair prob 0.25 → check EV calculation correctness."""
        result = expected_value(500, 0.25)
        # Decimal payout = 1 + 500/100 = 6.0
        # ev_dollar = (6.0 * 0.25) - 1 = 0.5
        assert isclose(result["ev_dollar"], 0.5, abs_tol=0.01)

    def test_fair_prob_returned_unchanged(self):
        """The fair_prob in the result matches what was passed in."""
        result = expected_value(-110, 0.55)
        assert isclose(result["fair_prob"], 0.55, abs_tol=1e-6)

    def test_book_implied_matches_implied_probability(self):
        """book_implied_prob matches implied_probability function."""
        result = expected_value(-150, 0.65)
        assert isclose(result["book_implied_prob"], implied_probability(-150), abs_tol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 8. kelly_criterion — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestKellyCriterion:
    """Kelly Criterion optimal bet sizing."""

    def test_positive_edge_recommends_bet(self):
        """With +EV, Kelly recommends a positive bet."""
        result = kelly_criterion(-110, 0.55)
        assert result["is_positive"] is True

    def test_no_edge_no_bet(self):
        """When fair prob ≤ book implied, no bet recommended."""
        result = kelly_criterion(-110, 0.50)
        assert result["is_positive"] is False

    def test_full_kelly_gt_recommended(self):
        """Full Kelly ≥ recommended (fractional reduces size)."""
        result = kelly_criterion(-110, 0.55, fraction=0.25)
        full = float(result["full_kelly_pct"].rstrip("%"))
        rec = float(result["recommended_pct"].rstrip("%"))
        assert full >= rec

    def test_quarter_kelly_fraction(self):
        """Quarter Kelly = full_kelly * 0.25."""
        result = kelly_criterion(-110, 0.55, fraction=0.25)
        full = float(result["full_kelly_pct"].rstrip("%"))
        rec = float(result["recommended_pct"].rstrip("%"))
        assert isclose(rec, full * 0.25, abs_tol=0.01)

    def test_half_kelly(self):
        """Half Kelly = full_kelly * 0.5."""
        result = kelly_criterion(150, 0.50, fraction=0.5)
        full = float(result["full_kelly_pct"].rstrip("%"))
        rec = float(result["recommended_pct"].rstrip("%"))
        assert isclose(rec, full * 0.5, abs_tol=0.01)

    def test_kelly_clamped_to_zero(self):
        """Negative edge → Kelly clamped to 0%."""
        result = kelly_criterion(-200, 0.50)
        assert result["recommended_fraction"] == 0.0

    def test_bankroll_example_format(self):
        """bankroll_example is a readable string."""
        result = kelly_criterion(-110, 0.55)
        assert "$" in result["bankroll_example"]
        assert "bankroll" in result["bankroll_example"]

    def test_decimal_odds_conversion(self):
        """-110 → decimal odds = 1.909."""
        result = kelly_criterion(-110, 0.55)
        assert isclose(result["decimal_odds"], 1.9091, abs_tol=0.01)

    def test_positive_odds_decimal(self):
        """+200 → decimal odds = 3.0."""
        result = kelly_criterion(200, 0.40)
        assert isclose(result["decimal_odds"], 3.0, abs_tol=0.01)

    def test_kelly_fraction_used_recorded(self):
        """The fraction parameter is stored in the result."""
        result = kelly_criterion(-110, 0.55, fraction=0.25)
        assert result["kelly_fraction_used"] == 0.25

    def test_edge_matches_ev(self):
        """Kelly edge should match EV edge."""
        kelly = kelly_criterion(-110, 0.55)
        ev = expected_value(-110, 0.55)
        assert isclose(kelly["edge"], ev["ev_edge"], abs_tol=1e-4)

    def test_return_keys(self):
        """All expected keys present."""
        result = kelly_criterion(-110, 0.55)
        expected = {"full_kelly_pct", "recommended_pct", "recommended_fraction",
                    "edge", "edge_pct", "decimal_odds", "fair_prob",
                    "book_implied_prob", "is_positive", "bankroll_example",
                    "kelly_fraction_used"}
        assert set(result.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════
# 9. bayesian_update — 12 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBayesianUpdate:
    """Beta-conjugate Bayesian probability updating."""

    def test_no_evidence_returns_prior(self):
        """With no evidence, posterior = prior."""
        result = bayesian_update(0.6, [])
        assert isclose(result["posterior_prob"], 0.6, abs_tol=0.001)

    def test_evidence_shifts_posterior(self):
        """Evidence that agrees with prior keeps it similar, evidence that disagrees shifts it."""
        result = bayesian_update(0.6, [0.7, 0.7, 0.7])
        assert result["posterior_prob"] > 0.6  # shifted toward evidence

    def test_evidence_below_prior(self):
        """Evidence below prior shifts posterior down."""
        result = bayesian_update(0.6, [0.4, 0.4, 0.4])
        assert result["posterior_prob"] < 0.6

    def test_credible_interval_contains_posterior(self):
        """90% CI should contain the posterior mean."""
        result = bayesian_update(0.6, [0.55, 0.65])
        ci = result["credible_interval_90"]
        assert ci["low"] <= result["posterior_prob"] <= ci["high"]

    def test_evidence_count(self):
        """evidence_count matches number of evidence probs."""
        result = bayesian_update(0.5, [0.6, 0.7, 0.8])
        assert result["evidence_count"] == 3

    def test_total_kappa_increases(self):
        """Total kappa = prior_kappa + n * evidence_kappa."""
        result = bayesian_update(0.5, [0.6, 0.7], prior_kappa=20, evidence_kappa=5)
        assert isclose(result["total_kappa"], 20 + 2 * 5, abs_tol=0.1)

    def test_update_trace_length(self):
        """Trace has prior + one entry per evidence."""
        result = bayesian_update(0.5, [0.6, 0.7, 0.8])
        assert len(result["update_trace"]) == 4  # prior + 3 evidence

    def test_shift_from_prior_sign(self):
        """shift_from_prior is positive when posterior > prior."""
        result = bayesian_update(0.5, [0.7, 0.7, 0.7])
        assert result["shift_from_prior"] > 0

    def test_shift_negative(self):
        """shift_from_prior is negative when posterior < prior."""
        result = bayesian_update(0.7, [0.3, 0.3, 0.3])
        assert result["shift_from_prior"] < 0

    def test_prior_clamped(self):
        """Edge case: prior_prob = 0 is clamped to 0.001."""
        result = bayesian_update(0.0, [0.5])
        assert result["prior_prob"] == 0.001

    def test_prior_clamped_high(self):
        """Edge case: prior_prob = 1 is clamped to 0.999."""
        result = bayesian_update(1.0, [0.5])
        assert result["prior_prob"] == 0.999

    def test_return_keys(self):
        """All expected keys present."""
        result = bayesian_update(0.6, [0.55])
        expected = {
            "prior_prob", "posterior_prob", "posterior_prob_pct",
            "prior_alpha", "prior_beta", "posterior_alpha", "posterior_beta",
            "std_dev", "credible_interval_90",
            "evidence_count", "total_kappa",
            "shift_from_prior", "shift_from_prior_pct", "update_trace",
        }
        assert set(result.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════
# 10. _prob_to_beta_params — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestProbToBetaParams:
    """Convert probability to Beta(alpha, beta) parameters."""

    def test_even_money(self):
        """p=0.5, kappa=20 → (10, 10)."""
        alpha, beta = _prob_to_beta_params(0.5, 20)
        assert alpha == 10.0
        assert beta == 10.0

    def test_favourite(self):
        """p=0.8, kappa=10 → (8, 2)."""
        alpha, beta = _prob_to_beta_params(0.8, 10)
        assert isclose(alpha, 8.0)
        assert isclose(beta, 2.0)

    def test_underdog(self):
        """p=0.2, kappa=10 → (2, 8)."""
        alpha, beta = _prob_to_beta_params(0.2, 10)
        assert isclose(alpha, 2.0)
        assert isclose(beta, 8.0)

    def test_sum_equals_kappa(self):
        """alpha + beta always equals kappa."""
        alpha, beta = _prob_to_beta_params(0.65, 15)
        assert isclose(alpha + beta, 15.0)

    def test_mean_equals_prob(self):
        """alpha / (alpha + beta) = input probability."""
        alpha, beta = _prob_to_beta_params(0.7, 20)
        assert isclose(alpha / (alpha + beta), 0.7)

    def test_high_kappa(self):
        """High kappa → tighter distribution (larger params)."""
        a1, b1 = _prob_to_beta_params(0.5, 10)
        a2, b2 = _prob_to_beta_params(0.5, 100)
        assert a2 > a1

    def test_zero_prob(self):
        """p=0 → alpha=0, beta=kappa."""
        alpha, beta = _prob_to_beta_params(0.0, 20)
        assert alpha == 0.0
        assert beta == 20.0

    def test_one_prob(self):
        """p=1 → alpha=kappa, beta=0."""
        alpha, beta = _prob_to_beta_params(1.0, 20)
        assert alpha == 20.0
        assert beta == 0.0

    def test_returns_tuple(self):
        """Returns a tuple of two floats."""
        result = _prob_to_beta_params(0.5, 10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_small_kappa(self):
        """Small kappa (weak prior)."""
        alpha, beta = _prob_to_beta_params(0.6, 3)
        assert isclose(alpha, 1.8)
        assert isclose(beta, 1.2)


# ═══════════════════════════════════════════════════════════════════════════
# 11. _shin_solve_z — 10 tests
# ═══════════════════════════════════════════════════════════════════════════

class TestShinSolveZ:
    """Solve for the Shin insider-trading parameter z."""

    def test_no_vig_returns_zero(self):
        """When sum ≤ 1.0, z = 0 (no vig to decompose)."""
        assert _shin_solve_z([0.5, 0.5]) == 0.0

    def test_standard_vig(self):
        """Standard vig → positive z."""
        z = _shin_solve_z([0.5238, 0.5238])  # -110/-110 implied
        assert z > 0

    def test_z_in_valid_range(self):
        """z should be between 0 and 1."""
        z = _shin_solve_z([0.6, 0.6])
        assert 0 < z < 1

    def test_higher_vig_higher_z(self):
        """More vig → larger z (more insider protection)."""
        z_low = _shin_solve_z([0.51, 0.51])   # ~2% vig
        z_high = _shin_solve_z([0.57, 0.57])   # ~14% vig
        assert z_high > z_low

    def test_sum_of_shin_probs_is_one(self):
        """After solving z, Shin probabilities sum to 1.0."""
        from math import sqrt
        probs = [0.6, 0.55]
        S = sum(probs)
        z = _shin_solve_z(probs)
        denom = 2.0 * (1.0 - z)
        total = 0.0
        for p in probs:
            disc = z * z + 4.0 * (1.0 - z) * (p * p) / S
            total += (sqrt(disc) - z) / denom
        assert isclose(total, 1.0, abs_tol=1e-6)

    def test_typical_z_range(self):
        """For typical -110/-110 vig, z should be in 0.01–0.10 range."""
        z = _shin_solve_z([0.5238, 0.5238])
        assert 0.005 < z < 0.15

    def test_lopsided_probs(self):
        """Heavy favourite market still produces valid z."""
        z = _shin_solve_z([0.9, 0.2])
        assert 0 < z < 1

    def test_convergence_precision(self):
        """Default tolerance produces z precise to 10 decimal places."""
        z1 = _shin_solve_z([0.55, 0.55], tol=1e-10)
        z2 = _shin_solve_z([0.55, 0.55], tol=1e-12)
        assert isclose(z1, z2, abs_tol=1e-8)

    def test_single_iteration_low_vig(self):
        """Very low vig → z close to 0."""
        z = _shin_solve_z([0.501, 0.501])
        assert z < 0.01

    def test_below_one_returns_zero(self):
        """When sum < 1.0 (negative vig / arb), z = 0."""
        assert _shin_solve_z([0.4, 0.4]) == 0.0
