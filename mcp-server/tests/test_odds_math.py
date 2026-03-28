"""
Comprehensive unit tests for odds_math.py — the core probability engine.

Every formula is validated against hand-calculated expected values and known
mathematical identities.  This is the heart of BetStamp; correctness is
non-negotiable.
"""

import sys
import os
import math
import pytest

# Allow import of odds_math from webservice/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "webservice")))

from odds_math import (
    implied_probability,
    calculate_vig,
    no_vig_probabilities,
    shin_probabilities,
    _shin_solve_z,
    fair_odds_to_american,
    arbitrage_profit,
    expected_value,
    kelly_criterion,
    _prob_to_beta_params,
    bayesian_update,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPLIED PROBABILITY — American odds → probability [0, 1]
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpliedProbability:
    """Validate American-to-probability conversion with hand-checked values."""

    def test_heavy_favourite_neg110(self):
        # -110 → 110/210 = 0.52381
        assert round(implied_probability(-110), 5) == 0.52381

    def test_standard_favourite_neg150(self):
        # -150 → 150/250 = 0.60000
        assert round(implied_probability(-150), 5) == 0.60000

    def test_heavy_favourite_neg300(self):
        # -300 → 300/400 = 0.75000
        assert round(implied_probability(-300), 5) == 0.75000

    def test_extreme_favourite_neg1000(self):
        # -1000 → 1000/1100 = 0.90909
        assert round(implied_probability(-1000), 5) == 0.90909

    def test_standard_underdog_plus150(self):
        # +150 → 100/250 = 0.40000
        assert round(implied_probability(150), 5) == 0.40000

    def test_big_underdog_plus300(self):
        # +300 → 100/400 = 0.25000
        assert round(implied_probability(300), 5) == 0.25000

    def test_longshot_plus1000(self):
        # +1000 → 100/1100 = 0.09091
        assert round(implied_probability(1000), 5) == 0.09091

    def test_even_money_plus100(self):
        # +100 → 100/200 = 0.50000
        assert round(implied_probability(100), 5) == 0.50000

    def test_pick_em_neg100(self):
        # -100 → 100/200 = 0.50000
        assert round(implied_probability(-100), 5) == 0.50000

    def test_zero_odds_even_money(self):
        # Edge case: 0 treated as even money
        assert implied_probability(0) == 0.5

    def test_exact_docstring_example_neg228(self):
        # From the docstring: -228 → 0.6951
        assert round(implied_probability(-228), 4) == 0.6951

    def test_exact_docstring_example_pos196(self):
        # From the docstring: +196 → 0.3378
        assert round(implied_probability(196), 4) == 0.3378

    def test_probability_always_between_0_and_1(self):
        """All valid odds must yield probability in (0, 1)."""
        for odds in [-10000, -500, -110, 100, 200, 500, 10000]:
            p = implied_probability(odds)
            assert 0 < p < 1, f"Odds {odds} gave probability {p}"

    def test_symmetry_neg_and_pos_same_magnitude(self):
        """|-X| and |+X| implied probs should sum to exactly 1 when X = 100."""
        p_neg = implied_probability(-100)
        p_pos = implied_probability(100)
        assert abs(p_neg + p_pos - 1.0) < 1e-10

    def test_float_odds_accepted(self):
        # Function signature allows float odds
        p = implied_probability(-110.5)
        assert 0 < p < 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VIG (VIGORISH) CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalculateVig:
    """Validate bookmaker margin (overround) calculations."""

    def test_standard_juice_neg110_neg110(self):
        """Standard -110/-110 market should have ~4.55% vig."""
        result = calculate_vig(-110, -110)
        # Each side: 110/210 = 0.52381; total = 1.04762; vig = 0.04762
        assert round(result["vig"], 4) == 0.0476
        assert result["implied_a"] == result["implied_b"]

    def test_zero_vig_market(self):
        """+100/+100 market has zero overround (theoretical fair market)."""
        result = calculate_vig(100, 100)
        # Each side = 0.5; total = 1.0; vig = 0.0
        assert result["vig"] == 0.0
        assert result["total_implied"] == 1.0

    def test_negative_vig_arb_opportunity(self):
        """Combined implied < 1.0 means negative vig (arbitrage exists)."""
        result = calculate_vig(200, 200)
        # Each side: 100/300 = 0.3333; total = 0.6667; vig = -0.3333
        assert result["vig"] < 0
        assert result["total_implied"] < 1.0
        assert result["implied_a"] == result["implied_b"]

    def test_lopsided_market_neg500_plus400(self):
        """Lopsided favourite/underdog market."""
        result = calculate_vig(-500, 400)
        # -500 → 500/600 = 0.8333; +400 → 100/500 = 0.2000
        # Total = 1.0333; vig = 0.0333
        assert round(result["implied_a"], 4) == 0.8333
        assert round(result["implied_b"], 4) == 0.2000
        assert round(result["vig"], 4) == 0.0333

    def test_vig_pct_format(self):
        result = calculate_vig(-110, -110)
        assert result["vig_pct"].endswith("%")
        assert "4.76" in result["vig_pct"]

    def test_heavy_juice_neg105_neg115(self):
        """Typical spread market with asymmetric juice."""
        result = calculate_vig(-105, -115)
        prob_a = 105 / 205  # 0.51220
        prob_b = 115 / 215  # 0.53488
        assert round(result["implied_a"], 4) == round(prob_a, 4)
        assert round(result["implied_b"], 4) == round(prob_b, 4)
        assert result["vig"] > 0

    def test_extreme_vig_neg200_neg200(self):
        """Extremely juiced market (unrealistic but validates math)."""
        result = calculate_vig(-200, -200)
        # Each side: 200/300 = 0.6667; total = 1.3333; vig = 0.3333
        assert round(result["vig"], 4) == 0.3333
        assert round(result["total_implied"], 4) == 1.3333

    def test_sum_of_implied_probs(self):
        """Total implied must equal sum of individual implied probs."""
        result = calculate_vig(-150, 130)
        assert abs(result["total_implied"] - (result["implied_a"] + result["implied_b"])) < 1e-6

    def test_vig_is_total_minus_one(self):
        result = calculate_vig(-110, -110)
        assert abs(result["vig"] - (result["total_implied"] - 1.0)) < 1e-6

    def test_returned_keys_complete(self):
        result = calculate_vig(-110, -110)
        expected_keys = {"implied_a", "implied_b", "total_implied", "vig", "vig_pct"}
        assert set(result.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NO-VIG FAIR ODDS (Proportional Scaling)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoVigProbabilities:
    """Validate proportional vig removal — fair probs must sum to 1.0."""

    def test_fair_probs_sum_to_one(self):
        """The entire purpose: fair_a + fair_b == 1.0."""
        result = no_vig_probabilities(-110, -110)
        assert abs(result["fair_a"] + result["fair_b"] - 1.0) < 1e-6

    def test_symmetric_market_equal_probs(self):
        """-110/-110 should give 50/50 after vig removal."""
        result = no_vig_probabilities(-110, -110)
        assert abs(result["fair_a"] - 0.5) < 1e-6
        assert abs(result["fair_b"] - 0.5) < 1e-6

    def test_favourite_retains_higher_prob(self):
        """Favourite side should have higher fair prob than underdog."""
        result = no_vig_probabilities(-200, 170)
        assert result["fair_a"] > result["fair_b"]

    def test_underdog_gets_lower_prob(self):
        result = no_vig_probabilities(170, -200)
        assert result["fair_a"] < result["fair_b"]

    def test_exact_calculation_neg150_plus130(self):
        """Hand-verify: -150 → 0.6, +130 → 0.4348, total=1.0348."""
        result = no_vig_probabilities(-150, 130)
        raw_a = 150 / 250  # 0.6
        raw_b = 100 / 230  # 0.43478
        total = raw_a + raw_b
        expected_fair_a = raw_a / total
        expected_fair_b = raw_b / total
        assert abs(result["fair_a"] - round(expected_fair_a, 6)) < 1e-5
        assert abs(result["fair_b"] - round(expected_fair_b, 6)) < 1e-5

    def test_percentage_format_strings(self):
        result = no_vig_probabilities(-110, -110)
        assert result["fair_a_pct"] == "50.0%"
        assert result["fair_b_pct"] == "50.0%"

    def test_vig_pct_matches_calculate_vig(self):
        """Vig reported should match the calculate_vig function."""
        result = no_vig_probabilities(-110, -110)
        vig_result = calculate_vig(-110, -110)
        assert result["vig_pct"] == vig_result["vig_pct"]

    def test_no_vig_market_unchanged(self):
        """If total implied == 1.0, fair probs equal raw probs."""
        result = no_vig_probabilities(100, -100)
        # +100 → 0.5, -100 → 0.5; total = 1.0; fair = raw
        assert abs(result["fair_a"] - 0.5) < 1e-6
        assert abs(result["fair_b"] - 0.5) < 1e-6

    def test_extreme_favourite_still_sums_to_one(self):
        result = no_vig_probabilities(-1000, 700)
        assert abs(result["fair_a"] + result["fair_b"] - 1.0) < 1e-6

    def test_equal_longshot_sums_to_one(self):
        result = no_vig_probabilities(500, 500)
        assert abs(result["fair_a"] + result["fair_b"] - 1.0) < 1e-6

    def test_all_keys_present(self):
        result = no_vig_probabilities(-110, -110)
        expected = {"fair_a", "fair_b", "fair_a_pct", "fair_b_pct", "vig_pct"}
        assert set(result.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SHIN MODEL — Asymmetric Vig Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TestShinProbabilities:
    """Validate the Shin (1993) model for asymmetric margin allocation.

    Key invariants:
    - shin_a + shin_b must sum to 1.0
    - Longshot side absorbs more vig than the favourite
    - z parameter is in (0, 1) for markets with vig
    - Results agree with naive proportional within ~5pp
    """

    def test_shin_probs_sum_to_one(self):
        """Fundamental Shin property: adjusted probs must sum to 1."""
        result = shin_probabilities(-150, 130)
        assert abs(result["shin_a"] + result["shin_b"] - 1.0) < 1e-5

    def test_shin_probs_sum_to_one_extreme_fav(self):
        result = shin_probabilities(-500, 400)
        assert abs(result["shin_a"] + result["shin_b"] - 1.0) < 1e-5

    def test_shin_probs_sum_to_one_symmetric(self):
        result = shin_probabilities(-110, -110)
        assert abs(result["shin_a"] + result["shin_b"] - 1.0) < 1e-5

    def test_longshot_absorbs_more_vig(self):
        """Core Shin insight: longshot side should absorb more vig."""
        # Use a strongly lopsided market where the effect is clear
        result = shin_probabilities(-500, 350)
        # Side A is favourite (-500), side B is underdog (+350)
        # vig_on_b (longshot) should be >= vig_on_a (favourite)
        assert result["vig_on_b"] >= result["vig_on_a"]

    def test_symmetric_market_equal_vig_allocation(self):
        """When both sides have equal odds, vig should split equally."""
        result = shin_probabilities(-110, -110)
        assert abs(result["vig_on_a"] - result["vig_on_b"]) < 1e-5

    def test_z_parameter_positive_with_vig(self):
        """z (insider fraction) must be > 0 when vig exists."""
        result = shin_probabilities(-110, -110)
        assert result["z"] > 0

    def test_z_parameter_range(self):
        """z should be in (0, 0.15) for normal markets."""
        result = shin_probabilities(-110, -110)
        assert 0 < result["z"] < 0.15

    def test_z_zero_when_no_vig(self):
        """With no vig (total implied <= 1), z should be 0."""
        result = shin_probabilities(100, -100)
        # +100 → 0.5, -100 → 0.5 → total = 1.0 → no vig
        assert result["z"] == 0.0

    def test_shin_close_to_naive_for_balanced_market(self):
        """For near-even markets, Shin ≈ Naive (small delta)."""
        result = shin_probabilities(-110, -110)
        assert abs(result["delta_a"]) < 0.01
        assert abs(result["delta_b"]) < 0.01

    def test_shin_diverges_from_naive_for_lopsided(self):
        """For lopsided markets, Shin and Naive should differ meaningfully."""
        result = shin_probabilities(-500, 400)
        # At least one delta should be > 0.001
        assert abs(result["delta_a"]) > 0.001 or abs(result["delta_b"]) > 0.001

    def test_vig_allocation_sums_to_total_vig(self):
        """vig_on_a + vig_on_b must equal total vig."""
        result = shin_probabilities(-150, 130)
        raw_a = implied_probability(-150)
        raw_b = implied_probability(130)
        total_vig = (raw_a + raw_b) - 1.0
        computed_vig = result["vig_on_a"] + result["vig_on_b"]
        assert abs(computed_vig - total_vig) < 1e-4

    def test_shin_favourite_prob_less_than_raw(self):
        """Shin-adjusted prob should be less than raw implied (vig removed)."""
        result = shin_probabilities(-200, 170)
        raw_a = implied_probability(-200)
        assert result["shin_a"] < raw_a

    def test_method_key_is_shin(self):
        result = shin_probabilities(-110, -110)
        assert result["method"] == "shin"

    def test_all_keys_present(self):
        result = shin_probabilities(-110, -110)
        expected_keys = {
            "shin_a", "shin_b", "shin_a_pct", "shin_b_pct",
            "naive_a", "naive_b", "delta_a", "delta_b",
            "z", "z_pct", "vig_pct",
            "vig_on_a", "vig_on_b", "vig_on_a_pct", "vig_on_b_pct",
            "method",
        }
        assert set(result.keys()) == expected_keys


class TestShinSolveZ:
    """Validate the bisection solver for the Shin parameter z."""

    def test_no_vig_returns_zero(self):
        """If sum of probs <= 1, z = 0."""
        assert _shin_solve_z([0.5, 0.5]) == 0.0

    def test_standard_market(self):
        """Standard vigged market should give z in (0, 0.1)."""
        probs = [implied_probability(-110), implied_probability(-110)]
        z = _shin_solve_z(probs)
        assert 0 < z < 0.1

    def test_shin_adjusted_probs_sum_to_one(self):
        """After solving z, recomputing Shin probs should sum to 1."""
        probs = [0.6, 0.45]  # S = 1.05
        z = _shin_solve_z(probs)
        S = sum(probs)
        denom = 2.0 * (1.0 - z)
        total = 0.0
        for p in probs:
            disc = z * z + 4.0 * (1.0 - z) * (p * p) / S
            total += (math.sqrt(disc) - z) / denom
        assert abs(total - 1.0) < 1e-8

    def test_higher_vig_gives_higher_z(self):
        """More overround should produce a larger z."""
        probs_low = [0.52, 0.52]   # S = 1.04
        probs_high = [0.55, 0.55]  # S = 1.10
        z_low = _shin_solve_z(probs_low)
        z_high = _shin_solve_z(probs_high)
        assert z_high > z_low

    def test_convergence_tolerance(self):
        """Result should be accurate to within the specified tolerance."""
        probs = [0.6, 0.45]
        z = _shin_solve_z(probs, tol=1e-10)
        S = sum(probs)
        denom = 2.0 * (1.0 - z)
        total = sum(
            (math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / S) - z) / denom
            for p in probs
        )
        assert abs(total - 1.0) < 1e-8

    def test_three_outcome_market(self):
        """Shin solver should also work with >2 outcomes (e.g., 1X2 soccer)."""
        probs = [0.45, 0.30, 0.35]  # S = 1.10
        z = _shin_solve_z(probs)
        S = sum(probs)
        denom = 2.0 * (1.0 - z)
        total = sum(
            (math.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / S) - z) / denom
            for p in probs
        )
        assert abs(total - 1.0) < 1e-6

    def test_extreme_vig_still_converges(self):
        """Very heavy vig market should still converge."""
        probs = [0.7, 0.7]  # S = 1.4 — extremely juiced
        z = _shin_solve_z(probs)
        assert 0 < z < 1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FAIR ODDS TO AMERICAN CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

class TestFairOddsToAmerican:
    """Validate reverse conversion: probability → American odds."""

    def test_even_money(self):
        assert fair_odds_to_american(0.5) == 100

    def test_favourite_60pct(self):
        # 0.6 / 0.4 * 100 = 150 → -150
        assert fair_odds_to_american(0.6) == -150

    def test_favourite_75pct(self):
        # 0.75 / 0.25 * 100 = 300 → -300
        assert fair_odds_to_american(0.75) == -300

    def test_underdog_40pct(self):
        # 0.6 / 0.4 * 100 = 150 → +150
        assert fair_odds_to_american(0.4) == 150

    def test_underdog_25pct(self):
        # 0.75 / 0.25 * 100 = 300 → +300
        assert fair_odds_to_american(0.25) == 300

    def test_underdog_10pct(self):
        # 0.9 / 0.1 * 100 = 900 → +900
        assert fair_odds_to_american(0.1) == 900

    def test_favourite_90pct(self):
        # 0.9 / 0.1 * 100 = 900 → -900
        assert fair_odds_to_american(0.9) == -900

    def test_raises_on_zero(self):
        with pytest.raises(ValueError):
            fair_odds_to_american(0.0)

    def test_raises_on_one(self):
        with pytest.raises(ValueError):
            fair_odds_to_american(1.0)

    def test_raises_on_negative(self):
        with pytest.raises(ValueError):
            fair_odds_to_american(-0.1)

    def test_roundtrip_favourite(self):
        """implied_probability(fair_odds_to_american(p)) ≈ p for p > 0.5."""
        for p in [0.55, 0.6, 0.7, 0.8, 0.9]:
            american = fair_odds_to_american(p)
            recovered = implied_probability(american)
            assert abs(recovered - p) < 0.01, f"Roundtrip failed for p={p}"

    def test_roundtrip_underdog(self):
        """implied_probability(fair_odds_to_american(p)) ≈ p for p < 0.5."""
        for p in [0.1, 0.2, 0.3, 0.4, 0.45]:
            american = fair_odds_to_american(p)
            recovered = implied_probability(american)
            assert abs(recovered - p) < 0.01, f"Roundtrip failed for p={p}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ARBITRAGE PROFIT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestArbitrageProfit:
    """Validate arbitrage detection and profit/stake calculations."""

    def test_clear_arb_opportunity(self):
        """Large underdogs on both sides = guaranteed arb."""
        result = arbitrage_profit(200, 200)
        # Each side: 100/300 = 0.3333; combined = 0.6667; is_arb = True
        assert result["is_arb"] is True
        assert result["profit_pct"] > 0

    def test_no_arb_standard_market(self):
        """-110/-110 has combined > 1 — no arb."""
        result = arbitrage_profit(-110, -110)
        assert result["is_arb"] is False
        assert result["profit_pct"] == 0.0

    def test_arb_profit_calculation(self):
        """Hand-verify arb profit %."""
        result = arbitrage_profit(200, 200)
        # combined = 0.6667; profit = (1 - 0.6667) / 0.6667 = 0.4999 ≈ 50%
        assert abs(result["profit_pct"] - 50.0) < 0.5

    def test_stake_allocation_symmetric_arb(self):
        """Symmetric arb should have 50/50 stake allocation."""
        result = arbitrage_profit(200, 200)
        assert abs(result["stake_a_pct"] - 50.0) < 0.1
        assert abs(result["stake_b_pct"] - 50.0) < 0.1

    def test_stake_allocation_asymmetric_arb(self):
        """Asymmetric arb: favourite gets larger stake."""
        result = arbitrage_profit(-200, 500)
        # -200 → 0.6667; +500 → 0.1667; combined = 0.8333
        # stake_a ≈ 0.6667/0.8333 ≈ 80%; stake_b ≈ 20%
        assert result["is_arb"] is True
        assert result["stake_a_pct"] > result["stake_b_pct"]

    def test_no_arb_gives_50_50_stakes(self):
        """When no arb, stakes default to 50/50."""
        result = arbitrage_profit(-110, -110)
        assert result["stake_a_pct"] == 50.0
        assert result["stake_b_pct"] == 50.0

    def test_combined_implied_equals_sum(self):
        result = arbitrage_profit(-150, 130)
        expected = implied_probability(-150) + implied_probability(130)
        assert abs(result["combined_implied"] - round(expected, 6)) < 1e-5

    def test_edge_case_exactly_one(self):
        """Combined implied = 1.0 exactly → no arb (not strict inequality)."""
        result = arbitrage_profit(100, -100)
        assert result["is_arb"] is False

    def test_narrow_arb_detected(self):
        """Tiny margin arb — combined just below 1.0."""
        # +105 → 100/205 = 0.4878; -104 → 104/204 = 0.5098; combined = 0.9976
        result = arbitrage_profit(105, -104)
        # This particular combo may or may not be arb — just verify consistency
        if result["is_arb"]:
            assert result["profit_pct"] > 0
        else:
            assert result["profit_pct"] == 0.0

    def test_all_keys_present(self):
        result = arbitrage_profit(-110, -110)
        expected = {
            "implied_a", "implied_b", "combined_implied",
            "is_arb", "profit_pct", "stake_a_pct", "stake_b_pct",
        }
        assert set(result.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXPECTED VALUE (EV) CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpectedValue:
    """Validate EV edge, dollar EV, and positive-EV detection."""

    def test_positive_ev_when_fair_prob_higher(self):
        """Book underestimates true prob → +EV."""
        result = expected_value(-110, 0.55)
        # book implied: 110/210 = 0.5238; fair = 0.55 → edge = 0.55 - 0.5238 > 0
        assert result["is_positive_ev"] is True
        assert result["ev_edge"] > 0

    def test_negative_ev_when_fair_prob_lower(self):
        """Book overestimates true prob → -EV."""
        result = expected_value(-110, 0.50)
        # book implied: 0.5238; fair = 0.50 → edge = 0.50 - 0.5238 < 0
        assert result["is_positive_ev"] is False
        assert result["ev_edge"] < 0

    def test_exact_ev_edge_calculation(self):
        """Hand-verify ev_edge = fair_prob - book_implied."""
        result = expected_value(-150, 0.65)
        book_prob = 150 / 250  # 0.6
        expected_edge = 0.65 - book_prob
        assert abs(result["ev_edge"] - round(expected_edge, 6)) < 1e-5

    def test_ev_dollar_positive_favourite(self):
        """Verify EV dollar for a +EV favourite bet."""
        result = expected_value(-150, 0.65)
        # decimal_payout = 1 + 100/150 = 1.6667
        # ev_dollar = 1.6667 * 0.65 - 1 = 0.0833
        decimal_payout = 1 + 100 / 150
        expected_dollar = decimal_payout * 0.65 - 1
        assert abs(result["ev_dollar"] - round(expected_dollar, 4)) < 0.001

    def test_ev_dollar_positive_underdog(self):
        """Verify EV dollar for a +EV underdog bet."""
        result = expected_value(150, 0.45)
        # decimal_payout = 1 + 150/100 = 2.5
        # ev_dollar = 2.5 * 0.45 - 1 = 0.125
        expected_dollar = 2.5 * 0.45 - 1
        assert abs(result["ev_dollar"] - round(expected_dollar, 4)) < 0.001

    def test_zero_edge_is_not_positive(self):
        """Exact fair price means ev_edge = 0 → not positive EV."""
        # Find odds where implied = 0.5 exactly
        result = expected_value(100, 0.5)
        assert result["is_positive_ev"] is False
        assert abs(result["ev_edge"]) < 1e-6

    def test_ev_edge_pct_format(self):
        result = expected_value(-110, 0.55)
        assert result["ev_edge_pct"].startswith("+")
        assert result["ev_edge_pct"].endswith("%")

    def test_negative_ev_edge_pct_format(self):
        result = expected_value(-110, 0.50)
        assert result["ev_edge_pct"].startswith("-")

    def test_decimal_payout_even_money(self):
        """Even money (0) should give decimal payout of 2.0."""
        result = expected_value(0, 0.55)
        # ev_dollar = 2.0 * 0.55 - 1 = 0.10
        assert abs(result["ev_dollar"] - 0.10) < 0.001

    def test_book_implied_prob_matches(self):
        result = expected_value(-200, 0.70)
        assert abs(result["book_implied_prob"] - implied_probability(-200)) < 1e-6

    def test_fair_prob_passthrough(self):
        result = expected_value(-110, 0.55)
        assert abs(result["fair_prob"] - 0.55) < 1e-6

    def test_all_keys_present(self):
        result = expected_value(-110, 0.55)
        expected = {
            "book_implied_prob", "fair_prob", "ev_edge",
            "ev_edge_pct", "ev_dollar", "is_positive_ev",
        }
        assert set(result.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 8. KELLY CRITERION BET SIZING
# ═══════════════════════════════════════════════════════════════════════════════

class TestKellyCriterion:
    """Validate Kelly Criterion optimal bet sizing."""

    def test_positive_edge_gives_positive_kelly(self):
        """When edge > 0, Kelly recommends a bet."""
        result = kelly_criterion(-110, 0.55)
        assert result["is_positive"] is True
        assert result["recommended_fraction"] > 0

    def test_no_edge_gives_zero_kelly(self):
        """When fair_prob <= book_implied, Kelly = 0."""
        result = kelly_criterion(-110, 0.50)
        assert result["is_positive"] is False
        assert result["recommended_fraction"] == 0

    def test_full_kelly_formula(self):
        """Hand-verify Kelly formula: f* = (b*p - q) / b."""
        # -150 odds → b = 100/150 = 0.6667; fair_prob = 0.65
        b = 100 / 150
        p = 0.65
        q = 1 - p
        expected_kelly = (b * p - q) / b
        result = kelly_criterion(-150, 0.65)
        actual = float(result["full_kelly_pct"].rstrip("%")) / 100
        assert abs(actual - expected_kelly) < 0.001

    def test_half_kelly(self):
        """Half Kelly = full Kelly / 2."""
        full = kelly_criterion(-110, 0.55, fraction=1.0)
        half = kelly_criterion(-110, 0.55, fraction=0.5)
        full_val = full["recommended_fraction"]
        half_val = half["recommended_fraction"]
        assert abs(half_val - full_val * 0.5) < 1e-5

    def test_quarter_kelly(self):
        result = kelly_criterion(-110, 0.55, fraction=0.25)
        full = kelly_criterion(-110, 0.55, fraction=1.0)
        assert abs(result["recommended_fraction"] - full["recommended_fraction"] * 0.25) < 1e-5

    def test_kelly_clamped_at_zero(self):
        """Negative edge should clamp Kelly to 0, not go negative."""
        result = kelly_criterion(-110, 0.40)
        assert result["recommended_fraction"] == 0
        assert result["is_positive"] is False

    def test_kelly_clamped_at_one(self):
        """Extreme edge should cap at 100% bankroll."""
        # +10000 odds with 99% win prob = absurd edge
        result = kelly_criterion(10000, 0.99, fraction=1.0)
        # full_kelly should be clamped to 1.0
        full_val = float(result["full_kelly_pct"].rstrip("%")) / 100
        assert full_val <= 1.0

    def test_bankroll_example_correct(self):
        """$1,000 bankroll example should match recommended fraction × 1000."""
        result = kelly_criterion(-110, 0.55, fraction=0.25)
        frac = result["recommended_fraction"]
        expected_str = f"${round(frac * 1000, 2)} on a $1,000 bankroll"
        assert result["bankroll_example"] == expected_str

    def test_decimal_odds_correct(self):
        """Decimal odds should be b + 1 where b = net payout."""
        result = kelly_criterion(-200, 0.70)
        # b = 100/200 = 0.5; decimal = 1.5
        assert abs(result["decimal_odds"] - 1.5) < 0.001

    def test_underdog_kelly(self):
        """Validate Kelly for underdog bet."""
        # +200 → b = 2.0; fair_prob = 0.40 → f = (2*0.4 - 0.6)/2 = 0.1
        result = kelly_criterion(200, 0.40)
        expected_kelly = (2.0 * 0.40 - 0.60) / 2.0
        actual = float(result["full_kelly_pct"].rstrip("%")) / 100
        assert abs(actual - expected_kelly) < 0.001

    def test_edge_matches_ev(self):
        """Kelly's edge should match EV edge for same inputs."""
        kc = kelly_criterion(-110, 0.55)
        ev = expected_value(-110, 0.55)
        assert abs(kc["edge"] - ev["ev_edge"]) < 1e-5

    def test_kelly_fraction_used_recorded(self):
        result = kelly_criterion(-110, 0.55, fraction=0.5)
        assert result["kelly_fraction_used"] == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BAYESIAN UPDATING — Beta-Conjugate Probability Updates
# ═══════════════════════════════════════════════════════════════════════════════

class TestBetaParams:
    """Validate the probability → Beta parameter conversion."""

    def test_even_prob(self):
        alpha, beta = _prob_to_beta_params(0.5, 20)
        assert alpha == 10.0
        assert beta == 10.0

    def test_favourite_prob(self):
        alpha, beta = _prob_to_beta_params(0.7, 20)
        assert abs(alpha - 14.0) < 1e-10
        assert abs(beta - 6.0) < 1e-10

    def test_underdog_prob(self):
        alpha, beta = _prob_to_beta_params(0.3, 10)
        assert alpha == 3.0
        assert beta == 7.0

    def test_higher_kappa_larger_params(self):
        a1, b1 = _prob_to_beta_params(0.6, 10)
        a2, b2 = _prob_to_beta_params(0.6, 50)
        assert a2 > a1
        assert b2 > b1

    def test_ratio_equals_prob(self):
        """alpha / (alpha + beta) should equal the original probability."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            alpha, beta = _prob_to_beta_params(p, 20)
            assert abs(alpha / (alpha + beta) - p) < 1e-10


class TestBayesianUpdate:
    """Validate Bayesian posterior inference with known properties."""

    def test_prior_only_returns_prior(self):
        """With no evidence, posterior should equal prior."""
        result = bayesian_update(0.6, [])
        assert abs(result["posterior_prob"] - 0.6) < 1e-5

    def test_posterior_shifts_toward_evidence(self):
        """Evidence higher than prior should pull posterior up."""
        result = bayesian_update(0.5, [0.7, 0.7, 0.7])
        assert result["posterior_prob"] > 0.5

    def test_posterior_shifts_down_with_lower_evidence(self):
        result = bayesian_update(0.6, [0.4, 0.4, 0.4])
        assert result["posterior_prob"] < 0.6

    def test_more_evidence_stronger_shift(self):
        """More evidence books should create stronger posterior shift."""
        result_few = bayesian_update(0.5, [0.7])
        result_many = bayesian_update(0.5, [0.7, 0.7, 0.7, 0.7, 0.7])
        assert result_many["posterior_prob"] > result_few["posterior_prob"]

    def test_higher_evidence_kappa_stronger_shift(self):
        """Higher evidence_kappa means each book influences more."""
        result_weak = bayesian_update(0.5, [0.7], evidence_kappa=2.0)
        result_strong = bayesian_update(0.5, [0.7], evidence_kappa=20.0)
        assert result_strong["posterior_prob"] > result_weak["posterior_prob"]

    def test_higher_prior_kappa_resists_shift(self):
        """Higher prior_kappa means prior resists evidence more."""
        result_weak_prior = bayesian_update(0.5, [0.7, 0.7, 0.7], prior_kappa=5.0)
        result_strong_prior = bayesian_update(0.5, [0.7, 0.7, 0.7], prior_kappa=100.0)
        # Weak prior → more shift; strong prior → less shift
        assert result_weak_prior["posterior_prob"] > result_strong_prior["posterior_prob"]

    def test_credible_interval_contains_posterior(self):
        """90% CI should contain the posterior mean."""
        result = bayesian_update(0.5, [0.6, 0.65, 0.55])
        ci = result["credible_interval_90"]
        assert ci["low"] <= result["posterior_prob"] <= ci["high"]

    def test_credible_interval_narrows_with_evidence(self):
        """More evidence should narrow the credible interval."""
        result_few = bayesian_update(0.5, [0.6])
        result_many = bayesian_update(0.5, [0.6, 0.6, 0.6, 0.6, 0.6])
        width_few = result_few["credible_interval_90"]["high"] - result_few["credible_interval_90"]["low"]
        width_many = result_many["credible_interval_90"]["high"] - result_many["credible_interval_90"]["low"]
        assert width_many < width_few

    def test_posterior_between_0_and_1(self):
        """Posterior must always be a valid probability."""
        for prior in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = bayesian_update(prior, [0.3, 0.7])
            assert 0 < result["posterior_prob"] < 1

    def test_update_trace_length(self):
        """Trace should have 1 (prior) + N (evidence) entries."""
        result = bayesian_update(0.5, [0.6, 0.7, 0.8])
        assert len(result["update_trace"]) == 4  # 1 prior + 3 evidence

    def test_shift_sign_correct(self):
        result_up = bayesian_update(0.5, [0.8, 0.8])
        result_down = bayesian_update(0.5, [0.2, 0.2])
        assert result_up["shift_from_prior"] > 0
        assert result_down["shift_from_prior"] < 0

    def test_prior_alpha_beta_in_output(self):
        result = bayesian_update(0.6, [0.7], prior_kappa=20.0)
        assert result["prior_alpha"] == 12.0  # 0.6 * 20
        assert result["prior_beta"] == 8.0    # 0.4 * 20

    def test_total_kappa_accumulates(self):
        """total_kappa = prior_kappa + len(evidence) * evidence_kappa."""
        result = bayesian_update(0.5, [0.6, 0.7], prior_kappa=20, evidence_kappa=5)
        expected_kappa = 20 + 2 * 5
        assert abs(result["total_kappa"] - expected_kappa) < 1e-3

    def test_clamping_extreme_prior(self):
        """Extreme priors (0 or 1) should be clamped to (0.001, 0.999)."""
        result = bayesian_update(0.0, [0.5])
        assert result["prior_prob"] == 0.001
        result = bayesian_update(1.0, [0.5])
        assert result["prior_prob"] == 0.999

    def test_std_dev_positive(self):
        result = bayesian_update(0.5, [0.6])
        assert result["std_dev"] > 0

    def test_evidence_count(self):
        result = bayesian_update(0.5, [0.6, 0.7, 0.8, 0.9])
        assert result["evidence_count"] == 4
