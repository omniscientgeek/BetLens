"""
Odds mathematics utilities for the detection phase.

Provides:
1. Implied Probability  – convert American odds → probability
2. Vig (Vigorish)       – bookmaker margin per market
3. No-Vig / Fair Odds   – true probability with vig removed
"""


def implied_probability(american_odds: int | float) -> float:
    """Convert American odds to an implied probability (0-1 scale).

    Negative odds (favourite):  prob = |odds| / (|odds| + 100)
    Positive odds (underdog):   prob = 100   / (odds + 100)

    Examples
    --------
    >>> round(implied_probability(-228), 4)
    0.6951
    >>> round(implied_probability(196), 4)
    0.3378
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    elif american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        # Odds of exactly 0 are undefined; treat as even money
        return 0.5


def calculate_vig(odds_side_a: int | float, odds_side_b: int | float) -> dict:
    """Calculate the vigorish (overround) for a two-outcome market.

    Parameters
    ----------
    odds_side_a : American odds for side A (e.g. home spread)
    odds_side_b : American odds for side B (e.g. away spread)

    Returns
    -------
    dict with keys:
        implied_a       – raw implied probability of side A
        implied_b       – raw implied probability of side B
        total_implied   – sum of both (>1.0 means vig exists)
        vig             – overround as a proportion (e.g. 0.039 ≈ 3.9 %)
        vig_pct         – overround as a percentage string ("3.9%")
    """
    prob_a = implied_probability(odds_side_a)
    prob_b = implied_probability(odds_side_b)
    total = prob_a + prob_b
    vig = total - 1.0

    return {
        "implied_a": round(prob_a, 6),
        "implied_b": round(prob_b, 6),
        "total_implied": round(total, 6),
        "vig": round(vig, 6),
        "vig_pct": f"{round(vig * 100, 2)}%",
    }


def no_vig_probabilities(odds_side_a: int | float, odds_side_b: int | float) -> dict:
    """Remove the vig and return fair (true) probabilities for both sides.

    Method: proportional scaling – divide each raw implied probability by the
    sum of both, so they add to exactly 1.0.

    Parameters
    ----------
    odds_side_a : American odds for side A
    odds_side_b : American odds for side B

    Returns
    -------
    dict with keys:
        fair_a     – fair probability of side A (0-1)
        fair_b     – fair probability of side B (0-1)
        fair_a_pct – as percentage string
        fair_b_pct – as percentage string
        vig_pct    – the vig that was removed
    """
    prob_a = implied_probability(odds_side_a)
    prob_b = implied_probability(odds_side_b)
    total = prob_a + prob_b

    fair_a = prob_a / total if total else 0.5
    fair_b = prob_b / total if total else 0.5
    vig = total - 1.0

    return {
        "fair_a": round(fair_a, 6),
        "fair_b": round(fair_b, 6),
        "fair_a_pct": f"{round(fair_a * 100, 2)}%",
        "fair_b_pct": f"{round(fair_b * 100, 2)}%",
        "vig_pct": f"{round(vig * 100, 2)}%",
    }


def fair_odds_to_american(fair_prob: float) -> int:
    """Convert a fair probability back to American odds.

    prob > 0.5  → negative odds (favourite)
    prob < 0.5  → positive odds (underdog)
    prob == 0.5 → +100 (even money / pick'em)
    """
    if fair_prob <= 0 or fair_prob >= 1:
        raise ValueError(f"Probability must be between 0 and 1, got {fair_prob}")
    if fair_prob == 0.5:
        return 100
    if fair_prob > 0.5:
        return -round(fair_prob / (1 - fair_prob) * 100)
    return round((1 - fair_prob) / fair_prob * 100)
