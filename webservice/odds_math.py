"""
Odds mathematics utilities for the detection phase.

Provides:
1. Implied Probability  – convert American odds → probability
2. Vig (Vigorish)       – bookmaker margin per market
3. No-Vig / Fair Odds   – true probability with vig removed
4. Expected Value (EV)  – edge vs consensus fair probability
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


def arbitrage_profit(odds_side_a: int | float, odds_side_b: int | float) -> dict:
    """Calculate the arbitrage profit for a pair of odds from (potentially different) books.

    If the combined implied probability is < 1.0, there is a guaranteed profit.
    The profit percentage tells you how much you earn per $100 wagered total.

    Parameters
    ----------
    odds_side_a : American odds for side A (e.g. home moneyline at Book 1)
    odds_side_b : American odds for side B (e.g. away moneyline at Book 2)

    Returns
    -------
    dict with keys:
        implied_a        – implied probability of side A
        implied_b        – implied probability of side B
        combined_implied – sum of implied probabilities
        is_arb           – True if an arbitrage opportunity exists
        profit_pct       – guaranteed profit as a percentage (0 if no arb)
        stake_a_pct      – optimal percentage of bankroll to place on side A
        stake_b_pct      – optimal percentage of bankroll to place on side B
    """
    prob_a = implied_probability(odds_side_a)
    prob_b = implied_probability(odds_side_b)
    combined = prob_a + prob_b
    is_arb = combined < 1.0

    if is_arb:
        profit_pct = round((1.0 - combined) / combined * 100, 4)
        # Optimal stakes: proportional to implied probability, normalized
        stake_a_pct = round(prob_a / combined * 100, 2)
        stake_b_pct = round(prob_b / combined * 100, 2)
    else:
        profit_pct = 0.0
        stake_a_pct = 50.0
        stake_b_pct = 50.0

    return {
        "implied_a": round(prob_a, 6),
        "implied_b": round(prob_b, 6),
        "combined_implied": round(combined, 6),
        "is_arb": is_arb,
        "profit_pct": profit_pct,
        "stake_a_pct": stake_a_pct,
        "stake_b_pct": stake_b_pct,
    }


def expected_value(book_odds: int | float, fair_prob: float) -> dict:
    """Calculate the Expected Value (EV) of a bet.

    EV measures whether a bet is +EV (profitable long-term) or -EV by
    comparing the book's implied probability against the consensus fair
    probability (true probability with vig removed).

    Formula:
        ev_edge   = fair_prob - book_implied_prob
        ev_pct    = ev_edge * 100  (as percentage points)
        ev_dollar = (decimal_payout * fair_prob) - 1  (profit per $1 wagered)

    A positive ev_edge means the book is underestimating the true
    probability — you're getting better odds than you should.

    Parameters
    ----------
    book_odds : American odds offered by the sportsbook
    fair_prob : Consensus fair (no-vig) probability for this side (0-1)

    Returns
    -------
    dict with keys:
        book_implied_prob – raw implied probability from book odds
        fair_prob         – the fair probability passed in
        ev_edge           – fair_prob - book_implied_prob (positive = +EV)
        ev_edge_pct       – ev_edge as percentage string (e.g. "+2.35%")
        ev_dollar         – expected profit per $1 wagered
        is_positive_ev    – True if ev_edge > 0
    """
    book_prob = implied_probability(book_odds)

    # Decimal payout: what $1 returns (including stake)
    if book_odds < 0:
        decimal_payout = 1 + (100 / abs(book_odds))
    elif book_odds > 0:
        decimal_payout = 1 + (book_odds / 100)
    else:
        decimal_payout = 2.0  # even money

    ev_edge = fair_prob - book_prob
    ev_dollar = (decimal_payout * fair_prob) - 1

    sign = "+" if ev_edge >= 0 else ""
    return {
        "book_implied_prob": round(book_prob, 6),
        "fair_prob": round(fair_prob, 6),
        "ev_edge": round(ev_edge, 6),
        "ev_edge_pct": f"{sign}{round(ev_edge * 100, 2)}%",
        "ev_dollar": round(ev_dollar, 4),
        "is_positive_ev": ev_edge > 0,
    }


def kelly_criterion(book_odds: int | float, fair_prob: float, fraction: float = 1.0) -> dict:
    """Calculate the optimal Kelly Criterion bet size.

    The Kelly Criterion determines the mathematically optimal fraction of your
    bankroll to wager given your edge and the odds offered.  Uses the "true"
    probability (typically derived from Pinnacle no-vig lines) as the edge
    reference.

    Formula (for decimal odds *b*):
        kelly_fraction = (b * p - q) / b
    where p = fair probability of winning, q = 1 - p, b = net decimal payout.

    A fractional Kelly (e.g. fraction=0.25 for quarter-Kelly) is common in
    practice to reduce variance.

    Parameters
    ----------
    book_odds : American odds offered by the sportsbook
    fair_prob : True (no-vig) probability of winning this side (0-1)
    fraction  : Kelly fraction to apply (1.0 = full Kelly, 0.5 = half, 0.25 = quarter).
                Default 1.0.

    Returns
    -------
    dict with keys:
        full_kelly_pct      – full Kelly as a bankroll percentage
        recommended_pct     – fractional Kelly (adjusted by *fraction*)
        edge                – your probability edge (fair_prob - book_implied_prob)
        decimal_odds        – the book odds as decimal
        fair_prob           – the fair probability used
        book_implied_prob   – implied probability from the book odds
        is_positive         – True if Kelly recommends a bet (edge > 0)
        bankroll_example    – example wager on a $1 000 bankroll
    """
    book_prob = implied_probability(book_odds)

    # Decimal payout per $1 wagered (net, excluding stake)
    if book_odds < 0:
        b = 100 / abs(book_odds)
    elif book_odds > 0:
        b = book_odds / 100
    else:
        b = 1.0  # even money

    p = fair_prob
    q = 1 - p

    # Kelly formula: f* = (b*p - q) / b
    if b == 0:
        full_kelly = 0.0
    else:
        full_kelly = (b * p - q) / b

    # Clamp: never recommend negative sizing (no edge) or > 100%
    full_kelly = max(0.0, min(full_kelly, 1.0))
    recommended = full_kelly * fraction

    bankroll_example = round(recommended * 1000, 2)

    sign = "+" if (p - book_prob) >= 0 else ""
    return {
        "full_kelly_pct": f"{round(full_kelly * 100, 2)}%",
        "recommended_pct": f"{round(recommended * 100, 2)}%",
        "recommended_fraction": round(recommended, 6),
        "edge": round(p - book_prob, 6),
        "edge_pct": f"{sign}{round((p - book_prob) * 100, 2)}%",
        "decimal_odds": round(b + 1, 4),
        "fair_prob": round(p, 6),
        "book_implied_prob": round(book_prob, 6),
        "is_positive": full_kelly > 0,
        "bankroll_example": f"${bankroll_example} on a $1,000 bankroll",
        "kelly_fraction_used": fraction,
    }
