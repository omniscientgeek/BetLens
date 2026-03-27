"""
Odds mathematics utilities for the detection phase.

Provides:
1. Implied Probability  – convert American odds → probability
2. Vig (Vigorish)       – bookmaker margin per market
3. No-Vig / Fair Odds   – true probability with vig removed
4. Expected Value (EV)  – edge vs consensus fair probability
5. Bayesian Updating    – Beta-conjugate updating of true probabilities
6. Shin Model           – asymmetric margin decomposition (longshot bias)
"""

from math import sqrt


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


# ---------------------------------------------------------------------------
# 3b. Shin Model — asymmetric margin decomposition (longshot bias)
# ---------------------------------------------------------------------------


def _shin_solve_z(probs: list[float], tol: float = 1e-10, max_iter: int = 200) -> float:
    """Solve for the Shin parameter *z* (insider trading fraction) via bisection.

    Given raw implied probabilities *probs* (which sum to S > 1 due to vig),
    find z ∈ (0, 1) such that the Shin-adjusted true probabilities sum to 1.

    The Shin formula for each outcome i:
        q_i = ( sqrt(z² + 4·(1-z)·p_i² / S²) − z ) / ( 2·(1-z) )

    We search for z where  Σ q_i(z) = 1.

    Parameters
    ----------
    probs : list of raw implied probabilities (sum > 1.0)
    tol   : convergence tolerance for bisection
    max_iter : maximum bisection iterations

    Returns
    -------
    z : the Shin parameter (typically 0.01–0.10 for normal markets)
    """
    S = sum(probs)
    if S <= 1.0:
        # No vig to decompose; return z=0
        return 0.0

    def _sum_shin_probs(z: float) -> float:
        """Sum of Shin-adjusted probabilities for a given z."""
        if z >= 1.0:
            return 0.0
        total = 0.0
        denom = 2.0 * (1.0 - z)
        for p in probs:
            discriminant = z * z + 4.0 * (1.0 - z) * (p / S) ** 2
            total += (sqrt(discriminant) - z) / denom
        return total

    # Bisection: z_lo → sum > 1, z_hi → sum < 1
    z_lo, z_hi = 0.0, 1.0 - 1e-12
    for _ in range(max_iter):
        z_mid = (z_lo + z_hi) / 2.0
        s = _sum_shin_probs(z_mid)
        if abs(s - 1.0) < tol:
            return z_mid
        if s > 1.0:
            z_lo = z_mid
        else:
            z_hi = z_mid
    return (z_lo + z_hi) / 2.0


def shin_probabilities(odds_side_a: int | float, odds_side_b: int | float) -> dict:
    """Remove vig using the Shin model (asymmetric margin decomposition).

    Unlike naive proportional scaling (``no_vig_probabilities``), the Shin
    method allocates *more* vig to the longshot side, which reflects how
    sportsbooks actually set their margins.  This produces more accurate
    "true" probabilities, especially for lopsided markets.

    **How it works:**

    The Shin model (Shin 1991, 1993) assumes a fraction *z* of bettors are
    insiders (informed traders).  The bookmaker protects against these insiders
    by inflating odds, but the inflation is *not symmetric* — longshots get
    inflated more because an insider betting on a longshot risks less capital
    for a larger payout.  The parameter *z* is solved numerically so that the
    resulting true probabilities sum to exactly 1.

    For each outcome *i* with raw implied probability *p_i* and total
    overround *S = Σp_i*:

        q_i = ( √(z² + 4·(1−z)·(p_i/S)²) − z ) / ( 2·(1−z) )

    Parameters
    ----------
    odds_side_a : American odds for side A (e.g. home / over)
    odds_side_b : American odds for side B (e.g. away / under)

    Returns
    -------
    dict with keys:
        shin_a       – Shin-adjusted true probability of side A (0-1)
        shin_b       – Shin-adjusted true probability of side B (0-1)
        shin_a_pct   – as percentage string
        shin_b_pct   – as percentage string
        naive_a      – naive proportional fair prob for comparison
        naive_b      – naive proportional fair prob for comparison
        delta_a      – shin_a - naive_a (positive = Shin gives more prob)
        delta_b      – shin_b - naive_b
        z            – the Shin parameter (insider trading fraction)
        z_pct        – z as a percentage string
        vig_pct      – the total vig (overround) that was decomposed
        vig_on_a     – vig allocated to side A by the Shin model
        vig_on_b     – vig allocated to side B by the Shin model
        vig_on_a_pct – as percentage string
        vig_on_b_pct – as percentage string
        method       – "shin"
    """
    prob_a = implied_probability(odds_side_a)
    prob_b = implied_probability(odds_side_b)
    S = prob_a + prob_b
    vig = S - 1.0

    if S <= 1.0:
        # No vig — Shin can't decompose; return raw probs
        return {
            "shin_a": round(prob_a, 6),
            "shin_b": round(prob_b, 6),
            "shin_a_pct": f"{round(prob_a * 100, 2)}%",
            "shin_b_pct": f"{round(prob_b * 100, 2)}%",
            "naive_a": round(prob_a, 6),
            "naive_b": round(prob_b, 6),
            "delta_a": 0.0,
            "delta_b": 0.0,
            "z": 0.0,
            "z_pct": "0.0%",
            "vig_pct": "0.0%",
            "vig_on_a": 0.0,
            "vig_on_b": 0.0,
            "vig_on_a_pct": "0.0%",
            "vig_on_b_pct": "0.0%",
            "method": "shin",
        }

    # Solve for z
    z = _shin_solve_z([prob_a, prob_b])

    # Compute Shin-adjusted true probabilities
    denom = 2.0 * (1.0 - z)
    disc_a = z * z + 4.0 * (1.0 - z) * (prob_a / S) ** 2
    disc_b = z * z + 4.0 * (1.0 - z) * (prob_b / S) ** 2
    shin_a = (sqrt(disc_a) - z) / denom
    shin_b = (sqrt(disc_b) - z) / denom

    # Naive proportional for comparison
    naive_a = prob_a / S
    naive_b = prob_b / S

    # Vig allocated to each side: implied_prob - shin_true_prob
    vig_on_a = prob_a - shin_a
    vig_on_b = prob_b - shin_b

    return {
        "shin_a": round(shin_a, 6),
        "shin_b": round(shin_b, 6),
        "shin_a_pct": f"{round(shin_a * 100, 2)}%",
        "shin_b_pct": f"{round(shin_b * 100, 2)}%",
        "naive_a": round(naive_a, 6),
        "naive_b": round(naive_b, 6),
        "delta_a": round(shin_a - naive_a, 6),
        "delta_b": round(shin_b - naive_b, 6),
        "z": round(z, 6),
        "z_pct": f"{round(z * 100, 4)}%",
        "vig_pct": f"{round(vig * 100, 2)}%",
        "vig_on_a": round(vig_on_a, 6),
        "vig_on_b": round(vig_on_b, 6),
        "vig_on_a_pct": f"{round(vig_on_a * 100, 2)}%",
        "vig_on_b_pct": f"{round(vig_on_b * 100, 2)}%",
        "method": "shin",
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


# ---------------------------------------------------------------------------
# 5. Bayesian Updating — Beta-conjugate probability updates
# ---------------------------------------------------------------------------

def _prob_to_beta_params(prob: float, kappa: float) -> tuple[float, float]:
    """Convert a probability into Beta distribution parameters (alpha, beta).

    Uses a concentration parameter *kappa* (equivalent sample size) to control
    how strongly the probability translates into a prior.  Higher kappa means
    more confidence (tighter distribution).

    Parameters
    ----------
    prob  : Probability to encode (0-1).
    kappa : Concentration / equivalent sample size.  Typical values:
            - 10-20 for a moderately informative prior (e.g., Pinnacle)
            -  3-5  for weak evidence (e.g., a single recreational book)

    Returns
    -------
    (alpha, beta) parameters for a Beta distribution.
    """
    alpha = prob * kappa
    beta = (1 - prob) * kappa
    return (alpha, beta)


def bayesian_update(
    prior_prob: float,
    evidence_probs: list[float],
    prior_kappa: float = 20.0,
    evidence_kappa: float = 5.0,
) -> dict:
    """Bayesian update of a true probability using Beta-conjugate updating.

    Starts with a prior probability (e.g., from Pinnacle's no-vig line)
    encoded as a Beta(alpha, beta) distribution, then sequentially
    incorporates each additional sportsbook's no-vig probability as
    pseudo-observations that shift the posterior.

    Each evidence observation contributes *evidence_kappa* worth of
    pseudo-counts, split proportionally by the book's fair probability.
    This means a book posting 60% adds 0.6 * evidence_kappa successes
    and 0.4 * evidence_kappa failures to the running Beta posterior.

    Parameters
    ----------
    prior_prob      : Starting probability (0-1), typically Pinnacle no-vig.
    evidence_probs  : List of additional probabilities from other sportsbooks
                      (each should be a no-vig fair probability, 0-1).
    prior_kappa     : Concentration for the prior.  Higher = more trust in
                      Pinnacle.  Default 20 (moderate-strong prior).
    evidence_kappa  : Concentration per evidence sportsbook.  Higher = each
                      book shifts the posterior more.  Default 5.

    Returns
    -------
    dict with keys:
        prior_prob        – the prior probability used
        posterior_prob     – final Bayesian posterior mean
        posterior_prob_pct – as percentage string
        prior_alpha       – prior Beta alpha
        prior_beta        – prior Beta beta
        posterior_alpha   – posterior Beta alpha
        posterior_beta    – posterior Beta beta
        credible_interval_90 – approximate 90% credible interval (mean ± 1.645 * std)
        evidence_count    – number of evidence observations incorporated
        total_kappa       – total equivalent sample size (prior + all evidence)
        shift_from_prior  – how much the posterior moved from the prior (pp)
        update_trace      – step-by-step trace showing how each book shifted the posterior
    """
    # Clamp probabilities to avoid degenerate Beta params
    prior_prob = max(0.001, min(0.999, prior_prob))

    # Initialize prior Beta parameters
    alpha, beta = _prob_to_beta_params(prior_prob, prior_kappa)

    trace = [{
        "step": 0,
        "source": "prior",
        "input_prob": round(prior_prob, 6),
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "posterior_mean": round(alpha / (alpha + beta), 6),
    }]

    # Sequentially update with each evidence probability
    for i, ev_prob in enumerate(evidence_probs):
        ev_prob = max(0.001, min(0.999, ev_prob))
        # Add pseudo-counts proportional to this book's fair probability
        alpha += ev_prob * evidence_kappa
        beta += (1 - ev_prob) * evidence_kappa

        posterior_mean = alpha / (alpha + beta)
        trace.append({
            "step": i + 1,
            "source": f"evidence_{i + 1}",
            "input_prob": round(ev_prob, 6),
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "posterior_mean": round(posterior_mean, 6),
        })

    # Final posterior statistics
    posterior_mean = alpha / (alpha + beta)
    total_kappa = alpha + beta

    # Beta standard deviation: sqrt(alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)))
    variance = (alpha * beta) / ((total_kappa ** 2) * (total_kappa + 1))
    std_dev = variance ** 0.5

    # Approximate 90% credible interval (mean ± 1.645 * std)
    ci_low = max(0.0, posterior_mean - 1.645 * std_dev)
    ci_high = min(1.0, posterior_mean + 1.645 * std_dev)

    shift = posterior_mean - prior_prob
    sign = "+" if shift >= 0 else ""

    return {
        "prior_prob": round(prior_prob, 6),
        "posterior_prob": round(posterior_mean, 6),
        "posterior_prob_pct": f"{round(posterior_mean * 100, 2)}%",
        "prior_alpha": round(_prob_to_beta_params(prior_prob, prior_kappa)[0], 4),
        "prior_beta": round(_prob_to_beta_params(prior_prob, prior_kappa)[1], 4),
        "posterior_alpha": round(alpha, 4),
        "posterior_beta": round(beta, 4),
        "std_dev": round(std_dev, 6),
        "credible_interval_90": {
            "low": round(ci_low, 6),
            "high": round(ci_high, 6),
            "low_pct": f"{round(ci_low * 100, 2)}%",
            "high_pct": f"{round(ci_high * 100, 2)}%",
        },
        "evidence_count": len(evidence_probs),
        "total_kappa": round(total_kappa, 4),
        "shift_from_prior": round(shift, 6),
        "shift_from_prior_pct": f"{sign}{round(shift * 100, 2)}pp",
        "update_trace": trace,
    }
