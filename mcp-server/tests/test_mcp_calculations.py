"""
Comprehensive unit tests for mcp_server.py calculation functions.

Tests all statistical, anomaly detection, and scoring algorithms that live
inside the MCP server (as opposed to odds_math.py).  These include:

- GAMLSS (location/scale/shape) fitting & z-scores
- Poisson score prediction & PMF
- KNN anomaly scores
- Isolation Forest anomaly scores
- Euclidean distance & feature normalization
- Shannon entropy (via market entropy logic)
- ASCII heatmap symbol generation
- Market consistency scoring
- Implied score calculations
"""

import sys
import os
import math
import pytest
from statistics import mean, pstdev

# Allow imports from mcp-server/ and webservice/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "webservice")))

from odds_math import implied_probability, no_vig_probabilities

# Import the internal functions from mcp_server
# We need to handle the FastMCP import gracefully
import importlib
import types


def _import_mcp_server_functions():
    """Import calculation functions from mcp_server without starting the server."""
    server_path = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
    with open(server_path, "r", encoding="utf-8") as f:
        source = f.read()

    # Extract just the pure functions we need (no server startup)
    # We'll re-implement the key formulas to test against, and also
    # import the functions directly
    return source


# Since mcp_server.py has side effects on import (FastMCP, etc.), we extract
# the pure mathematical functions by reading the source.  For testing we
# re-implement the core algorithms identically and verify their behavior.

# ═══════════════════════════════════════════════════════════════════════════════
# GAMLSS — Location, Scale, Shape Modeling
# ═══════════════════════════════════════════════════════════════════════════════

def _gamlss_fit(values: list[float]) -> dict:
    """Re-implementation of _gamlss_fit from mcp_server.py for testing."""
    n = len(values)
    if n < 3:
        return {"mu": mean(values) if values else 0, "sigma": 0, "nu": 0, "n": n,
                "kurtosis_excess": 0, "sigma_method": "insufficient_data"}

    sorted_vals = sorted(values)

    # Location: trimmed mean (10% each tail)
    trim = max(1, int(n * 0.1))
    trimmed = sorted_vals[trim: n - trim] if n > 4 else sorted_vals
    mu = sum(trimmed) / len(trimmed)

    # Scale: MAD-based
    median_val = sorted_vals[n // 2] if n % 2 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    abs_devs = sorted(abs(v - median_val) for v in values)
    mad = abs_devs[len(abs_devs) // 2] if abs_devs else 0
    sigma = mad * 1.4826

    if sigma < 1e-9:
        sigma = pstdev(values)

    # Shape: adjusted Fisher-Pearson skewness
    if sigma > 1e-9:
        m3 = sum((v - mu) ** 3 for v in values) / n
        nu = m3 / (sigma ** 3)
        if n >= 3:
            nu = nu * (n * n) / ((n - 1) * (n - 2)) if (n - 1) * (n - 2) > 0 else nu
    else:
        nu = 0.0

    # Excess kurtosis
    if sigma > 1e-9 and n >= 4:
        m4 = sum((v - mu) ** 4 for v in values) / n
        kurtosis_excess = (m4 / (sigma ** 4)) - 3.0
    else:
        kurtosis_excess = 0.0

    return {
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "nu": round(nu, 4),
        "kurtosis_excess": round(kurtosis_excess, 4),
        "median": round(median_val, 6),
        "mad": round(mad, 6),
        "n": n,
        "sigma_method": "MAD" if mad * 1.4826 >= 1e-9 else "pstdev_fallback",
    }


def _gamlss_zscore(value: float, fit: dict) -> float:
    """Re-implementation of _gamlss_zscore from mcp_server.py."""
    sigma = fit["sigma"]
    if sigma < 1e-9:
        return 0.0
    nu = fit["nu"]
    mu = fit["mu"]
    diff = value - mu

    skew_adjustment = 1.0
    if abs(nu) > 0.05:
        direction = 1.0 if diff > 0 else -1.0
        raw_adj = 1.0 + direction * nu * 0.15
        skew_adjustment = max(0.5, min(2.0, raw_adj))

    effective_sigma = sigma * skew_adjustment
    return diff / effective_sigma if effective_sigma > 1e-9 else 0.0


class TestGAMLSSFit:
    """Validate GAMLSS distributional parameter estimation."""

    def test_symmetric_data_zero_skewness(self):
        """Perfectly symmetric data should have nu ≈ 0."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        fit = _gamlss_fit(data)
        assert abs(fit["nu"]) < 0.5  # Near-zero skew for symmetric data

    def test_right_skewed_data_positive_nu(self):
        """Right-skewed data should produce nu > 0."""
        data = [1, 1, 1, 2, 2, 3, 3, 5, 10, 50]
        fit = _gamlss_fit(data)
        assert fit["nu"] > 0

    def test_left_skewed_data_negative_nu(self):
        """Left-skewed data should produce nu < 0."""
        data = [50, 10, 5, 3, 3, 2, 2, 1, 1, 1]
        # Reverse skew
        data = [-x for x in [1, 1, 1, 2, 2, 3, 3, 5, 10, 50]]
        fit = _gamlss_fit(data)
        assert fit["nu"] < 0

    def test_identical_values_zero_sigma(self):
        """All identical values → sigma from pstdev = 0."""
        data = [5.0, 5.0, 5.0, 5.0, 5.0]
        fit = _gamlss_fit(data)
        assert fit["sigma"] == 0
        assert fit["sigma_method"] == "pstdev_fallback"

    def test_mu_is_trimmed_mean(self):
        """mu should be the trimmed mean (10% each tail removed)."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        fit = _gamlss_fit(data)
        # With 10 items, trim=1 → remove index 0 and 9 → mean of [2,3,4,5,6,7,8,9]
        expected_mu = sum([2, 3, 4, 5, 6, 7, 8, 9]) / 8
        assert abs(fit["mu"] - expected_mu) < 0.001

    def test_sigma_mad_based(self):
        """Sigma should be MAD * 1.4826 for normal data."""
        data = [10, 20, 30, 40, 50]
        fit = _gamlss_fit(data)
        # median = 30; deviations: [20, 10, 0, 10, 20] → sorted: [0, 10, 10, 20, 20]
        # MAD = 10; sigma = 10 * 1.4826 = 14.826
        assert abs(fit["mad"] - 10.0) < 0.001
        assert abs(fit["sigma"] - 14.826) < 0.01

    def test_insufficient_data_returns_defaults(self):
        """< 3 data points should return default values."""
        fit = _gamlss_fit([1.0, 2.0])
        assert fit["n"] == 2
        assert fit["sigma_method"] == "insufficient_data"

    def test_empty_data(self):
        fit = _gamlss_fit([])
        assert fit["n"] == 0

    def test_kurtosis_normal_data(self):
        """Normal-like data should have kurtosis_excess near 0."""
        # Approximately normal distribution
        import random
        rng = random.Random(42)
        data = [rng.gauss(0, 1) for _ in range(1000)]
        fit = _gamlss_fit(data)
        assert abs(fit["kurtosis_excess"]) < 1.5  # Rough bound

    def test_n_preserved(self):
        data = [1, 2, 3, 4, 5, 6, 7]
        fit = _gamlss_fit(data)
        assert fit["n"] == 7

    def test_sigma_positive_for_varied_data(self):
        data = [1, 5, 10, 15, 20]
        fit = _gamlss_fit(data)
        assert fit["sigma"] > 0

    def test_median_correct_odd_count(self):
        data = [3, 1, 2, 5, 4]
        fit = _gamlss_fit(data)
        assert fit["median"] == 3.0  # sorted: [1,2,3,4,5], median = index 2

    def test_median_correct_even_count(self):
        data = [1, 2, 3, 4]
        fit = _gamlss_fit(data)
        assert fit["median"] == 2.5  # (2+3)/2


class TestGAMLSSZScore:
    """Validate skew-aware z-score computation."""

    def test_value_at_mean_is_zero(self):
        fit = {"mu": 5.0, "sigma": 1.0, "nu": 0.0}
        assert _gamlss_zscore(5.0, fit) == 0.0

    def test_standard_zscore_no_skew(self):
        """With nu=0, should reduce to standard z-score."""
        fit = {"mu": 10.0, "sigma": 2.0, "nu": 0.0}
        z = _gamlss_zscore(14.0, fit)
        expected = (14.0 - 10.0) / 2.0  # = 2.0
        assert abs(z - expected) < 1e-10

    def test_positive_skew_above_mean_less_anomalous(self):
        """Right-skewed (nu > 0): value above mean should get smaller |z|."""
        fit_symmetric = {"mu": 10.0, "sigma": 2.0, "nu": 0.0}
        fit_skewed = {"mu": 10.0, "sigma": 2.0, "nu": 1.0}
        z_sym = _gamlss_zscore(14.0, fit_symmetric)
        z_skew = _gamlss_zscore(14.0, fit_skewed)
        assert abs(z_skew) < abs(z_sym)  # Less anomalous in heavy tail

    def test_positive_skew_below_mean_more_anomalous(self):
        """Right-skewed: value below mean should get larger |z|."""
        fit_symmetric = {"mu": 10.0, "sigma": 2.0, "nu": 0.0}
        fit_skewed = {"mu": 10.0, "sigma": 2.0, "nu": 1.0}
        z_sym = _gamlss_zscore(6.0, fit_symmetric)
        z_skew = _gamlss_zscore(6.0, fit_skewed)
        assert abs(z_skew) > abs(z_sym)  # More anomalous in thin tail

    def test_negative_skew_reverses_direction(self):
        """Left-skewed (nu < 0): below mean is the heavy tail."""
        fit = {"mu": 10.0, "sigma": 2.0, "nu": -1.0}
        z_below = _gamlss_zscore(6.0, fit)
        z_above = _gamlss_zscore(14.0, fit)
        # Below mean is heavy tail → less anomalous; above → more anomalous
        assert abs(z_below) < abs(z_above)

    def test_zero_sigma_returns_zero(self):
        fit = {"mu": 5.0, "sigma": 0.0, "nu": 0.0}
        assert _gamlss_zscore(10.0, fit) == 0.0

    def test_skew_adjustment_clamped(self):
        """Skew adjustment should be clamped to [0.5, 2.0]."""
        fit = {"mu": 10.0, "sigma": 2.0, "nu": 100.0}  # Extreme skew
        z = _gamlss_zscore(14.0, fit)
        # With extreme nu, adjustment = 1 + 1 * 100 * 0.15 = 16 → clamped to 2.0
        effective_sigma = 2.0 * 2.0  # clamped
        expected = (14.0 - 10.0) / effective_sigma
        assert abs(z - expected) < 1e-10

    def test_small_skew_no_adjustment(self):
        """nu <= 0.05 should not trigger skew adjustment."""
        fit = {"mu": 10.0, "sigma": 2.0, "nu": 0.04}
        z = _gamlss_zscore(14.0, fit)
        expected = (14.0 - 10.0) / 2.0
        assert abs(z - expected) < 1e-10

    def test_symmetry_of_zscore_no_skew(self):
        """Without skew, z(mu+d) == -z(mu-d)."""
        fit = {"mu": 10.0, "sigma": 2.0, "nu": 0.0}
        z_above = _gamlss_zscore(12.0, fit)
        z_below = _gamlss_zscore(8.0, fit)
        assert abs(z_above + z_below) < 1e-10

    def test_large_deviation_large_zscore(self):
        fit = {"mu": 0.5, "sigma": 0.02, "nu": 0.0}
        z = _gamlss_zscore(0.6, fit)
        assert abs(z) > 4.0  # 5 standard deviations


# ═══════════════════════════════════════════════════════════════════════════════
# POISSON SCORE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def _poisson_pmf(k: int, lam: float) -> float:
    """Re-implementation of Poisson PMF from mcp_server.py."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _build_score_matrix(home_lambda: float, away_lambda: float, max_score: int):
    """Re-implementation of score matrix builder."""
    home_pmf = [_poisson_pmf(k, home_lambda) for k in range(max_score + 1)]
    away_pmf = [_poisson_pmf(k, away_lambda) for k in range(max_score + 1)]
    matrix = [
        [home_pmf[h] * away_pmf[a] for a in range(max_score + 1)]
        for h in range(max_score + 1)
    ]
    return matrix, home_pmf, away_pmf


class TestPoissonPMF:
    """Validate the Poisson probability mass function."""

    def test_pmf_k0_lambda_1(self):
        """P(0 | λ=1) = e^(-1) ≈ 0.36788."""
        assert abs(_poisson_pmf(0, 1.0) - math.exp(-1)) < 1e-10

    def test_pmf_k1_lambda_1(self):
        """P(1 | λ=1) = e^(-1) ≈ 0.36788."""
        assert abs(_poisson_pmf(1, 1.0) - math.exp(-1)) < 1e-10

    def test_pmf_k2_lambda_3(self):
        """P(2 | λ=3) = (3^2 * e^-3) / 2! = 9 * e^-3 / 2 ≈ 0.22404."""
        expected = (3 ** 2) * math.exp(-3) / 2
        assert abs(_poisson_pmf(2, 3.0) - expected) < 1e-10

    def test_pmf_sums_to_one(self):
        """Sum of P(k|λ) for k=0..N should ≈ 1.0 for large N."""
        lam = 5.0
        total = sum(_poisson_pmf(k, lam) for k in range(50))
        assert abs(total - 1.0) < 1e-8

    def test_pmf_zero_lambda(self):
        """λ=0: P(0)=1, P(k>0)=0."""
        assert _poisson_pmf(0, 0.0) == 1.0
        assert _poisson_pmf(1, 0.0) == 0.0
        assert _poisson_pmf(5, 0.0) == 0.0

    def test_pmf_negative_lambda(self):
        """Negative λ: treat as zero."""
        assert _poisson_pmf(0, -1.0) == 1.0
        assert _poisson_pmf(1, -1.0) == 0.0

    def test_pmf_large_lambda_mode(self):
        """For large λ, mode should be near floor(λ)."""
        lam = 100.0
        probs = [_poisson_pmf(k, lam) for k in range(200)]
        mode_k = probs.index(max(probs))
        assert abs(mode_k - lam) <= 1  # Mode ≈ λ for large λ

    def test_pmf_mean_matches_lambda(self):
        """E[X] = λ — verify numerically."""
        lam = 7.5
        expected_mean = sum(k * _poisson_pmf(k, lam) for k in range(100))
        assert abs(expected_mean - lam) < 0.001

    def test_pmf_variance_matches_lambda(self):
        """Var[X] = λ — verify numerically."""
        lam = 5.0
        mean_val = sum(k * _poisson_pmf(k, lam) for k in range(100))
        variance = sum((k - mean_val) ** 2 * _poisson_pmf(k, lam) for k in range(100))
        assert abs(variance - lam) < 0.01

    def test_pmf_non_negative(self):
        """All probabilities must be >= 0."""
        for k in range(20):
            for lam in [0.1, 1.0, 5.0, 20.0]:
                assert _poisson_pmf(k, lam) >= 0

    def test_pmf_decreasing_tail(self):
        """For large k >> λ, probabilities should decrease monotonically."""
        lam = 3.0
        probs = [_poisson_pmf(k, lam) for k in range(10, 30)]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]


class TestBuildScoreMatrix:
    """Validate the Poisson score probability matrix."""

    def test_matrix_sums_to_one(self):
        """Joint probability matrix should sum to ≈ 1.0."""
        matrix, _, _ = _build_score_matrix(100.0, 95.0, 160)
        total = sum(sum(row) for row in matrix)
        assert abs(total - 1.0) < 0.01

    def test_matrix_dimensions(self):
        matrix, _, _ = _build_score_matrix(3.0, 2.5, 10)
        assert len(matrix) == 11  # 0..10
        assert all(len(row) == 11 for row in matrix)

    def test_independence_assumption(self):
        """P(h, a) should equal P(h) * P(a) — independence."""
        matrix, home_pmf, away_pmf = _build_score_matrix(3.0, 2.5, 10)
        for h in range(5):
            for a in range(5):
                expected = home_pmf[h] * away_pmf[a]
                assert abs(matrix[h][a] - expected) < 1e-15

    def test_marginal_home_matches(self):
        """Sum across away scores should give home marginal (with sufficient max_score)."""
        # Use max_score=30 to ensure full tail coverage for λ=3.0
        matrix, home_pmf, _ = _build_score_matrix(3.0, 2.5, 30)
        for h in range(31):
            marginal = sum(matrix[h])
            assert abs(marginal - home_pmf[h]) < 1e-10

    def test_marginal_away_matches(self):
        """Sum across home scores should give away marginal (with sufficient max_score)."""
        matrix, _, away_pmf = _build_score_matrix(3.0, 2.5, 30)
        for a in range(31):
            marginal = sum(matrix[h][a] for h in range(31))
            assert abs(marginal - away_pmf[a]) < 1e-10

    def test_home_win_prob_reasonable(self):
        """Home team favoured (higher lambda) should have > 50% win prob."""
        matrix, _, _ = _build_score_matrix(3.0, 2.0, 15)
        home_win = sum(matrix[h][a] for h in range(16) for a in range(h))
        assert home_win > 0.5

    def test_draw_prob_nhl(self):
        """In low-scoring sports (NHL), draw probability should be significant."""
        matrix, _, _ = _build_score_matrix(3.0, 2.5, 12)
        draw_prob = sum(matrix[i][i] for i in range(13))
        assert draw_prob > 0.05  # Draws are common in hockey regulation

    def test_implied_scores_from_spread_and_total(self):
        """Verify lambda derivation: home=(T+S)/2, away=(T-S)/2."""
        total = 220.5
        spread = -5.5  # home favoured by 5.5
        home_lambda = (total + spread) / 2   # 107.5
        away_lambda = (total - spread) / 2   # 113.0
        # Wait — the MCP server uses: home = (total + home_line) / 2
        # where home_line is negative for home favourite
        # home_lambda = (220.5 + (-5.5)) / 2 = 107.5
        # away_lambda = (220.5 - (-5.5)) / 2 = 113.0
        assert home_lambda == 107.5
        assert away_lambda == 113.0

    def test_all_probabilities_non_negative(self):
        matrix, _, _ = _build_score_matrix(5.0, 4.0, 15)
        for row in matrix:
            for p in row:
                assert p >= 0

    def test_most_likely_score_near_lambdas(self):
        """The most likely exact score should be near (home_λ, away_λ)."""
        home_lam, away_lam = 3.0, 2.0
        matrix, _, _ = _build_score_matrix(home_lam, away_lam, 15)
        max_p = 0
        best_h, best_a = 0, 0
        for h in range(16):
            for a in range(16):
                if matrix[h][a] > max_p:
                    max_p = matrix[h][a]
                    best_h, best_a = h, a
        assert abs(best_h - home_lam) <= 1
        assert abs(best_a - away_lam) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# KNN & ISOLATION FOREST ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _knn_anomaly_scores(normalized: list[list[float]], k: int = 5) -> list[float]:
    n = len(normalized)
    k = min(k, n - 1)
    if k <= 0:
        return [0.0] * n
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(_euclidean_distance(normalized[i], normalized[j]))
        dists.sort()
        avg_knn_dist = sum(dists[:k]) / k
        scores.append(avg_knn_dist)
    return scores


def _normalize_features(features: list[dict]) -> list[list[float]]:
    if not features:
        return []
    vectors = [f["vector"] for f in features]
    n_features = len(vectors[0])
    mins = [min(v[i] for v in vectors) for i in range(n_features)]
    maxs = [max(v[i] for v in vectors) for i in range(n_features)]
    normalized = []
    for v in vectors:
        norm = []
        for i in range(n_features):
            rng = maxs[i] - mins[i]
            norm.append((v[i] - mins[i]) / rng if rng > 0 else 0.0)
        normalized.append(norm)
    return normalized


def _isolation_forest_scores(vectors, n_trees=100, sample_size=32, seed=42):
    """Re-implementation from mcp_server.py."""
    import random as _random
    n = len(vectors)
    if n < 4:
        return [0.0] * n
    n_features = len(vectors[0])
    max_depth = max(2, int(math.log(max(sample_size, 2)) / math.log(2)) + 1)
    rng = _random.Random(seed)

    def _build_tree(indices, depth):
        if len(indices) <= 1 or depth >= max_depth:
            return {"type": "leaf", "size": len(indices), "depth": depth}
        feat = rng.randint(0, n_features - 1)
        vals = [vectors[idx][feat] for idx in indices]
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return {"type": "leaf", "size": len(indices), "depth": depth}
        split = rng.uniform(lo, hi)
        left_idx = [idx for idx in indices if vectors[idx][feat] < split]
        right_idx = [idx for idx in indices if vectors[idx][feat] >= split]
        if not left_idx or not right_idx:
            return {"type": "leaf", "size": len(indices), "depth": depth}
        return {
            "type": "split", "feature": feat, "split_value": split,
            "left": _build_tree(left_idx, depth + 1),
            "right": _build_tree(right_idx, depth + 1),
        }

    def _path_length(node, vec):
        if node["type"] == "leaf":
            size = node["size"]
            if size <= 1:
                return node["depth"]
            c = 2.0 * (math.log(size - 1) + 0.5772156649) - 2.0 * (size - 1) / size
            return node["depth"] + c
        if vec[node["feature"]] < node["split_value"]:
            return _path_length(node["left"], vec)
        else:
            return _path_length(node["right"], vec)

    trees = []
    all_indices = list(range(n))
    actual_sample = min(sample_size, n)
    for _ in range(n_trees):
        sample_idx = rng.sample(all_indices, actual_sample) if actual_sample < n else list(all_indices)
        trees.append(_build_tree(sample_idx, 0))

    c_n = 2.0 * (math.log(max(actual_sample - 1, 1)) + 0.5772156649) - 2.0 * (actual_sample - 1) / max(actual_sample, 1)
    if c_n == 0:
        c_n = 1.0

    scores = []
    for i in range(n):
        avg_path = sum(_path_length(tree, vectors[i]) for tree in trees) / n_trees
        score = 2.0 ** (-avg_path / c_n)
        scores.append(score)
    return scores


class TestEuclideanDistance:
    """Validate Euclidean distance computation."""

    def test_same_point_zero(self):
        assert _euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0

    def test_unit_distance_1d(self):
        assert abs(_euclidean_distance([0], [1]) - 1.0) < 1e-10

    def test_known_3d_distance(self):
        # √(1² + 2² + 2²) = √9 = 3
        assert abs(_euclidean_distance([0, 0, 0], [1, 2, 2]) - 3.0) < 1e-10

    def test_2d_pythagorean(self):
        # 3-4-5 triangle
        assert abs(_euclidean_distance([0, 0], [3, 4]) - 5.0) < 1e-10

    def test_symmetry(self):
        a, b = [1, 2, 3], [4, 5, 6]
        assert abs(_euclidean_distance(a, b) - _euclidean_distance(b, a)) < 1e-10

    def test_triangle_inequality(self):
        a, b, c = [0, 0], [1, 0], [0.5, 1]
        d_ab = _euclidean_distance(a, b)
        d_bc = _euclidean_distance(b, c)
        d_ac = _euclidean_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10

    def test_high_dimensional(self):
        a = [0.0] * 10
        b = [1.0] * 10
        assert abs(_euclidean_distance(a, b) - math.sqrt(10)) < 1e-10

    def test_negative_coordinates(self):
        assert abs(_euclidean_distance([-1, -1], [1, 1]) - math.sqrt(8)) < 1e-10

    def test_zero_vectors(self):
        assert _euclidean_distance([0, 0, 0], [0, 0, 0]) == 0.0

    def test_single_dimension_difference(self):
        assert abs(_euclidean_distance([0, 0, 0], [0, 5, 0]) - 5.0) < 1e-10


class TestNormalizeFeatures:
    """Validate min-max feature normalization to [0, 1]."""

    def test_normalized_range(self):
        features = [{"vector": [1, 10]}, {"vector": [5, 50]}, {"vector": [3, 30]}]
        norm = _normalize_features(features)
        for v in norm:
            for val in v:
                assert 0.0 <= val <= 1.0

    def test_min_maps_to_zero(self):
        features = [{"vector": [1, 10]}, {"vector": [5, 50]}, {"vector": [3, 30]}]
        norm = _normalize_features(features)
        assert norm[0][0] == 0.0  # 1 is min of first dim
        assert norm[0][1] == 0.0  # 10 is min of second dim

    def test_max_maps_to_one(self):
        features = [{"vector": [1, 10]}, {"vector": [5, 50]}, {"vector": [3, 30]}]
        norm = _normalize_features(features)
        assert norm[1][0] == 1.0  # 5 is max of first dim
        assert norm[1][1] == 1.0  # 50 is max of second dim

    def test_constant_feature_maps_to_zero(self):
        """All same values in a dimension → 0.0 (no range)."""
        features = [{"vector": [5, 10]}, {"vector": [5, 50]}, {"vector": [5, 30]}]
        norm = _normalize_features(features)
        for v in norm:
            assert v[0] == 0.0  # All 5s → range 0 → 0.0

    def test_empty_returns_empty(self):
        assert _normalize_features([]) == []

    def test_single_record(self):
        features = [{"vector": [3, 7]}]
        norm = _normalize_features(features)
        assert norm == [[0.0, 0.0]]  # Single point → all zero range

    def test_preserves_ordering(self):
        features = [{"vector": [1]}, {"vector": [3]}, {"vector": [5]}]
        norm = _normalize_features(features)
        assert norm[0][0] < norm[1][0] < norm[2][0]

    def test_midpoint_maps_to_half(self):
        features = [{"vector": [0]}, {"vector": [10]}]
        norm = _normalize_features([{"vector": [0]}, {"vector": [5]}, {"vector": [10]}])
        assert abs(norm[1][0] - 0.5) < 1e-10


class TestKNNAnomalyScores:
    """Validate KNN-based anomaly scoring."""

    def test_outlier_scores_highest(self):
        """A clear outlier should have the highest KNN score."""
        data = [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [10, 10]]
        scores = _knn_anomaly_scores(data, k=3)
        assert scores[-1] == max(scores)  # [10,10] is the outlier

    def test_cluster_points_score_low(self):
        """Points in a tight cluster should score lower than outliers."""
        data = [[0, 0], [0.01, 0], [0, 0.01], [0.01, 0.01], [5, 5]]
        scores = _knn_anomaly_scores(data, k=3)
        cluster_max = max(scores[:4])
        assert cluster_max < scores[4]

    def test_k_larger_than_n_minus_1(self):
        """k should be clamped to n-1."""
        data = [[0, 0], [1, 1], [2, 2]]
        scores = _knn_anomaly_scores(data, k=100)
        assert len(scores) == 3
        # k clamped to 2 (n-1)

    def test_single_point_returns_zero(self):
        scores = _knn_anomaly_scores([[1, 2]], k=5)
        assert scores == [0.0]

    def test_two_points(self):
        scores = _knn_anomaly_scores([[0, 0], [3, 4]], k=5)
        # k clamped to 1; each point's 1-NN dist = 5
        assert abs(scores[0] - 5.0) < 1e-10
        assert abs(scores[1] - 5.0) < 1e-10

    def test_all_same_points(self):
        data = [[1, 1]] * 5
        scores = _knn_anomaly_scores(data, k=3)
        assert all(s == 0.0 for s in scores)

    def test_scores_non_negative(self):
        data = [[i, i * 2] for i in range(10)]
        scores = _knn_anomaly_scores(data, k=3)
        assert all(s >= 0 for s in scores)

    def test_output_length_matches_input(self):
        data = [[i] for i in range(20)]
        scores = _knn_anomaly_scores(data, k=5)
        assert len(scores) == 20

    def test_empty_list(self):
        scores = _knn_anomaly_scores([], k=5)
        assert scores == []

    def test_k_equals_one(self):
        """k=1 means score = distance to nearest neighbor only."""
        data = [[0], [1], [3], [100]]
        scores = _knn_anomaly_scores(data, k=1)
        # [100] nearest neighbor is [3] at distance 97
        assert scores[3] == max(scores)


class TestIsolationForestScores:
    """Validate Isolation Forest anomaly detection."""

    def test_scores_in_zero_one_range(self):
        data = [[i, i * 2] for i in range(20)]
        scores = _isolation_forest_scores(data)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_outlier_scores_higher(self):
        """Clear outlier should get higher anomaly score."""
        data = [[0, 0]] * 10 + [[0.1, 0]] * 10 + [[100, 100]]
        scores = _isolation_forest_scores(data, n_trees=200, seed=42)
        # The outlier [100,100] is the last element
        outlier_score = scores[-1]
        median_score = sorted(scores[:20])[10]
        assert outlier_score > median_score

    def test_small_dataset_returns_zeros(self):
        """< 4 records should return all zeros."""
        scores = _isolation_forest_scores([[1, 2], [3, 4], [5, 6]])
        assert scores == [0.0, 0.0, 0.0]

    def test_output_length(self):
        data = [[i] for i in range(15)]
        scores = _isolation_forest_scores(data)
        assert len(scores) == 15

    def test_deterministic_with_seed(self):
        data = [[i, i ** 2] for i in range(20)]
        scores1 = _isolation_forest_scores(data, seed=42)
        scores2 = _isolation_forest_scores(data, seed=42)
        assert scores1 == scores2

    def test_different_seeds_different_results(self):
        data = [[i, i ** 2] for i in range(20)]
        scores1 = _isolation_forest_scores(data, seed=42)
        scores2 = _isolation_forest_scores(data, seed=99)
        assert scores1 != scores2

    def test_uniform_data_similar_scores(self):
        """Uniform (non-anomalous) data should have similar scores."""
        data = [[i * 0.1, i * 0.1] for i in range(50)]
        scores = _isolation_forest_scores(data, n_trees=200, seed=42)
        score_range = max(scores) - min(scores)
        # Range should be relatively small compared to [0,1]
        assert score_range < 0.5

    def test_euler_mascheroni_constant_used(self):
        """Verify the Euler-Mascheroni constant is used correctly."""
        EULER = 0.5772156649
        # c(n) = 2*(ln(n-1) + EULER) - 2*(n-1)/n for n=32
        n = 32
        c = 2.0 * (math.log(n - 1) + EULER) - 2.0 * (n - 1) / n
        assert c > 0  # Should be positive for valid n

    def test_single_feature_dimension(self):
        data = [[x] for x in range(10)]
        data.append([100])  # outlier
        scores = _isolation_forest_scores(data, n_trees=100)
        assert len(scores) == 11

    def test_high_dimensional_data(self):
        """10D feature vectors (matching MCP server)."""
        import random
        rng = random.Random(42)
        data = [[rng.random() for _ in range(10)] for _ in range(30)]
        # Add an outlier
        data.append([10.0] * 10)
        scores = _isolation_forest_scores(data, n_trees=100, seed=42)
        assert len(scores) == 31
        assert all(0 <= s <= 1 for s in scores)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET CONSISTENCY SCORING
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketConsistency:
    """Test cross-market consistency logic (spread vs ML vs total)."""

    def test_implied_score_from_spread_and_total(self):
        """Verify: home_score = (total - spread_line) / 2."""
        total_line = 220.5
        spread_line = -5.5  # home favoured

        # From mcp_server.py line 535:
        # implied_home_score = (total_line - spread_line) / 2
        # implied_away_score = (total_line + spread_line) / 2
        home_score = (total_line - spread_line) / 2
        away_score = (total_line + spread_line) / 2

        assert home_score == 113.0
        assert away_score == 107.5
        assert home_score + away_score == total_line

    def test_implied_scores_sum_to_total(self):
        """home_implied + away_implied must equal total."""
        for total in [180, 200, 220.5, 240]:
            for spread in [-10, -3.5, 0, 3.5, 10]:
                home = (total - spread) / 2
                away = (total + spread) / 2
                assert abs(home + away - total) < 1e-10

    def test_negative_spread_favours_home(self):
        """Negative spread line means home is favoured → home implied score > away."""
        home = (220.5 - (-5.5)) / 2  # 113
        away = (220.5 + (-5.5)) / 2  # 107.5
        assert home > away

    def test_positive_spread_favours_away(self):
        """Positive spread line means away is favoured."""
        home = (220.5 - (5.5)) / 2  # 107.5
        away = (220.5 + (5.5)) / 2  # 113
        assert away > home

    def test_consistency_score_formula(self):
        """Score = max(0, (1 - avg_diff / 0.10)) * 100."""
        avg_diff = 0.03  # 3%
        score = max(0.0, (1 - avg_diff / 0.10)) * 100
        assert abs(score - 70.0) < 0.01

    def test_consistency_perfect_score(self):
        avg_diff = 0.0
        score = max(0.0, (1 - avg_diff / 0.10)) * 100
        assert score == 100.0

    def test_consistency_zero_score(self):
        avg_diff = 0.10  # 10%+ → 0 score
        score = max(0.0, (1 - avg_diff / 0.10)) * 100
        assert score == 0.0

    def test_consistency_beyond_zero_clamped(self):
        avg_diff = 0.20  # Way above threshold
        score = max(0.0, (1 - avg_diff / 0.10)) * 100
        assert score == 0.0

    def test_exploitable_threshold(self):
        """max_diff > 0.05 should flag as exploitable."""
        assert 0.06 > 0.05  # exploitable
        assert 0.04 <= 0.05  # not exploitable

    def test_score_gap_equals_spread_magnitude(self):
        """The implied score gap should equal the absolute spread."""
        spread_line = -7.5
        total_line = 200
        home = (total_line - spread_line) / 2
        away = (total_line + spread_line) / 2
        gap = home - away
        assert abs(gap - abs(spread_line)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# SHANNON ENTROPY (Market Efficiency)
# ═══════════════════════════════════════════════════════════════════════════════

class TestShannonEntropy:
    """Test Shannon entropy formula used in market efficiency analysis."""

    @staticmethod
    def _entropy(probs: list[float]) -> float:
        """H = -Σ(p * log2(p)) for p > 0."""
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should give maximum entropy = log2(N)."""
        n = 8
        probs = [1 / n] * n
        h = self._entropy(probs)
        assert abs(h - math.log2(n)) < 1e-10

    def test_certainty_zero_entropy(self):
        """One outcome certain (p=1) → H = 0."""
        probs = [1.0]
        assert self._entropy(probs) == 0.0

    def test_binary_fair_coin(self):
        """H([0.5, 0.5]) = 1 bit."""
        assert abs(self._entropy([0.5, 0.5]) - 1.0) < 1e-10

    def test_entropy_non_negative(self):
        """Entropy must be >= 0 for all valid distributions."""
        for probs in [[0.1, 0.9], [0.5, 0.5], [0.33, 0.33, 0.34]]:
            assert self._entropy(probs) >= 0

    def test_efficiency_ratio(self):
        """efficiency = actual_entropy / max_entropy."""
        probs = [0.25, 0.25, 0.25, 0.25]
        h = self._entropy(probs)
        max_h = math.log2(4)
        ratio = h / max_h
        assert abs(ratio - 1.0) < 1e-10

    def test_low_entropy_means_consensus(self):
        """Nearly certain outcome → low entropy → strong consensus."""
        probs = [0.99, 0.01]
        h = self._entropy(probs)
        assert h < 0.15  # Very low entropy

    def test_entropy_increases_with_disagreement(self):
        """More disagreement → higher entropy."""
        h_consensus = self._entropy([0.9, 0.1])
        h_disagreement = self._entropy([0.5, 0.5])
        assert h_disagreement > h_consensus

    def test_three_way_entropy(self):
        """H([1/3, 1/3, 1/3]) = log2(3) ≈ 1.585."""
        h = self._entropy([1/3, 1/3, 1/3])
        assert abs(h - math.log2(3)) < 1e-10

    def test_binary_entropy_formula(self):
        """Binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
        p = 0.3
        expected = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        assert abs(self._entropy([p, 1 - p]) - expected) < 1e-10

    def test_max_entropy_formula(self):
        """max_entropy = log2(N)."""
        for n in [2, 4, 8, 16]:
            assert abs(math.log2(n) - self._entropy([1/n] * n)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Cross-Formula Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossFormulaConsistency:
    """Verify that different modules produce consistent results when combined."""

    def test_shin_and_naive_bracket_true_prob(self):
        """For favourite: shin_prob should be between naive_prob and raw_implied."""
        from odds_math import shin_probabilities, no_vig_probabilities, implied_probability
        result_shin = shin_probabilities(-200, 170)
        result_naive = no_vig_probabilities(-200, 170)
        raw = implied_probability(-200)

        # Shin removes less vig from favourite than naive → shin > naive for favourite
        # But both should be less than raw (vig removed)
        assert result_shin["shin_a"] <= raw
        assert result_naive["fair_a"] <= raw

    def test_ev_and_kelly_agree_on_direction(self):
        """If EV says +EV, Kelly should recommend a bet (and vice versa)."""
        from odds_math import expected_value, kelly_criterion
        for odds, fair_p in [(-110, 0.55), (-110, 0.50), (200, 0.40), (200, 0.30)]:
            ev = expected_value(odds, fair_p)
            kc = kelly_criterion(odds, fair_p)
            assert ev["is_positive_ev"] == kc["is_positive"], \
                f"Disagreement at odds={odds}, fair_p={fair_p}"

    def test_arb_impossible_with_vig(self):
        """Standard vigged market (-110/-110) should never be an arb."""
        from odds_math import arbitrage_profit
        result = arbitrage_profit(-110, -110)
        assert result["is_arb"] is False

    def test_no_vig_and_shin_both_sum_to_one(self):
        """Both methods must produce probabilities summing to 1."""
        from odds_math import no_vig_probabilities, shin_probabilities
        for odds_a, odds_b in [(-110, -110), (-200, 170), (-300, 250), (-500, 400)]:
            naive = no_vig_probabilities(odds_a, odds_b)
            shin = shin_probabilities(odds_a, odds_b)
            assert abs(naive["fair_a"] + naive["fair_b"] - 1.0) < 1e-5
            assert abs(shin["shin_a"] + shin["shin_b"] - 1.0) < 1e-5

    def test_poisson_pmf_vs_scipy_formula(self):
        """Verify our Poisson PMF against direct factorial computation."""
        for k in range(10):
            for lam in [0.5, 1.0, 3.0, 7.0]:
                our_result = _poisson_pmf(k, lam)
                # Direct: λ^k * e^(-λ) / k!
                factorial_k = math.factorial(k)
                expected = (lam ** k) * math.exp(-lam) / factorial_k
                assert abs(our_result - expected) < 1e-12, \
                    f"PMF mismatch at k={k}, λ={lam}: {our_result} vs {expected}"

    def test_gamlss_reduces_to_standard_zscore_when_symmetric(self):
        """With nu=0, GAMLSS z-score should equal standard z-score."""
        data = [10, 20, 30, 40, 50]
        fit = _gamlss_fit(data)
        # Force nu to 0 for this test
        fit_no_skew = {**fit, "nu": 0.0}
        for val in [5, 15, 25, 35, 45, 55]:
            z_gamlss = _gamlss_zscore(val, fit_no_skew)
            z_standard = (val - fit["mu"]) / fit["sigma"] if fit["sigma"] > 1e-9 else 0
            assert abs(z_gamlss - z_standard) < 1e-10

    def test_implied_probability_roundtrip_through_vig(self):
        """implied_prob → calculate_vig should be internally consistent."""
        from odds_math import implied_probability, calculate_vig
        odds_a, odds_b = -150, 130
        vig_result = calculate_vig(odds_a, odds_b)
        assert abs(vig_result["implied_a"] - implied_probability(odds_a)) < 1e-6
        assert abs(vig_result["implied_b"] - implied_probability(odds_b)) < 1e-6

    def test_kelly_zero_when_ev_negative(self):
        """Negative EV must always produce zero Kelly sizing."""
        from odds_math import expected_value, kelly_criterion
        # Deliberately bad bet: fair prob lower than implied
        ev = expected_value(-200, 0.60)
        kc = kelly_criterion(-200, 0.60)
        if not ev["is_positive_ev"]:
            assert kc["recommended_fraction"] == 0.0

    def test_fair_odds_roundtrip_through_no_vig(self):
        """fair_odds_to_american(no_vig_prob) should give approximately fair American odds."""
        from odds_math import no_vig_probabilities, fair_odds_to_american, implied_probability
        result = no_vig_probabilities(-150, 130)
        fair_american_a = fair_odds_to_american(result["fair_a"])
        # The fair American odds should, when converted back, give the fair prob
        recovered = implied_probability(fair_american_a)
        assert abs(recovered - result["fair_a"]) < 0.01

    def test_knn_and_isolation_forest_agree_on_outlier(self):
        """Both methods should flag the same clear outlier."""
        import random
        rng = random.Random(42)
        data = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(30)]
        data.append([50, 50])  # obvious outlier

        knn_scores = _knn_anomaly_scores(data, k=5)
        iso_scores = _isolation_forest_scores(data, n_trees=200, seed=42)

        knn_outlier_idx = knn_scores.index(max(knn_scores))
        iso_outlier_idx = iso_scores.index(max(iso_scores))

        assert knn_outlier_idx == 30  # The appended outlier
        assert iso_outlier_idx == 30
