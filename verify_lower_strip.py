#!/usr/bin/env python3
"""Exact replay for the lower-strip argument and its E1 input.

The replay derives the asymptotic coefficients from the displayed Darboux
formulas, proves the uniform remainder budgets with rational interval and
Bernstein bounds, checks the two analytic column tails, and reconstructs every
cell in the finite residue.  It deliberately has no reduced or quick mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from functools import lru_cache
from fractions import Fraction as F
from pathlib import Path
from typing import Any, Iterable

import sympy as sp


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "lower_strip_certificate.json"

x, h = sp.symbols("x h")


class GateBook:
    """Collect exact checks and fail only after reporting every failed gate."""

    def __init__(self) -> None:
        self.gates: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def check(self, name: str, condition: Any, evidence: Any) -> None:
        passed = bool(condition)
        self.gates.append(
            {"name": name, "passed": passed, "evidence": json_value(evidence)}
        )
        if not passed:
            self.failures.append(name)


def as_fraction(value: Any) -> F:
    value = sp.cancel(value)
    if not value.is_Rational:
        raise TypeError(f"expected a rational value, got {value!r}")
    return F(int(value.p), int(value.q))


def fraction_text(value: F | sp.Rational | int) -> str:
    value = as_fraction(value) if isinstance(value, sp.Basic) else F(value)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def json_value(value: Any) -> Any:
    if isinstance(value, F):
        return fraction_text(value)
    if isinstance(value, sp.Rational):
        return fraction_text(value)
    if isinstance(value, sp.Basic):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def bernstein_coefficients(poly: Any, variable: sp.Symbol = x) -> list[F]:
    """Bernstein coefficients on x in [1/2, 1], computed exactly."""

    u = sp.symbols("u")
    shifted = sp.Poly(sp.expand(poly.subs(variable, (1 + u) / 2)), u, domain=sp.QQ)
    degree = shifted.degree()
    if degree < 0:
        return [F(0)]
    powers = [as_fraction(shifted.nth(j)) for j in range(degree + 1)]
    return [
        sum(
            (
                powers[j] * F(math.comb(k, j), math.comb(degree, j))
                for j in range(k + 1)
            ),
            F(0),
        )
        for k in range(degree + 1)
    ]


def interval(poly: Any) -> tuple[F, F]:
    coeffs = bernstein_coefficients(sp.expand(poly))
    return min(coeffs), max(coeffs)


def abs_bound(poly: Any) -> F:
    lower, upper = interval(poly)
    return max(abs(lower), abs(upper))


def coeff(expr: Any, variable: sp.Symbol, degree: int) -> Any:
    return sp.expand(expr).coeff(variable, degree)


def gamma_log_series(a: Any, b: Any, order: int) -> Any:
    """log(Gamma(n+a)/Gamma(n+b)) through n**(-order)."""

    return sp.expand(
        sum(
            (-1) ** (m + 1)
            * (sp.bernoulli(m + 1, a) - sp.bernoulli(m + 1, b))
            * h**m
            / (m * (m + 1))
            for m in range(1, order + 1)
        )
    )


def homogeneous(arguments: Iterable[Any], degree: int) -> Any:
    z = sp.symbols("z")
    generating = sp.prod(1 / (1 - a * z) for a in arguments)
    return sp.expand(sp.series(generating, z, 0, degree + 1).removeO()).coeff(z, degree)


def polynomial_tail_bound(poly: Any, start: int, scale: F) -> F:
    expanded = sp.Poly(sp.expand(poly), h)
    if expanded.degree() < start:
        return F(0)
    return sum(
        (
            abs_bound(expanded.nth(j)) * scale ** (j - start)
            for j in range(start, expanded.degree() + 1)
        ),
        F(0),
    )


def log_rational_bounds(q: F, terms: int) -> tuple[F, F]:
    """Exact atanh-series bounds for log(q), q > 1."""

    if q <= 1 or terms < 1:
        raise ValueError("log_rational_bounds requires q > 1 and terms >= 1")
    r = (q - 1) / (q + 1)
    lower = 2 * sum((r ** (2 * k + 1) / (2 * k + 1) for k in range(terms)), F(0))
    first_omitted = 2 * r ** (2 * terms + 1) / (2 * terms + 1)
    upper = lower + first_omitted / (1 - r * r)
    return lower, upper


def derive_e1(book: GateBook) -> dict[str, Any]:
    """Derive and certify the displayed E1 expansion for n >= 105."""

    H = F(1, 105)

    D = (
        1
        + x**2 * h / (2 * (1 + (1 - x) * h))
        + x**2 * (1 - x) ** 2 * h**2 / (8 * (1 + (1 - x) * h) * (1 + (2 - x) * h))
        + x**2
        * (1 - x) ** 2
        * (2 - x) ** 2
        * h**3
        / (48 * (1 + (1 - x) * h) * (1 + (2 - x) * h) * (1 + (3 - x) * h))
    )
    N = (
        1
        + (x + 1) ** 2 * h / (2 * (1 + (x + 2) * h))
        + (x + 1) ** 2
        * (x + 2) ** 2
        * h**2
        / (8 * (1 + (x + 2) * h) * (1 + (x + 3) * h))
    )

    log_D = sp.series(sp.log(D), h, 0, 5).removeO()
    smooth = sp.expand(gamma_log_series(1, 1 - x, 4) + log_D)
    a1 = sp.factor(coeff(smooth, h, 1))
    a2 = sp.factor(coeff(smooth, h, 2))
    a3 = sp.factor(coeff(smooth, h, 3))
    s4 = sp.factor(coeff(smooth, h, 4))

    expected_a1 = x / 2
    expected_a2 = x * (x - 2) * (2 * x + 1) / 24
    expected_a3 = -(x**2) * (2 * x - 3) / 24
    book.check("E1 a1 derived from (7)", sp.simplify(a1 - expected_a1) == 0, a1)
    book.check("E1 a2 derived from (7)", sp.simplify(a2 - expected_a2) == 0, a2)
    book.check("E1 a3 derived from (7)", sp.simplify(a3 - expected_a3) == 0, a3)

    log_Q = sp.expand(
        gamma_log_series(1 - x, x + 2, 2)
        + sp.series(sp.log(N), h, 0, 3).removeO()
        - sp.series(sp.log(D), h, 0, 3).removeO()
    )
    q_log_1 = sp.factor(coeff(log_Q, h, 1))
    q_log_2 = sp.factor(coeff(log_Q, h, 2))
    Q = sp.series(sp.exp(log_Q), h, 0, 3).removeO()
    q1 = sp.factor(coeff(Q, h, 1))
    b2 = sp.factor(coeff(Q, h, 2))
    expected_q1 = -(2 * x + 1) / 2
    expected_b2 = -x * (x - 2) * (2 * x + 1) / 12
    book.check(
        "E1 alternating linear term derived from (8)",
        sp.simplify(q1 - expected_q1) == 0,
        q1,
    )
    book.check("E1 b2 derived from (8)", sp.simplify(b2 - expected_b2) == 0, b2)

    s4_range = interval(s4)
    book.check(
        "smooth fourth coefficient is uniformly small",
        abs_bound(s4) <= F(1, 60),
        s4_range,
    )

    # Binet's integral remainder makes each omitted gamma-ratio coefficient no
    # larger than the following exact majorant at h <= 1/105.
    # These are the first omitted terms of the five Binet kernels.  The first
    # four are alternating for 0 <= u*h <= H; epsilon contributes h^6/252.
    gamma_smooth_parts = {
        "log": F(1, 4 + 1),
        "one_over_2y": F(1, 2) * math.comb(3 + 1 - 1, 3),
        "one_over_12y2": F(1, 12) * math.comb(2 + 3 - 1, 3),
        "one_over_120y4": F(1, 120) * math.comb(4 + 1 - 1, 1),
        "Binet_epsilon": H / 252,
    }
    gamma_smooth_h5 = sum(gamma_smooth_parts.values(), F(0))
    smooth_binet_ratio = F(5, 2) * H
    book.check(
        "smooth Binet coefficient tails decrease",
        smooth_binet_ratio < 1,
        {"worst_successive_ratio": smooth_binet_ratio},
    )
    book.check(
        "smooth Binet fifth-order tail",
        gamma_smooth_h5 < F(107, 100),
        gamma_smooth_parts,
    )

    d_tail_ratios = (F(1, 2) * H, F(2) * H, F(9, 2) * H)
    d_first_omitted = (
        abs_bound(x**2 * homogeneous([1 - x], 4) / 2),
        abs_bound(x**2 * (1 - x) ** 2 * homogeneous([1 - x, 2 - x], 3) / 8),
        abs_bound(
            x**2
            * (1 - x) ** 2
            * (2 - x) ** 2
            * homogeneous([1 - x, 2 - x, 3 - x], 2)
            / 48
        ),
    )
    book.check(
        "D geometric tails contract at n=105",
        all(r < 1 for r in d_tail_ratios),
        {"ratio_bounds": d_tail_ratios},
    )
    raw_D_h5 = sum(
        (first / (1 - ratio) for first, ratio in zip(d_first_omitted, d_tail_ratios)),
        F(0),
    )
    book.check(
        "full D rational tail after h^4",
        raw_D_h5 < F(1, 8),
        {"first_omitted": d_first_omitted, "contracted": raw_D_h5},
    )

    p4 = sp.series(D - 1, h, 0, 5).removeO()
    p_over_h = sum(
        (abs_bound(coeff(p4, h, j)) * H ** (j - 1) for j in range(1, 5)),
        F(0),
    )
    p_lower_over_h = interval(coeff(p4, h, 1))[0] - sum(
        (abs_bound(coeff(p4, h, j)) * H ** (j - 1) for j in range(2, 5)),
        F(0),
    )
    book.check(
        "D stays above one",
        p_lower_over_h > 0,
        {"lower_bound_for_(D-1)/h": p_lower_over_h},
    )
    book.check(
        "log(D) power series is inside its disk",
        p_over_h * H < 1,
        {"upper_bound_for_D-1": p_over_h * H},
    )

    log_D_algebra = sp.expand(p4 - p4**2 / 2 + p4**3 / 3 - p4**4 / 4)
    log_D_h5 = (
        raw_D_h5
        + polynomial_tail_bound(log_D_algebra, 5, H)
        + p_over_h**5 / (5 * (1 - p_over_h * H))
    )
    smooth_h5 = gamma_smooth_h5 + log_D_h5
    smooth_n4 = abs_bound(s4) + smooth_h5 * H
    book.check(
        "smooth E1 envelope", smooth_n4 < F(1, 20), {"coefficient_of_n^-4": smooth_n4}
    )

    # The alternating quotient Q=N/D times its gamma ratio.  All bounds below
    # are generated from the exact rational functions above.
    gamma_alt_parts = {
        "log": F(3**3, 3),
        "one_over_2y": F(3**2, 2),
        "one_over_12y2": F(3, 6),
        "one_over_120y4": H / 120,
        "Binet_epsilon": H**3 / 252,
    }
    gamma_alt_h3 = 3 * sum(gamma_alt_parts.values(), F(0))
    alternating_binet_ratio = F(9, 2) * H
    book.check(
        "alternating Binet coefficient tails decrease",
        alternating_binet_ratio < 1,
        {"worst_successive_ratio": alternating_binet_ratio},
    )
    book.check("alternating gamma-ratio tail", gamma_alt_h3 < 43, gamma_alt_h3)

    A = (x + 1) ** 2 / 2
    B = (x + 1) ** 2 * (x + 2) ** 2 / 8
    v1 = sp.factor(A)
    v2 = sp.factor(-A * (x + 2) + B)
    n_tail_ratios = (F(3) * H, F(7) * H)
    n_first_omitted = (abs_bound(A * (x + 2) ** 2), abs_bound(B * (2 * x + 5)))
    book.check(
        "N geometric tails contract at n=105",
        all(r < 1 for r in n_tail_ratios),
        {"ratio_bounds": n_tail_ratios},
    )
    raw_N_h3 = sum(
        (first / (1 - ratio) for first, ratio in zip(n_first_omitted, n_tail_ratios)),
        F(0),
    )
    v1_abs = abs_bound(v1)
    v2_abs = abs_bound(v2)
    v_over_h = v1_abs + v2_abs * H
    v_floor = 1 + interval(v1)[0] * H - v2_abs * H**2
    book.check("N quadratic truncation stays above one", v_floor > 1, v_floor)
    book.check(
        "N logarithm tail denominator is positive",
        v_over_h * H < 1,
        {"upper_bound_for_N_truncation": v_over_h * H},
    )
    log_N_h3 = (
        raw_N_h3
        + abs_bound(v1 * v2)
        + abs_bound(v2**2) * H / 2
        + v_over_h**3 / (3 * (1 - v_over_h * H))
    )
    book.check("N logarithm tail", log_N_h3 < 80, log_N_h3)

    d1 = sp.factor(coeff(sp.series(D - 1, h, 0, 3).removeO(), h, 1))
    d2 = sp.factor(coeff(sp.series(D - 1, h, 0, 3).removeO(), h, 2))
    d2_tail_ratios = (F(1, 2) * H, F(2) * H, F(9, 2) * H)
    d2_first_omitted = (
        abs_bound(x**2 * (1 - x) ** 2 / 2),
        abs_bound(x**2 * (1 - x) ** 2 * (3 - 2 * x) / 8),
        abs_bound(x**2 * (1 - x) ** 2 * (2 - x) ** 2 / 48),
    )
    book.check(
        "quadratic D tails contract at n=105",
        all(r < 1 for r in d2_tail_ratios),
        {"ratio_bounds": d2_tail_ratios},
    )
    raw_D_h3 = sum(
        (first / (1 - ratio) for first, ratio in zip(d2_first_omitted, d2_tail_ratios)),
        F(0),
    )
    d_over_h = abs_bound(d1) + abs_bound(d2) * H
    d_floor = 1 + interval(d1)[0] * H - abs_bound(d2) * H**2
    book.check("D quadratic truncation stays above one", d_floor > 1, d_floor)
    book.check(
        "D logarithm tail denominator is positive",
        d_over_h * H < 1,
        {"upper_bound_for_D_truncation": d_over_h * H},
    )
    log_D_h3 = (
        raw_D_h3
        + abs_bound(d1 * d2)
        + abs_bound(d2**2) * H / 2
        + d_over_h**3 / (3 * (1 - d_over_h * H))
    )
    book.check("D logarithm tail for Q", log_D_h3 < 1, log_D_h3)

    log_Q_h3 = gamma_alt_h3 + log_N_h3 + log_D_h3
    book.check("log(Q) tail", log_Q_h3 < 124, log_Q_h3)
    q1_abs = abs_bound(q_log_1)
    q2_abs = abs_bound(q_log_2)
    q_over_h = q1_abs + q2_abs * H
    z_bound = q_over_h * H + log_Q_h3 * H**3
    book.check(
        "Q exponential outer denominator is positive",
        z_bound < 1,
        {"upper_bound_for_log_Q": z_bound},
    )
    book.check(
        "Q exponential inner denominator is positive",
        q_over_h * H < 1,
        {"upper_bound_for_quadratic_log_Q": q_over_h * H},
    )
    q_h3 = (
        log_Q_h3 / (1 - z_bound)
        + abs_bound(q_log_1 * q_log_2)
        + abs_bound(q_log_2**2) * H / 2
        + q_over_h**3 / (6 * (1 - q_over_h * H))
    )
    book.check("Q exponential tail", q_h3 < 200, q_h3)

    omitted_alt_linear = F(1, 16) * (abs_bound(b2) + q_h3 * H)
    Q_upper = 1 + abs_bound(q1) * H + abs_bound(b2) * H**2 + q_h3 * H**3
    alternating_log_argument = F(1, 16) * H**2 * Q_upper
    book.check(
        "alternating quadratic denominator is positive",
        alternating_log_argument < 1,
        {"upper_bound_for_kappa_n^(-2x-1)_times_Q": alternating_log_argument},
    )
    alternating_square = (
        F(1, 16) ** 2 * Q_upper**2 / (2 * (1 - F(1, 16) * H**2 * Q_upper))
    )
    book.check(
        "alternating linear envelope", omitted_alt_linear < F(3, 20), omitted_alt_linear
    )
    book.check(
        "alternating quadratic envelope",
        alternating_square < F(1, 400),
        alternating_square,
    )

    # Exact elementary fences used for kappa and for the coefficient-tail
    # constants.  The log bounds are terminating rational certificates.
    log2_lower, _ = log_rational_bounds(F(2), 12)
    _, log7_upper = log_rational_bounds(F(7), 40)
    harmonic_6 = sum((F(1, j) for j in range(1, 7)), F(0))
    euler_gamma_lower = harmonic_6 - log7_upper
    book.check("rational certificate log(2)>2/3", log2_lower > F(2, 3), log2_lower)
    book.check(
        "Euler integral certificate gamma>1/2",
        euler_gamma_lower > F(1, 2),
        euler_gamma_lower,
    )
    kappa_decay_margin = 2 * (log2_lower - (1 - euler_gamma_lower))
    # For phi=log(x/(8*kappa)), cot(pi*x)<=0 and psi(x+1)<=psi(2)
    # give phi' >= 1 + 2log(2) - 2(1-gamma).
    phi_growth_margin = 2 * log2_lower - 1 + 2 * euler_gamma_lower
    special_edge_decay_margin = 4 * log2_lower / 3
    book.check(
        "kappa decreases on the strip", kappa_decay_margin > 0, kappa_decay_margin
    )
    book.check(
        "2*kappa <= x/4 endpoint reduction", phi_growth_margin > 0, phi_growth_margin
    )
    book.check(
        "m=1 alternating envelope decreases",
        special_edge_decay_margin > 0,
        special_edge_decay_margin,
    )
    kappa_half_exact = sp.simplify(
        2 ** (-2 * sp.Rational(1, 2) - 1) * sp.gamma(sp.Rational(3, 2)) ** 2 / sp.pi
    )
    book.check(
        "kappa endpoint is 1/16",
        kappa_half_exact == sp.Rational(1, 16),
        kappa_half_exact,
    )
    psi_derivative_margin = log2_lower * F(1, 2)
    book.check(
        "psi(m,nu) decreases for m>=3",
        psi_derivative_margin > 0,
        {"negative_derivative_margin": psi_derivative_margin},
    )
    book.check(
        "D_m(nu) decreases for m>=2",
        2 * log2_lower > 1,
        {"2*log(2)_lower": 2 * log2_lower, "integrand": "t^(-nu-1)*(1-nu*log(t))"},
    )

    k_s = x * (1 - x) * (2 - x) * (3 - x)
    k_a = (x + 1) * (x + 2) * (x + 3)
    k_s_bound = abs_bound(k_s)
    k_a_bound = abs_bound(k_a)
    pi_quarter_lower = sum((F((-1) ** j, 2 * j + 1) for j in range(8)), F(0))
    book.check(
        "Gregory integral certificate pi>3",
        4 * pi_quarter_lower > 3,
        4 * pi_quarter_lower,
    )
    book.check("sqrt(2)>7/5", F(2) > F(7, 5) ** 2, F(7, 5))
    book.check("K_S Bernstein envelope", k_s_bound <= F(15, 16), k_s_bound)
    book.check("K_A Bernstein envelope", k_a_bound <= 24, k_a_bound)

    # U_S has an exact four-factor quotient, each factor at least n.  For U_A,
    # the beta integral with y=n+1-x and a=2x+4 gives
    # Gamma(y)/Gamma(y+a) <= (y-1)^(-a); the endpoint arithmetic below turns
    # that into 2*n^(-a), then spends one more 1/n at n=105.
    gamma_offsets = tuple(interval(j - x)[0] for j in range(1, 5))
    book.check(
        "U_S quotient is a direct four-factor product",
        gamma_offsets == (F(0), F(1), F(2), F(3)),
        {"factor_offsets_from_n": gamma_offsets},
    )
    book.check(
        "Gamma(x+1) endpoint chord",
        sp.gamma(1) == 1 and sp.gamma(2) == 1,
        "log-Gamma convexity gives Gamma(x+1)<=1",
    )
    beta_n = sp.symbols("beta_n", integer=True, positive=True)
    beta_y = beta_n + 1 - x
    beta_a = 2 * x + 4
    book.check(
        "beta-integral parameter identity",
        sp.simplify(beta_y + beta_a - (beta_n + x + 5)) == 0,
        {"y": beta_y, "a": beta_a, "y_plus_a": beta_y + beta_a},
    )
    book.check(
        "beta-integral hypotheses on the E1 range",
        105 - interval(x)[1] == 104 and interval(beta_a) == (F(5), F(6)) and F(104) > 0,
        {
            "y_minus_one_lower": 104,
            "a_range": interval(beta_a),
            "lemma": "Gamma(y)/Gamma(y+a) <= (y-1)^(-a) for y>1 and a>0",
        },
    )
    beta_ratio_factor = F(105, 104) ** 6
    book.check(
        "beta-integral exponent range",
        interval(2 * x + 4) == (F(5), F(6)),
        interval(2 * x + 4),
    )
    book.check(
        "beta-integral quotient factor",
        beta_ratio_factor < 2,
        {"(n/(n-x))^a_upper": beta_ratio_factor},
    )
    u_s_coefficient = k_s_bound**2 / (24 * F(7, 5))
    u_a_coefficient = F(1, 3) * k_a_bound**2 * 2 * H / (6 * F(7, 5))
    book.check(
        "symmetric coefficient remainder", u_s_coefficient < F(1, 36), u_s_coefficient
    )
    book.check(
        "alternating coefficient remainder", u_a_coefficient < F(4, 9), u_a_coefficient
    )

    denominator_floor = 1 - F(1, 16) * H**2 * Q_upper - (F(1, 36) + F(4, 9)) * H**4
    book.check(
        "logarithm denominator floor", denominator_floor > F(99, 100), denominator_floor
    )
    rho_symmetric = F(1, 36) / denominator_floor
    rho_alternating = F(4, 9) / denominator_floor
    book.check("rho symmetric allocation", rho_symmetric < F(1, 30), rho_symmetric)
    book.check(
        "rho alternating allocation", rho_alternating < F(9, 20), rho_alternating
    )

    final_symmetric = smooth_n4 + rho_symmetric
    final_alternating = omitted_alt_linear + rho_alternating
    final_quadratic = alternating_square
    book.check("E1 n^-4 budget", final_symmetric < 1, final_symmetric)
    book.check("E1 n^(-2x-3) budget", final_alternating < 1, final_alternating)
    book.check("E1 n^(-4x-2) budget", final_quadratic < 1, final_quadratic)

    return {
        "range": {"x": "1/2 <= x <= 1", "n": "n >= 105"},
        "derived_coefficients": {
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "s4": s4,
            "q1": q1,
            "b2": b2,
        },
        "envelopes": {
            "smooth_n^-4": final_symmetric,
            "alternating_n^(-2x-3)": final_alternating,
            "quadratic_n^(-4x-2)": final_quadratic,
        },
        "analytic_inputs": [
            "Binet integral remainder for log-Gamma ratios",
            "beta-integral bound for the alternating gamma quotient",
            "Euler integral bound gamma > H_6 - log(7)",
            "digamma recurrence and monotonicity",
            "log-convexity of Gamma on [1,2]",
            "alternating geometric and Gregory integral remainders",
        ],
    }


def D_node(m: int, k: int) -> F:
    return F(1, m**k) - F(1, (m + 1) ** k)


def nodes(p: int, q: int) -> tuple[int, int, int, int]:
    return (p + 1) * (q + 1), p * q, p * (q + 1), (p + 1) * q


def odd_cell_node(p: int, q: int) -> tuple[int, bool]:
    a, b, c, d = nodes(p, q)
    odds = [n for n in (a, b, c, d) if n % 2 == 1]
    if len(odds) != 1:
        raise ArithmeticError(f"cell ({p},{q}) has {len(odds)} odd nodes")
    odd = odds[0]
    return odd, odd in (c, d)


def beta(m: int) -> F:
    return F(3, 4) if m == 1 else D_node(m, 2)


A2_COEFFICIENT = F(25, 192)


@lru_cache(maxsize=None)
def a_polynomial(n: int) -> tuple[int, ...]:
    if n == 0:
        return (1,)
    if n == 1:
        return (0, 1)
    previous = a_polynomial(n - 1)
    before_previous = a_polynomial(n - 2)
    out = [0] * (n + 1)
    for degree, value in enumerate(previous):
        out[degree + 1] += value
    factor = (n - 1) ** 2
    for degree, value in enumerate(before_previous):
        out[degree] += factor * value
    return tuple(out)


def polynomial_product(
    left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        if a:
            for j, b in enumerate(right):
                if b:
                    out[i + j] += a * b
    return tuple(out)


def taylor_shift(coefficients: tuple[int, ...], shift: int) -> tuple[int, ...]:
    """Coefficients of P(t), re-expanded in v=t-shift."""

    work = list(coefficients)
    out: list[int] = []
    for _ in range(len(work)):
        accumulator = 0
        quotient = [0] * (len(work) - 1) if len(work) > 1 else []
        for j in range(len(work) - 1, 0, -1):
            accumulator += work[j]
            quotient[j - 1] = accumulator
            accumulator *= shift
        out.append(work[0] + accumulator)
        work = quotient
        if not work:
            break
    return tuple(out)


def cell_numerator(p: int, q: int) -> tuple[int, ...]:
    a, b, c, d = nodes(p, q)
    fa, fb, fc, fd = (math.factorial(n) for n in (a, b, c, d))
    left = polynomial_product(a_polynomial(a), a_polynomial(b))
    right = polynomial_product(a_polynomial(c), a_polynomial(d))
    common = math.gcd(fc * fd, fa * fb)
    left_weight = (fc * fd) // common
    right_weight = (fa * fb) // common
    degree = max(len(left), len(right))
    numerator = [0] * degree
    for j, value in enumerate(left):
        numerator[j] += left_weight * value
    for j, value in enumerate(right):
        numerator[j] -= right_weight * value
    while len(numerator) > 1 and numerator[-1] == 0:
        numerator.pop()
    return tuple(numerator)


def positive_shift_certificate(coefficients: tuple[int, ...]) -> bool:
    return (
        bool(coefficients)
        and coefficients[0] > 0
        and all(value >= 0 for value in coefficients)
    )


def vector_digest(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def symbolic_column_gate(parity: str) -> tuple[Any, Any]:
    p = sp.symbols("p", positive=True, integer=True)

    def d(m: Any, k: int) -> Any:
        m, k = sp.sympify(m), sp.sympify(k)
        return m ** (-k) - (m + 1) ** (-k)

    beta_p = d(p, 2)
    beta_1 = sp.Rational(3, 4)
    s1 = d(p, 1) * d(1, 1)
    s2 = d(p, 2) * d(1, 2)
    G = (
        sp.Rational(1, 2) * s1
        - sp.Rational(25, 192) * s2
        - sp.Rational(1, 8) * beta_p * beta_1
    )
    if parity == "even":
        odd = p + 1
        G -= sp.Rational(1, 4) * (1 / odd**2 - 1 / odd**3)
    a, b, c, d_node = 2 * (p + 1), p, 2 * p, p + 1
    rem = 3 * (1 / a**4 + 1 / b**4 + 1 / c**4 + 1 / d_node**4)
    numerator, denominator = sp.fraction(sp.cancel(G - 2 * rem))
    if as_fraction(denominator.subs(p, 2)) < 0:
        numerator, denominator = -numerator, -denominator
    return sp.factor(numerator), sp.factor(denominator)


def residue_cells() -> tuple[list[tuple[int, int]], dict[str, int]]:
    residue: list[tuple[int, int]] = []
    split = {"column_q_le_104": 0, "column_q_106_112": 0, "interior_pq_le_104": 0}
    for p in range(1, 105):
        for q in range(p, 113):
            analytic = (p >= 2 and p * q >= 105) or (
                p == 1 and q >= 105 and (q % 2 == 1 or q >= 113)
            )
            if analytic:
                continue
            residue.append((p, q))
            if p == 1 and q <= 104:
                split["column_q_le_104"] += 1
            elif p == 1:
                split["column_q_106_112"] += 1
            else:
                split["interior_pq_le_104"] += 1
    return residue, split


def derive_lower_strip(
    book: GateBook, e1: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    a2 = sp.sympify(e1["derived_coefficients"]["a2"])
    b2 = sp.sympify(e1["derived_coefficients"]["b2"])
    a2_defect = sp.factor(sp.Rational(25, 8) + 24 * a2 / x)
    b2_scaled = sp.factor(12 * b2 / x)
    book.check(
        "lower-strip a2 envelope derived from E1",
        sp.simplify(a2_defect - 2 * (x - sp.Rational(3, 4)) ** 2) == 0,
        a2_defect,
    )
    book.check(
        "lower-strip b2 polynomial derived from E1",
        sp.simplify(b2_scaled - (2 + 3 * x - 2 * x**2)) == 0,
        b2_scaled,
    )
    a3 = sp.sympify(e1["derived_coefficients"]["a3"])
    book.check(
        "lower-strip a3 term is strictly positive",
        interval(a3)[0] > 0,
        {"Bernstein_range": interval(a3)},
    )
    derived_beta_one = F(8) * F(1, 16) / F(1, 2) * (1 - F(1, 4))
    book.check(
        "special beta(1) is derived at the endpoint",
        derived_beta_one == beta(1) == F(3, 4),
        {"8*kappa(1/2)/(1/2)": F(1), "D_1(2)": F(3, 4)},
    )

    p = sp.symbols("p", integer=True, positive=True)
    q = sp.symbols("q", integer=True, positive=True)
    w, v = sp.symbols("w v")
    a, b, c, d = (p + 1) * (q + 1), p * q, p * (q + 1), (p + 1) * q
    book.check(
        "adjacent-cell product identity",
        sp.expand(a * b - c * d) == 0,
        {"nodes": [a, b, c, d]},
    )
    direct_identities = []
    for power in (1, 2, 3, 4):
        direct = a ** (-power) + b ** (-power) - c ** (-power) - d ** (-power)
        product = (p ** (-power) - (p + 1) ** (-power)) * (
            q ** (-power) - (q + 1) ** (-power)
        )
        direct_identities.append(sp.cancel(direct - product) == 0)
    book.check(
        "four-node differences factor as D_p D_q",
        all(direct_identities),
        {"powers": [1, 2, 3, 4]},
    )
    parity_cases = []
    for p_value, q_value in ((2, 2), (2, 3), (3, 2), (3, 3)):
        odd, adverse = odd_cell_node(p_value, q_value)
        parity_cases.append(
            {"p": p_value % 2, "q": q_value % 2, "odd": odd % 2, "adverse": adverse}
        )
    book.check(
        "odd-node parity classification",
        [case["adverse"] for case in parity_cases] == [False, True, True, False],
        parity_cases,
    )

    column_results: dict[str, Any] = {}
    shifted_for_mutation: tuple[int, ...] | None = None
    expected_columns = {
        "even": 30 * p**5 - 3317 * p**4 - 6660 * p**3 - 9841 * p**2 - 6528 * p - 1632,
        "odd": 64 * p**6
        + 94 * p**5
        - 3317 * p**4
        - 6660 * p**3
        - 9841 * p**2
        - 6528 * p
        - 1632,
    }
    for parity, threshold, preceding in (("even", 113, 112), ("odd", 8, 7)):
        numerator, denominator = symbolic_column_gate(parity)
        denominator_poly = sp.Poly(sp.expand(denominator), p, domain=sp.QQ)
        denominator_coefficients = [
            as_fraction(c) for c in denominator_poly.all_coeffs()
        ]
        denominator_positive = all(c >= 0 for c in denominator_coefficients) and any(
            c > 0 for c in denominator_coefficients
        )
        shifted = sp.Poly(sp.expand(numerator.subs(p, w + threshold)), w, domain=sp.QQ)
        shifted_coefficients_f = tuple(
            as_fraction(shifted.nth(j)) for j in range(shifted.degree() + 1)
        )
        common_denominator = math.lcm(*(v.denominator for v in shifted_coefficients_f))
        shifted_coefficients = tuple(
            int(v * common_denominator) for v in shifted_coefficients_f
        )
        previous_value = as_fraction(numerator.subs(p, preceding))
        book.check(
            f"{parity} column polynomial derived",
            sp.expand(numerator - expected_columns[parity]) == 0,
            numerator,
        )
        book.check(
            f"{parity} column clearing denominator derived",
            sp.expand(denominator - 256 * p**4 * (p + 1) ** 4) == 0,
            denominator,
        )
        book.check(
            f"{parity} column denominator positive", denominator_positive, denominator
        )
        book.check(
            f"{parity} column Taylor certificate at {threshold}",
            positive_shift_certificate(shifted_coefficients),
            {"degree": shifted.degree(), "constant": shifted_coefficients[0]},
        )
        book.check(
            f"{parity} column threshold is sharp",
            previous_value < 0,
            {"at": preceding, "numerator": previous_value},
        )
        column_results[parity] = {
            "threshold": threshold,
            "numerator": numerator,
            "denominator": denominator,
            "shifted_coefficients": shifted_coefficients,
        }
        if parity == "even":
            shifted_for_mutation = shifted_coefficients

    ratio_2 = F(5, 6)
    ratio_3 = F(7, 12)
    mixed_coefficient = A2_COEFFICIENT + F(1, 8)
    m = sp.symbols("m", integer=True, positive=True)
    ratio_m = (2 * m + 1) / (m * (m + 1))
    ratio_difference = sp.factor(ratio_m - ratio_m.subs(m, m + 1))
    ratio_num, ratio_den = sp.fraction(ratio_difference)
    ratio_num_shift = sp.Poly(sp.expand(ratio_num.subs(m, w + 1)), w, domain=sp.QQ)
    ratio_den_shift = sp.Poly(sp.expand(ratio_den.subs(m, w + 1)), w, domain=sp.QQ)
    ratio_decreases = (
        all(as_fraction(value) >= 0 for value in ratio_num_shift.all_coeffs())
        and ratio_num_shift.eval(0) > 0
        and all(as_fraction(value) >= 0 for value in ratio_den_shift.all_coeffs())
        and ratio_den_shift.eval(0) > 0
    )
    book.check("r_m decreases exactly", ratio_decreases, ratio_difference)
    book.check(
        "r_2 and r_3 endpoints",
        ratio_m.subs(m, 2) == sp.Rational(5, 6)
        and ratio_m.subs(m, 3) == sp.Rational(7, 12),
        {"r2": ratio_m.subs(m, 2), "r3": ratio_m.subs(m, 3)},
    )
    d1_symbolic = 1 / m - 1 / (m + 1)
    d2_symbolic = 1 / m**2 - 1 / (m + 1) ** 2
    d3_symbolic = 1 / m**3 - 1 / (m + 1) ** 3
    book.check(
        "D_m(2)=r_m D_m(1)",
        sp.cancel(d2_symbolic - ratio_m * d1_symbolic) == 0,
        ratio_m,
    )
    d3_expected = (3 * m**2 + 3 * m + 1) / (m**3 * (m + 1) ** 3)
    d3_numerator_shift = sp.Poly(
        sp.expand((3 * m**2 + 3 * m + 1).subs(m, w + 1)), w, domain=sp.QQ
    )
    book.check(
        "discarded a3 term has S3>0",
        sp.cancel(d3_symbolic - d3_expected) == 0
        and all(value > 0 for value in d3_numerator_shift.all_coeffs()),
        {"D_m(3)": d3_expected, "shifted_numerator": d3_numerator_shift.as_expr()},
    )
    adverse_difference = sp.factor(sp.Rational(4, 3) - p * (q + 1) / ((p + 1) * q))
    adverse_numerator, adverse_denominator = sp.fraction(adverse_difference)
    adverse_shift = sp.Poly(
        sp.expand(adverse_numerator.subs({p: w + 2, q: v + 3})),
        w,
        v,
        domain=sp.QQ,
    )
    book.check(
        "adverse psi ratio envelope",
        sp.expand(adverse_numerator - (p * (q - 3) + 4 * q)) == 0
        and sp.expand(adverse_denominator - 3 * q * (p + 1)) == 0
        and all(as_fraction(value) >= 0 for value in adverse_shift.coeffs())
        and adverse_shift.eval({w: 0, v: 0}) > 0,
        {
            "difference": adverse_difference,
            "shifted_numerator": adverse_shift.as_expr(),
        },
    )
    ratio_floor = sp.factor(p / (p + 1) - sp.Rational(2, 3))
    floor_numerator, floor_denominator = sp.fraction(ratio_floor)
    floor_shift = sp.Poly(sp.expand(floor_numerator.subs(p, w + 2)), w, domain=sp.QQ)
    book.check(
        "interior S1 product envelope",
        sp.expand(floor_numerator - (p - 2)) == 0
        and sp.expand(floor_denominator - 3 * (p + 1)) == 0
        and all(as_fraction(value) >= 0 for value in floor_shift.all_coeffs())
        and F(2, 3) ** 2 == F(4, 9),
        {"single_factor_difference": ratio_floor, "product_floor": F(4, 9)},
    )
    node_differences = (
        sp.expand(a - b),
        sp.expand(c - b),
        sp.expand(d - b),
    )
    shifted_node_differences = tuple(
        sp.Poly(expression.subs({p: w + 2, q: v + 2}), w, v, domain=sp.QQ)
        for expression in node_differences
    )
    book.check(
        "all four interior nodes are at least pq",
        node_differences == (p + q + 1, p, q)
        and all(
            all(as_fraction(value) >= 0 for value in polynomial.coeffs())
            and polynomial.eval({w: 0, v: 0}) > 0
            for polynomial in shifted_node_differences
        ),
        {
            "a-b": node_differences[0],
            "c-b": node_differences[1],
            "d-b": node_differences[2],
        },
    )
    c_adverse = F(1, 2) - F(1, 3) - mixed_coefficient * ratio_2 * ratio_3
    c_favorable = F(1, 2) - mixed_coefficient * ratio_2**2
    two_remainder_coefficient = F(2) * F(3) * F(4)
    s1_product_floor = F(4, 9)
    threshold_numerator = two_remainder_coefficient / s1_product_floor
    threshold_square = threshold_numerator / c_adverse
    book.check(
        "interior threshold constant is derived",
        two_remainder_coefficient == 24 and threshold_numerator == 54,
        {
            "two_times_three_E1_terms_times_four_nodes": two_remainder_coefficient,
            "S1_product_floor": s1_product_floor,
            "threshold_numerator": threshold_numerator,
        },
    )
    book.check("interior adverse constant", c_adverse == F(589, 13824), c_adverse)
    book.check("interior favorable constant", c_favorable == F(2231, 6912), c_favorable)
    book.check(
        "interior threshold pq>=36",
        F(35) ** 2 < threshold_square < F(36) ** 2,
        threshold_square,
    )
    book.check(
        "E1 begins beyond the interior threshold",
        105 >= 36,
        {"E1_pq": 105, "analytic_threshold": 36},
    )

    residue, split = residue_cells()
    book.check(
        "finite residue has exactly 260 canonical cells",
        len(residue) == 260,
        {"count": len(residue), "split": split},
    )
    book.check(
        "finite residue split is complete",
        split
        == {"column_q_le_104": 104, "column_q_106_112": 4, "interior_pq_le_104": 152},
        split,
    )

    aggregate = hashlib.sha256()
    degrees: list[int] = []
    first_shift: tuple[int, ...] | None = None
    failed_cells: list[tuple[int, int]] = []
    for p_value, q_value in residue:
        numerator = cell_numerator(p_value, q_value)
        shifted = taylor_shift(numerator, 2)
        if first_shift is None:
            first_shift = shifted
        if not positive_shift_certificate(shifted):
            failed_cells.append((p_value, q_value))
        degrees.append(len(numerator) - 1)
        aggregate.update(f"{p_value},{q_value},{len(numerator) - 1},".encode("ascii"))
        aggregate.update(vector_digest(shifted).encode("ascii"))
        aggregate.update(b"\n")
    book.check(
        "all 260 residue cells are coefficientwise positive after t=2",
        not failed_cells,
        {"failures": failed_cells, "digest": aggregate.hexdigest()},
    )

    # Six independent mutations must be rejected by the same local predicates.
    coefficient_mutation = sp.factor(a2 + x / 24)
    coefficient_rejected = sp.simplify(coefficient_mutation - expected_a2()) != 0

    edge_w = sp.symbols("edge_w", positive=True)
    edge_p = 1 / edge_w
    half = sp.Rational(1, 2)
    nu = 2 * half + 1
    a2_half = a2.subs(x, half)
    a3_half = sp.sympify(e1["derived_coefficients"]["a3"]).subs(x, half)
    kappa_half = sp.Rational(1, 16)

    def edge_series(include_linear_correction: bool) -> Any:
        a, b, c, d = 2 * edge_p + 2, edge_p, 2 * edge_p, edge_p + 1
        expression = 0
        for node, sign, parity in ((a, 1, 1), (b, 1, 1), (c, -1, 1), (d, -1, -1)):
            alternating_factor = 1 - nu / (2 * node) if include_linear_correction else 1
            expression += sign * (
                half / (2 * node)
                + a2_half / node**2
                + a3_half / node**3
                - kappa_half * parity * node ** (-nu) * alternating_factor
            )
        return sp.series(sp.simplify(expression), edge_w, 0, 4).removeO()

    corrected_edge = edge_series(True)
    dropped_edge = edge_series(False)
    corrected_cubic = sp.expand(corrected_edge).coeff(edge_w, 3)
    dropped_cubic = sp.expand(dropped_edge).coeff(edge_w, 3)

    def edge_validator(value: Any) -> bool:
        return sp.simplify(value - sp.Rational(1, 16)) == 0

    edge_rejected = (
        edge_validator(corrected_cubic)
        and dropped_cubic == -sp.Rational(1, 16)
        and not edge_validator(dropped_cubic)
    )

    if shifted_for_mutation is None or first_shift is None:
        raise RuntimeError(
            "mutation controls did not receive their source certificates"
        )
    bad_column = list(shifted_for_mutation)
    bad_column[0] = -abs(bad_column[0])
    column_rejected = positive_shift_certificate(
        shifted_for_mutation
    ) and not positive_shift_certificate(tuple(bad_column))
    bad_finite = list(first_shift)
    bad_finite[0] = -abs(bad_finite[0])
    finite_rejected = positive_shift_certificate(
        first_shift
    ) and not positive_shift_certificate(tuple(bad_finite))
    even_numerator, _ = symbolic_column_gate("even")
    mutated_threshold_poly = sp.Poly(
        sp.expand(even_numerator.subs(p, w + 112)), w, domain=sp.QQ
    )
    mutated_threshold_fractions = tuple(
        as_fraction(mutated_threshold_poly.nth(j))
        for j in range(mutated_threshold_poly.degree() + 1)
    )
    mutated_threshold_denominator = math.lcm(
        *(value.denominator for value in mutated_threshold_fractions)
    )
    mutated_threshold_coefficients = tuple(
        int(value * mutated_threshold_denominator)
        for value in mutated_threshold_fractions
    )
    threshold_rejected = not positive_shift_certificate(mutated_threshold_coefficients)
    alternating_envelope = as_fraction(e1["envelopes"]["alternating_n^(-2x-3)"])
    envelope_rejected = alternating_envelope < 1 and not alternating_envelope < F(1, 2)
    book.check(
        "mutation: altered E1 coefficient rejected",
        coefficient_rejected,
        coefficient_mutation,
    )
    book.check(
        "mutation: dropped alternating correction rejected",
        edge_rejected,
        {"correct": corrected_cubic, "mutated": dropped_cubic},
    )
    book.check(
        "mutation: bad column threshold certificate rejected",
        column_rejected,
        "negative shifted constant",
    )
    book.check(
        "mutation: bad finite coefficient rejected",
        finite_rejected,
        "negative shifted constant",
    )
    book.check(
        "mutation: changed analytic threshold rejected",
        threshold_rejected,
        {
            "correct": 113,
            "mutated": 112,
            "mutated_constant": mutated_threshold_coefficients[0],
        },
    )
    book.check(
        "mutation: tightened E1 envelope rejected",
        envelope_rejected,
        {"derived": alternating_envelope, "mutated_claim": F(1, 2)},
    )

    lower = {
        "range": "1/2 <= x <= 1",
        "analytic_column": column_results,
        "interior": {
            "c_adverse": c_adverse,
            "c_favorable": c_favorable,
            "analytic_pq_threshold": 36,
            "E1_pq_threshold": 105,
        },
        "finite_residue": {
            "cell_count": len(residue),
            "split": split,
            "minimum_degree": min(degrees),
            "maximum_degree": max(degrees),
            "aggregate_sha256": aggregate.hexdigest(),
        },
        "adjacent_minor_scope": "exact adjacent cells; the standard adjacent-minor reduction is external to this replay",
    }
    mutations = {
        "altered_e1_coefficient": coefficient_rejected,
        "dropped_alternating_linear_correction": edge_rejected,
        "negative_column_shift_coefficient": column_rejected,
        "negative_finite_shift_coefficient": finite_rejected,
        "changed_analytic_threshold": threshold_rejected,
        "tightened_e1_envelope": envelope_rejected,
    }
    return lower, mutations


def expected_a2() -> Any:
    return x * (x - 2) * (2 * x + 1) / 24


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"certificate path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    book = GateBook()
    e1 = derive_e1(book)
    lower, mutations = derive_lower_strip(book, e1)
    if book.failures:
        for gate in book.gates:
            status = "PASS" if gate["passed"] else "FAIL"
            print(f"{status} {gate['name']}: {gate['evidence']}")
        print(
            f"FAILED {len(book.failures)} gate(s): {', '.join(book.failures)}",
            file=sys.stderr,
        )
        return 1

    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    body = {
        "schema": "lower-strip-e1-v1",
        "passed": True,
        "source_sha256": source_sha256,
        "e1": e1,
        "lower_strip": lower,
        "mutation_controls": mutations,
        "gates": book.gates,
    }
    canonical_body = json.dumps(
        json_value(body), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    body["body_sha256"] = hashlib.sha256(canonical_body).hexdigest()
    atomic_json(output, body)
    print(f"PASS {len(book.gates)}/{len(book.gates)} exact gates")
    print("PASS E1 envelopes derived from formulas (7)-(9)")
    print(
        "PASS lower strip: 2 analytic columns, interior pq>=36, 260/260 residue cells"
    )
    print("PASS 6/6 mutation controls rejected")
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
