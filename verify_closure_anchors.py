#!/usr/bin/env python3
"""Reconstruct the exact certificates for the three closure anchors."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import sympy as sp


def endpoint_t(n: int) -> Fraction:
    m = n // 2
    central = Fraction(math.comb(2 * m, m), 4**m)
    if n % 2 == 0:
        return (4 * m + 1) * central
    return (4 * m + 2) * central


def endpoint_ratio(p: int, q: int) -> Fraction:
    a = (p + 1) * (q + 1)
    b = p * q
    c = p * (q + 1)
    d = (p + 1) * q
    return endpoint_t(a) * endpoint_t(b) / (endpoint_t(c) * endpoint_t(d))


def generalized_binomial(value: Fraction, degree: int) -> Fraction:
    result = Fraction(1)
    for index in range(degree):
        result *= value - index
        result /= index + 1
    return result


def defining_t(value: Fraction, n: int) -> Fraction:
    return sum(
        Fraction(2**degree)
        * generalized_binomial(value, degree)
        * math.comb(n, degree)
        for degree in range(n + 1)
    )


def alpha_exp_difference(n: int, shift: int) -> Fraction:
    """Return exp(alpha_n-alpha_(n+shift)) exactly."""

    low = 1 - Fraction(1, 4 * n * n)
    high_n = n + shift
    high = 1 - Fraction(1, 4 * high_n * high_n)
    return high / low


def eta_float(n: int, shift: int) -> float:
    """Finite sanity only; no theorem gate depends on this value."""

    return math.log(float(alpha_exp_difference(n, shift)))


def alternating_eta(start: int, length: int, shift: int) -> float:
    return sum(
        (1.0 if index % 2 == 0 else -1.0)
        * eta_float(start + index, shift)
        for index in range(length)
    )


def smooth_identity_gate() -> tuple[int, str]:
    p, q, u, v = sp.symbols("p q u v", integer=True)

    def smooth(row: sp.Expr) -> sp.Expr:
        b = p * row
        c = p * (row + 1)
        d = (p + 1) * row
        a = (p + 1) * (row + 1)
        return (2 * a + 1) * (2 * b + 1) / ((2 * c + 1) * (2 * d + 1))

    b = p * q
    n = b + 1
    shifted_n = n + 2 * p
    exp_eta = (
        1 - sp.Rational(1, 4) / shifted_n**2
    ) / (1 - sp.Rational(1, 4) / n**2)
    difference = sp.factor(sp.together(smooth(q) / smooth(q + 2) - exp_eta))
    numerator, denominator = sp.fraction(difference)
    expected_denominator = (
        (2 * p * q + 1)
        * (2 * p * q + 3)
        * (p * q + 2 * p + 1) ** 2
        * (2 * p * q + 2 * p + 1)
        * (2 * p * q + 4 * p + 1)
        * (2 * p * q + 2 * q + 1)
        * (2 * p * q + 6 * p + 2 * q + 7)
    )
    assert sp.expand(denominator - expected_denominator) == 0

    positive_part = sp.factor(numerator / 4)
    shifted = sp.Poly(sp.expand(positive_part.subs({p: u + 1, q: v + 2})), u, v)
    coefficients = [int(value) for _, value in shifted.terms()]
    assert len(coefficients) == 41
    assert min(coefficients) == 16
    assert max(coefficients) == 101_008
    assert all(value > 0 for value in coefficients)

    payload = ",".join(str(value) for value in coefficients).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()

    # The q >= 2 fence is active even in this sufficient smooth bound.
    assert difference.subs({p: 8, q: 1}) == sp.Rational(-8392, 127_430_625)
    return len(coefficients), digest


def parity_comparison_gates() -> None:
    p, q = sp.symbols("p q", positive=True, integer=True)

    pair_surplus = sp.expand(4 * p * (q - 1) - (p * q + 3 * p - 1))
    assert sp.expand(pair_surplus - (p * (3 * q - 7) + 1)) == 0
    assert all(
        value > 0
        for value in sp.Poly(
            sp.expand(pair_surplus.subs({p: p + 2, q: q + 3})), p, q
        ).coeffs()
    )

    tail_surplus = sp.expand(3 * (q - 1) - (q + 3))
    assert tail_surplus == 2 * q - 6
    assert sp.expand(tail_surplus.subs(q, q + 5)) == 2 * q + 4

    q3_power = sp.factor((1 + 1 / (3 * p)) ** 3 - (1 + 1 / p))
    q3_num, q3_den = sp.fraction(q3_power)
    assert sp.Poly(q3_num, p).all_coeffs()
    assert all(value > 0 for value in sp.Poly(q3_num, p).coeffs())
    assert q3_den > 0

    # Exact checks of the two integral identities at independent exponents.
    n, s, x, y = sp.symbols("n s x y", positive=True)
    for beta in (2, 4):
        pair_integral = beta * (beta + 1) * sp.integrate(
            sp.integrate((n + x + y) ** (-beta - 2), (y, 0, s)),
            (x, 0, 1),
        )
        pair_direct = (
            n ** (-beta)
            - (n + 1) ** (-beta)
            - (n + s) ** (-beta)
            + (n + s + 1) ** (-beta)
        )
        assert sp.factor(pair_integral - pair_direct) == 0

        tail_integral = beta * sp.integrate(
            (n + y) ** (-beta - 1), (y, 0, s)
        )
        tail_direct = n ** (-beta) - (n + s) ** (-beta)
        assert sp.factor(tail_integral - tail_direct) == 0

    # Direct exact boundary instance of eta_(4p,2p)>eta_(4p+2,2p+2).
    assert alpha_exp_difference(8, 4) - alpha_exp_difference(10, 6) == Fraction(
        2545, 3_907_008
    )


def three_halves_quotient_gate() -> dict[tuple[int, int], str]:
    r, s = sp.symbols("r s", nonnegative=True, integer=True)
    expected = {
        (0, 0): (72, 402_653_184, 18_573_439_451_136,
                 "12c3d85424ead495f25ff27aba2d07301e8abbdc3ee7d56eb20533129495ee3d"),
        (0, 1): (72, 671_088_640, 138_104_133_615_616,
                 "9dcc4ed32301b8c8b3c97ef5bfcee58a3509ba2291ce2bd2e0febe9dc4add61e"),
        (1, 0): (72, 100_663_296, 6_191_307_563_008,
                 "27badf74cfe2b74681f2434764479a8770efc2af407cb2e10fee682523420687"),
        (1, 1): (72, 33_554_432, 15_160_069_849_088,
                 "436d30e25d16f727148f8dbe144b0da1ebd3784f693cd4c167e73fa600cd43ee"),
    }

    def quotient(index: sp.Expr, parity: int) -> sp.Expr:
        if parity == 0:
            return (8 * index**2 + 8 * index + 3) / (3 * (2 * index + 1))
        return 2 * (2 * index + 1) / 3

    def cell(p: sp.Expr, q: sp.Expr, p_parity: int, q_parity: int) -> sp.Expr:
        data = (
            ((p + 1) * (q + 1), (1 - p_parity) * (1 - q_parity)),
            (p * q, p_parity * q_parity),
            (p * (q + 1), p_parity * (1 - q_parity)),
            ((p + 1) * q, (1 - p_parity) * q_parity),
        )
        values = [quotient(index, parity) for index, parity in data]
        return sp.factor(values[0] * values[1] / (values[2] * values[3]))

    digests: dict[tuple[int, int], str] = {}
    for p_parity in (0, 1):
        for q_parity in (0, 1):
            p = 2 * (r + 1) if p_parity == 0 else 2 * r + 1
            q = 2 * (s + 1) if q_parity == 0 else 2 * s + 3
            difference = sp.factor(
                sp.together(
                    cell(p, q, p_parity, q_parity)
                    / cell(p, q + 2, p_parity, q_parity)
                    - 1
                )
            )
            numerator, denominator = sp.fraction(difference)
            assert denominator.subs({r: 0, s: 0}) > 0
            polynomial = sp.Poly(sp.expand(numerator), r, s)
            coefficients = [int(value) for _, value in polynomial.terms()]
            assert all(value > 0 for value in coefficients)
            payload = ",".join(str(value) for value in coefficients).encode("ascii")
            digest = hashlib.sha256(payload).hexdigest()
            summary = (len(coefficients), min(coefficients), max(coefficients), digest)
            assert summary == expected[(p_parity, q_parity)]
            digests[(p_parity, q_parity)] = digest
    return digests


def ratio_monotonicity_strike_gate() -> Fraction:
    w = sp.symbols("w")
    t = sp.symbols("t")
    p_values = [sp.Poly(1, t), sp.Poly(t, t)]
    for n in range(1, 24):
        p_values.append(sp.Poly(t * p_values[n].as_expr() + n**2 * p_values[n - 1].as_expr(), t))

    s_values: list[sp.Expr] = []
    for n, polynomial in enumerate(p_values):
        coefficients = polynomial.all_coeffs()
        expression = polynomial.as_expr()
        if n % 2:
            expression /= t
        expression = sp.Poly(sp.expand(expression), t)
        parity_polynomial = sum(
            coefficient * (w + 4) ** (degree // 2)
            for (degree,), coefficient in expression.terms()
        )
        assert coefficients
        s_values.append(sp.expand(parity_polynomial / math.factorial(n)))

    left_indices = (16, 9, 18, 20)
    right_indices = (12, 12, 24, 15)
    left = sp.prod(s_values[index] for index in left_indices)
    right = sp.prod(s_values[index] for index in right_indices)
    derivative = sp.factor(
        (sp.diff(left, w) / left - sp.diff(right, w) / right).subs(w, 0)
    )
    expected = sp.Rational(-15_735_955_899_430_519_108, 54_938_841_295_981_420_122_345)
    assert derivative == expected
    return Fraction(int(sp.numer(derivative)), int(sp.denom(derivative)))


def direct_endpoint_gates() -> tuple[int, int]:
    for n in range(25):
        endpoint = endpoint_t(n)
        assert defining_t(Fraction(1, 2), n) == endpoint
        quotient = (
            Fraction(8 * n * n + 8 * n + 3, 3 * (2 * n + 1))
            if n % 2 == 0
            else Fraction(2 * (2 * n + 1), 3)
        )
        assert defining_t(Fraction(3, 2), n) == endpoint * quotient

    checked = 0
    for p in range(1, 17):
        for q in range(2, 18):
            assert endpoint_ratio(p, q) > endpoint_ratio(p, q + 2)
            checked += 1

    expected_q1 = {
        7: Fraction(218_009, 63_339_654),
        8: Fraction(-107, 9_088_200),
        9: Fraction(4_176_826, 1_906_449_531),
        10: Fraction(-55_171, 3_139_181_892),
        11: Fraction(3_568_693, 2_353_538_250),
        12: Fraction(-20_509, 1_188_820_360),
    }
    for p, expected in expected_q1.items():
        assert endpoint_ratio(p, 1) - endpoint_ratio(p, 3) == expected

    adverse_checks = 0
    for p in range(2, 18, 2):
        for q in range(3, 20, 2):
            b = p * q
            c = b + p
            d = b + q
            a = b + p + q + 1
            low_remainder = alternating_eta(b + 2, p - 1, 2 * p)
            high_magnitude = alternating_eta(d + 1, p + 1, 2 * p + 2)
            assert low_remainder > high_magnitude
            assert eta_float(c, 2 * p) > eta_float(a - 2, 2 * p + 2)
            adverse_checks += 1
    return checked, adverse_checks


def main() -> None:
    coefficient_count, coefficient_digest = smooth_identity_gate()
    parity_comparison_gates()
    quotient_digests = three_halves_quotient_gate()
    strike = ratio_monotonicity_strike_gate()
    direct_count, adverse_count = direct_endpoint_gates()
    print("closure_anchor_certificates=PASS")
    print(f"smooth_positive_coefficients={coefficient_count}")
    print(f"smooth_coefficient_digest={coefficient_digest}")
    print(f"direct_endpoint_sanity_cells={direct_count}")
    print(f"adverse_parity_sanity_cells={adverse_count}")
    print(f"three_halves_parity_digests={quotient_digests}")
    print(f"ratio_monotonicity_strike={strike}")
    print("all_parameter_gates=9/9")


if __name__ == "__main__":
    main()
