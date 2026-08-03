#!/usr/bin/env python3
"""Reconstruct the algebraic identities used across the Delannoy proof.

This checker is intentionally independent of the manuscript file.  It covers
the defining recurrence, threshold and endpoint anchors, the exact corner
identities behind the margin law, and the finite-dimensional determinant and
grounded-path formulas.  The strip and dimension arguments live in their own
verifiers.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
import math
import sys

import sympy as sp


FAILURES: list[str] = []
CHECKS = 0
EXPECTED_CHECKS = 22


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Record one fail-closed gate; this remains active under ``python -O``."""

    global CHECKS
    CHECKS += 1
    passed = bool(condition)
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not passed:
        FAILURES.append(label)
    return passed


def equal(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.factor(sp.together(left - right)) == 0


def cell(p: int | sp.Expr, q: int | sp.Expr) -> tuple[sp.Expr, ...]:
    return (p + 1) * (q + 1), p * q, p * (q + 1), (p + 1) * q


def generalized_binomial(value: Fraction, degree: int) -> Fraction:
    result = Fraction(1)
    for offset in range(degree):
        result *= value - offset
        result /= offset + 1
    return result


def t_rational(value: Fraction, n: int) -> Fraction:
    return sum(
        Fraction(2**degree * math.comb(n, degree))
        * generalized_binomial(value, degree)
        for degree in range(n + 1)
    )


def t_symbolic(value: sp.Expr, n: int) -> sp.Expr:
    return sp.expand(
        sum(
            2**degree
            * sp.prod(value - offset for offset in range(degree))
            / sp.factorial(degree)
            * sp.binomial(n, degree)
            for degree in range(n + 1)
        )
    )


def p_polynomial(n: int, variable: sp.Expr, coefficient_shift: int = 0) -> sp.Expr:
    """Return P_n with recurrence coefficient ``j^2 + coefficient_shift``."""

    previous, current = sp.Integer(1), variable
    if n == 0:
        return previous
    for index in range(1, n):
        previous, current = (
            current,
            sp.expand(variable * current + (index**2 + coefficient_shift) * previous),
        )
    return current


def endpoint_half(n: int) -> Fraction:
    m = n // 2
    central = Fraction(math.comb(2 * m, m), 4**m)
    return central * (4 * m + (2 if n % 2 else 1))


def three_halves_quotient(n: int) -> Fraction:
    if n % 2:
        return Fraction(2 * (2 * n + 1), 3)
    return Fraction(8 * n * n + 8 * n + 3, 3 * (2 * n + 1))


def skew_path(n: int) -> sp.Matrix:
    matrix = sp.zeros(n, n)
    for index in range(n - 1):
        matrix[index, index + 1] = index + 1
        matrix[index + 1, index] = -(index + 1)
    return matrix


def green_tail(index: int, n: int) -> sp.Rational:
    total = sum(
        (sp.Rational(1, (edge + 1) * (edge + 2)) for edge in range(index, n - 1, 2)),
        sp.Rational(0),
    )
    if (index - (n - 1)) % 2 == 0:
        total += sp.Rational(1, n)
    return total


def parity_blocks(n: int) -> tuple[list[int], list[int]]:
    plus = [index for index in range(n) if (index - n) % 2 == 0]
    minus = [index for index in range(n) if (index - n) % 2 != 0]
    return plus, minus


def block(matrix: sp.Matrix, indices: list[int]) -> sp.Matrix:
    return matrix.extract(indices, indices) if indices else sp.zeros(0, 0)


def q_forms(n: int, y: sp.Symbol) -> tuple[sp.Expr, sp.Expr]:
    inverse = (sp.eye(n) - skew_path(n) ** 2).inv()
    values: list[sp.Expr] = []
    for indices in parity_blocks(n):
        submatrix = block(inverse, indices)
        if not indices:
            values.append(sp.Integer(0))
            continue
        ones = sp.ones(len(indices), 1)
        values.append(
            sp.cancel((ones.T * (sp.eye(len(indices)) + y * submatrix).inv() * ones)[0])
        )
    return values[0], values[1]


def even_polynomial_in_y(expression: sp.Expr, t: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), t)
    if any(degree % 2 for (degree,), _ in polynomial.terms()):
        raise ArithmeticError("expected an even polynomial in t")
    return sp.expand(
        sum(coefficient * (y + 1) ** (degree // 2) for (degree,), coefficient in polynomial.terms())
    )


def normalized_t_in_y(n: int, t: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    parity = n % 2
    expression = sp.cancel(p_polynomial(n, t) / (t**parity * sp.factorial(n)))
    return even_polynomial_in_y(expression, t, y)


def positive_rational_on_positive_axis(expression: sp.Expr, y: sp.Symbol) -> bool:
    numerator, denominator = sp.fraction(sp.cancel(expression))
    numerator_coefficients = sp.Poly(sp.expand(numerator), y).all_coeffs()
    denominator_coefficients = sp.Poly(sp.expand(denominator), y).all_coeffs()
    return (
        bool(numerator_coefficients)
        and any(value > 0 for value in numerator_coefficients)
        and all(value >= 0 for value in numerator_coefficients)
        and bool(denominator_coefficients)
        and all(value > 0 for value in denominator_coefficients)
    )


def main() -> int:
    x, t, y = sp.symbols("x t y")
    p, q = sp.symbols("p q", positive=True, integer=True)

    recurrence_matches = all(
        equal(t_symbolic(x, n), p_polynomial(n, 2 * x + 1) / sp.factorial(n))
        for n in range(13)
    )
    check("defining sum agrees with the three-term recurrence through n=12", recurrence_matches)

    a, b, c, d = cell(p, q)
    check(
        "adjacent cells preserve product and have unit additive defect",
        equal(a * b, c * d) and equal(a + b - c - d, 1),
    )

    a11, b11, c11, d11 = cell(1, 1)
    delta11 = sp.expand(
        t_symbolic(x, int(a11)) * t_symbolic(x, int(b11))
        - t_symbolic(x, int(c11)) * t_symbolic(x, int(d11))
    )
    first_cell = sp.Rational(2, 3) * x * (2 * x**4 - x**3 + x + 1)
    check("first-cell threshold factorization", equal(delta11, first_cell))
    quartic = sp.Poly(2 * x**4 - x**3 + x + 1, x)
    check("the first-cell quartic has no real roots", len(sp.real_roots(quartic)) == 0)
    check(
        "x=0 is degenerate",
        all(t_rational(Fraction(0), n) == 1 for n in range(16)) and delta11.subs(x, 0) == 0,
    )

    check(
        "x=1/2 parity closed form",
        all(t_rational(Fraction(1, 2), n) == endpoint_half(n) for n in range(25)),
    )
    check(
        "x=1 closed form and constant adjacent margin",
        all(t_rational(Fraction(1), n) == 2 * n + 1 for n in range(25))
        and equal((2 * a + 1) * (2 * b + 1) - (2 * c + 1) * (2 * d + 1), 2),
    )
    check(
        "x=3/2 quotient over the endpoint",
        all(
            t_rational(Fraction(3, 2), n) == endpoint_half(n) * three_halves_quotient(n)
            for n in range(25)
        ),
    )

    corner = lambda expression: sp.factor(
        expression.subs({p: p, q: q})
    )
    reciprocal_first = 1 / a + 1 / b - 1 / c - 1 / d
    reciprocal_second = 1 / a**2 + 1 / b**2 - 1 / c**2 - 1 / d**2
    scale = p * (p + 1) * q * (q + 1)
    check("four-corner identity for n^-1", equal(corner(reciprocal_first), 1 / scale))
    check(
        "four-corner identity for n^-2",
        equal(corner(reciprocal_second), (2 * p + 1) * (2 * q + 1) / scale**2),
    )

    determinant_recurrence = True
    factorial_determinant = True
    dpp_determinant = True
    green_entries = True
    green_minors = True
    matrices: dict[int, sp.Matrix] = {}
    for n in range(1, 10):
        path = skew_path(n)
        gram = sp.eye(n) - path**2
        inverse = gram.inv()
        matrices[n] = inverse
        determinant_recurrence = determinant_recurrence and equal(
            (t * sp.eye(n) - path).det(), p_polynomial(n, t)
        )
        factorial_determinant = factorial_determinant and gram.det() == sp.factorial(n) ** 2
        determinant = sp.expand((sp.eye(n) + y * inverse).det())
        square = even_polynomial_in_y(p_polynomial(n, t) ** 2, t, y) / sp.factorial(n) ** 2
        dpp_determinant = dpp_determinant and equal(determinant, square)
        for row in range(n):
            for column in range(n):
                expected = (
                    green_tail(max(row, column), n)
                    if (row - column) % 2 == 0
                    else sp.Integer(0)
                )
                green_entries = green_entries and inverse[row, column] == expected
        for indices in parity_blocks(n):
            for size in range(1, len(indices) + 1):
                for chosen in combinations(indices, size):
                    expected = green_tail(chosen[-1], n)
                    for left, right in zip(chosen, chosen[1:]):
                        expected *= green_tail(left, n) - green_tail(right, n)
                    actual = inverse.extract(chosen, chosen).det()
                    green_minors = green_minors and actual == expected and actual > 0

    check("characteristic determinant equals P_n through n=9", determinant_recurrence)
    check("det(I-K_n^2)=(n!)^2 through n=9", factorial_determinant)
    check("squared Delannoy determinant identity through n=9", dpp_determinant)
    check("grounded-path Green entries through n=9", green_entries)
    check("grounded-path principal-minor product formula through n=9", green_minors)

    updates_hold = True
    for n in range(2, 8):
        old_plus, old_minus = parity_blocks(n)
        new_plus, new_minus = parity_blocks(n + 2)
        alpha = sp.Rational(1, (n + 1) * (n + 2))
        beta = sp.Rational(1, n + 2)
        plus_old = block(matrices[n], old_plus)
        minus_old = block(matrices[n], old_minus)
        plus_expected = (plus_old + alpha * sp.ones(len(old_plus))).row_join(
            alpha * sp.ones(len(old_plus), 1)
        ).col_join(
            (alpha * sp.ones(1, len(old_plus))).row_join(sp.Matrix([[alpha]]))
        )
        minus_expected = (minus_old - alpha * sp.ones(len(old_minus))).row_join(
            beta * sp.ones(len(old_minus), 1)
        ).col_join(
            (beta * sp.ones(1, len(old_minus))).row_join(sp.Matrix([[beta]]))
        )
        updates_hold = updates_hold and block(matrices[n + 2], new_plus) == plus_expected
        updates_hold = updates_hold and block(matrices[n + 2], new_minus) == minus_expected
    check("opposite-sign rank-one parity updates through n=7", updates_hold)

    paired_law = True
    complement_law = True
    scalar_recurrence = True
    for n in range(1, 8):
        q_plus, q_minus = q_forms(n, y)
        alpha = sp.Rational(1, (n + 1) * (n + 2))
        beta = sp.Rational(1, n + 2)
        ratio = sp.cancel(normalized_t_in_y(n + 2, t, y) / normalized_t_in_y(n, t, y))
        paired_law = paired_law and equal(ratio, 1 + alpha * y * (1 + q_plus))
        paired_law = paired_law and equal(
            ratio, 1 + beta * y - alpha * y * (1 + y) * q_minus
        )
        complement_law = complement_law and equal(q_plus + (1 + y) * q_minus, n)
        next_plus, _ = q_forms(n + 2, y)
        scalar_recurrence = scalar_recurrence and equal(
            next_plus, (1 + q_plus) / (1 + alpha * y * (1 + q_plus))
        )
    check("paired determinant-ratio law through n=7", paired_law)
    check("resolvent complement law through n=7", complement_law)
    check("positive-block scalar recurrence through n=7", scalar_recurrence)

    enclosure = True
    for n in range(2, 13):
        ratio = sp.cancel(normalized_t_in_y(n + 2, t, y) / normalized_t_in_y(n, t, y))
        lower = 1 + y / ((n + 1) * (n + 2))
        middle = 1 + y * (1 + n // 2) / ((n + 1) * (n + 2))
        upper = 1 + y / (n + 2)
        enclosure = enclosure and positive_rational_on_positive_axis(ratio - lower, y)
        enclosure = enclosure and positive_rational_on_positive_axis(middle - ratio, y)
        enclosure = enclosure and positive_rational_on_positive_axis(upper - middle, y)
    check("paired-ratio enclosure has positive cleared coefficients through n=12", enclosure)
    ratio_one = sp.cancel(normalized_t_in_y(3, t, y) / normalized_t_in_y(1, t, y))
    check("n=1 lower enclosure edge is equality", equal(ratio_one, 1 + y / 6))

    mutated_matches = all(
        equal(t_symbolic(x, n), p_polynomial(n, 2 * x + 1, coefficient_shift=1) / sp.factorial(n))
        for n in range(2, 7)
    )
    check("negative control rejects n^2 -> n^2+1 in the recurrence", not mutated_matches)

    if CHECKS != EXPECTED_CHECKS:
        FAILURES.append(f"gate count {CHECKS}, expected {EXPECTED_CHECKS}")
    if FAILURES:
        print(f"STRUCTURAL: FAIL ({len(FAILURES)} failure(s), {CHECKS} gates)")
        for label in FAILURES:
            print(f"  - {label}")
        return 1
    print(f"STRUCTURAL: PASS ({CHECKS} gates, including one rejected mutation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
