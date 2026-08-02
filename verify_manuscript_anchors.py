#!/usr/bin/env python3.14t
"""Recompute the numeric anchors quoted in the Delannoy TP2 manuscript.

Coverage is the list enumerated in the manuscript's Reproducibility section;
the polarization values and the 4,168/775 census in the negative-results
register are quoted there with their derivations and are not gated here.

Exits nonzero on any mismatch.  Each check prints the claim as it appears in
the manuscript alongside the recomputed value, so a failure names the exact
statement that drifted.

Run:  PYTHONNOUSERSITE=1 python3.14t verify_manuscript_anchors.py
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import sys

import sympy as sp


FAILURES: list[str] = []
CHECKS = 0


def check(label: str, actual, expected) -> None:
    global CHECKS
    CHECKS += 1
    ok = sp.simplify(actual - expected) == 0 if isinstance(
        actual, sp.Expr
    ) or isinstance(expected, sp.Expr) else actual == expected
    print(f"[{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        print(f"        expected {expected}")
        print(f"        actual   {actual}")
        FAILURES.append(label)


def cell(p: int, q: int) -> tuple[int, int, int, int]:
    """Adjacent multiplication-table cell (a, b, c, d) with ab=cd."""
    return (p + 1) * (q + 1), p * q, p * (q + 1), q * (p + 1)


# ---------------------------------------------------------------- the object

def T_sym(x, n: int):
    """T(x,n) = sum_j 2^j binom(x,j) binom(n,j), exact in x."""
    return sp.expand(
        sum(2**j * sp.binomial(x, j) * sp.binomial(n, j) for j in range(n + 1))
    )


def T_rat(x: Fraction, n: int) -> Fraction:
    total = Fraction(0)
    for j in range(n + 1):
        binom_x = Fraction(1)
        for offset in range(j):
            binom_x *= (x - offset) / (offset + 1)
        total += Fraction(2**j) * binom_x * Fraction(int(sp.binomial(n, j)))
    return total


def P(n: int, t):
    """P_{n+1} = t P_n + n^2 P_{n-1}, with T(x,n) = P_n(2x+1)/n!."""
    previous, current = sp.Integer(1), t
    if n == 0:
        return previous
    for index in range(1, n):
        previous, current = current, sp.expand(t * current + index**2 * previous)
    return current


x = sp.symbols("x")
t = sp.symbols("t")

# The two definitions of T agree.  This is the hinge the whole paper turns on.
for n in range(0, 12):
    check(
        f"T(x,{n}) = P_{n}(2x+1)/{n}!",
        sp.expand(T_sym(x, n) - sp.expand(P(n, 2 * x + 1) / sp.factorial(n))),
        0,
    )

# ------------------------------------------------------- Prop: index identity
for p in range(1, 8):
    for q in range(1, 8):
        a, b, c, d = cell(p, q)
        check(f"cell({p},{q}): ab=cd", a * b, c * d)
        check(f"cell({p},{q}): a+b-c-d=1", a + b - c - d, 1)

# ------------------------------------------------ Thm A: the first-cell cubic
a, b, c, d = cell(1, 1)
delta_11 = sp.factor(sp.expand(T_sym(x, a) * T_sym(x, b) - T_sym(x, c) * T_sym(x, d)))
check(
    "Delta_{1,1}(x) = (2x/3)(2x^4 - x^3 + x + 1)",
    sp.expand(delta_11 - sp.Rational(2, 3) * x * (2 * x**4 - x**3 + x + 1)),
    0,
)
# The quartic is positive for every real x, so the sign of Delta_{1,1} is the
# sign of x.  Confirm it has no real root.
quartic = 2 * x**4 - x**3 + x + 1
check(
    "2x^4 - x^3 + x + 1 has no real root",
    len([r for r in sp.real_roots(sp.Poly(quartic, x))]),
    0,
)

# ------------------------------------------------ Thm: the x=1 degeneracy
for n in range(0, 15):
    check(f"T(1,{n}) = 2*{n}+1", T_sym(sp.Integer(1), n), 2 * n + 1)
for p in range(1, 7):
    for q in range(1, 7):
        a, b, c, d = cell(p, q)
        value = (2 * a + 1) * (2 * b + 1) - (2 * c + 1) * (2 * d + 1)
        check(f"Delta_{{{p},{q}}}(1) = 2", value, 2)

# ------------------------------------------------ Thm: reflection / Pell
for n in range(0, 11):
    alternating = sp.expand(
        sum((-1) ** k * T_sym(x, k) * T_sym(x, n - k) for k in range(n + 1))
    )
    check(
        f"sum_k (-1)^k T(x,k)T(x,{n}-k) = [{n} even]",
        alternating,
        sp.Integer(1) if n % 2 == 0 else sp.Integer(0),
    )

# ------------------------------------------- Thm: Pell-only Gram obstruction
z = sp.symbols("z")
ORDER = 14


def series_coefficients(expression, order: int) -> list:
    expanded = sp.series(expression, z, 0, order).removeO()
    poly = sp.Poly(sp.expand(expanded), z)
    return [sp.nsimplify(poly.coeff_monomial(z**k)) for k in range(order)]


g = sp.symbols("g")
V0 = sp.exp(2 * z) / sp.sqrt(1 - z**2)
Vg = sp.exp(2 * z + g * z**3) / sp.sqrt(1 - z**2)
v0 = series_coefficients(V0, ORDER)
vg = series_coefficients(Vg, ORDER)

# V0 satisfies the reflection relation to the computed order.
reflection = sp.series(V0.subs(z, z) * V0.subs(z, -z) - 1 / (1 - z**2), z, 0, ORDER)
check("V0(z)V0(-z) = 1/(1-z^2)", sp.simplify(reflection.removeO()), 0)

# V0 matches the true endpoint through degree two but fails the first cell.
endpoint = [T_rat(Fraction(1, 2), n) for n in range(3)]
check(
    "V0 matches T(1/2,n) through degree 2",
    [sp.nsimplify(value) for value in v0[:3]],
    [sp.Rational(value.numerator, value.denominator) for value in endpoint],
)

a, b, c, d = cell(1, 1)
check(
    "M_{1,1}(V0) = -13/6",
    sp.nsimplify(sp.expand(v0[a] * v0[b] - v0[c] * v0[d])),
    sp.Rational(-13, 6),
)


check(
    "M_{1,1}(V_g) = 4g - 13/6",
    sp.expand(vg[a] * vg[b] - vg[c] * vg[d] - (4 * g - sp.Rational(13, 6))),
    0,
)
check(
    "true endpoint odd generator g = 2/3 restores M_{1,1} = 1/2",
    sp.expand((4 * g - sp.Rational(13, 6)).subs(g, sp.Rational(2, 3))),
    sp.Rational(1, 2),
)

# ------------------------------------------- Thm: two-parameter region false
alpha, beta, lam = sp.symbols("alpha beta lam")


def T_two(alpha_value, beta_value, n: int):
    expression = (1 + z) ** alpha_value * (1 - z) ** (-beta_value)
    expanded = sp.series(expression, z, 0, n + 1).removeO()
    return sp.expand(sp.Poly(sp.expand(expanded), z).coeff_monomial(z**n))


def minor_two(alpha_value, beta_value, p: int, q: int):
    a, b, c, d = cell(p, q)
    return sp.simplify(
        T_two(alpha_value, beta_value, a) * T_two(alpha_value, beta_value, b)
        - T_two(alpha_value, beta_value, c) * T_two(alpha_value, beta_value, d)
    )


check(
    "(alpha,beta)=(3/4,5/4), cell (2,1) = -25/512",
    sp.nsimplify(minor_two(sp.Rational(3, 4), sp.Rational(5, 4), 2, 1)),
    sp.Rational(-25, 512),
)
check(
    "(alpha,beta)=(3,2), first cell = -4",
    sp.nsimplify(minor_two(sp.Integer(3), sp.Integer(2), 1, 1)),
    sp.Integer(-4),
)
check(
    "M_{1,1}(lam,lam) = (4/3) lam^3 (lam-1)(lam-2)",
    sp.expand(
        minor_two(lam, lam, 1, 1)
        - sp.Rational(4, 3) * lam**3 * (lam - 1) * (lam - 2)
    ),
    0,
)

# ---------------------------------- Thm D/E: determinant, complement, ratio
def K(n: int):
    matrix = sp.zeros(n, n)
    for j in range(n - 1):
        matrix[j, j + 1] = j + 1
        matrix[j + 1, j] = -(j + 1)
    return matrix


y = sp.symbols("y")
for n in range(1, 9):
    Kn = K(n)
    check(f"det(tI - K_{n}) = P_{n}(t)", sp.expand((t * sp.eye(n) - Kn).det() - P(n, t)), 0)
    An = sp.eye(n) - Kn * Kn
    check(f"det(I - K_{n}^2) = ({n}!)^2", sp.expand(An.det()), sp.factorial(n) ** 2)
    Bn = An.inv()
    left = sp.expand(sp.simplify((sp.eye(n) + y * Bn).det()))
    right = sp.expand(
        sp.simplify((P(n, t) / sp.factorial(n)) ** 2).subs(t, sp.sqrt(y + 1))
    )
    check(f"T_{n}^2 = det(I + y B_{n})", sp.simplify(left - right), 0)

# Green kernel entries.
def s_entry(j: int, n: int):
    total = sum(
        sp.Rational(1, (k + 1) * (k + 2))
        for k in range(j, n - 1)
        if (k - j) % 2 == 0
    )
    if (j - (n - 1)) % 2 == 0:
        total += sp.Rational(1, n)
    return total


for n in range(1, 10):
    Bn = (sp.eye(n) - K(n) * K(n)).inv()
    for i in range(n):
        for j in range(n):
            expected = s_entry(max(i, j), n) if (i - j) % 2 == 0 else sp.Integer(0)
            if sp.simplify(Bn[i, j] - expected) != 0:
                FAILURES.append(f"Green kernel entry ({i},{j}) at n={n}")
print(f"[{'ok ' if not any('Green kernel' in f for f in FAILURES) else 'FAIL'}]"
      " Green kernel entries (B_n)_{ij} = s_max(i,j) on same parity, 0 otherwise")

# Complement law and the paired-ratio enclosure.
def Q_blocks(n: int):
    Bn = (sp.eye(n) - K(n) * K(n)).inv()
    plus = [i for i in range(n) if (i - n) % 2 == 0]
    minus = [i for i in range(n) if (i - n) % 2 != 0]
    results = []
    for block in (plus, minus):
        if not block:
            results.append(sp.Integer(0))
            continue
        sub = Bn[block, block]
        size = len(block)
        ones = sp.ones(size, 1)
        results.append(
            sp.simplify((ones.T * (sp.eye(size) + y * sub).inv() * ones)[0, 0])
        )
    return results


for n in range(1, 9):
    q_plus, q_minus = Q_blocks(n)
    check(
        f"Q_{n}^+ + (1+y) Q_{n}^- = {n}",
        sp.simplify(q_plus + (1 + y) * q_minus - n),
        0,
    )

check(
    "T_3/T_1 = 1 + y/6 (the n=1 equality edge)",
    sp.simplify(
        (P(3, t) / sp.factorial(3)) / (P(1, t) / sp.factorial(1))
        - (1 + (t**2 - 1) / 6)
    ),
    0,
)

# The enclosure, checked exactly at sample (n, y).
for n in range(2, 12):
    for y_value in (sp.Rational(1, 2), sp.Integer(3), sp.Integer(17)):
        ratio = sp.simplify(
            ((P(n + 2, t) / sp.factorial(n + 2)) / (P(n, t) / sp.factorial(n))).subs(
                t, sp.sqrt(y_value + 1)
            )
        )
        low = 1 + y_value / ((n + 1) * (n + 2))
        mid = 1 + y_value * (1 + n // 2) / ((n + 1) * (n + 2))
        high = 1 + y_value / (n + 2)
        if not (low < ratio < mid <= high or low < ratio < mid < high):
            FAILURES.append(f"enclosure at n={n}, y={y_value}")
print(f"[{'ok ' if not any('enclosure' in f for f in FAILURES) else 'FAIL'}]"
      " paired-ratio enclosure strict on 2 <= n <= 11")

# ------------------------------------------- the (2,2) base of the induction
def T_of_y(n: int):
    return sp.expand(sp.simplify((P(n, t) / sp.factorial(n)) ** 2).rewrite(sp.Pow)
                     .subs(t**2, y + 1))


w = sp.symbols("w")


def D_squared(p: int, q: int):
    a, b, c, d = cell(p, q)

    def g(n):
        return n * sp.expand(sp.Poly(P(n, t) ** 2, t).as_expr() / sp.factorial(n) ** 2)

    expression = sp.expand(g(a) * g(b) - g(c) * g(d))
    return sp.expand(expression.subs(t**2, y + 1).subs(t, sp.sqrt(y + 1)))


base = sp.simplify(D_squared(2, 2).subs(y, w + 3))
base_poly = sp.Poly(sp.expand(base), w)
factored = sp.factor(base)
check(
    "D_{2,2}(3+w) has the factor (w+3)",
    sp.simplify(sp.rem(sp.Poly(sp.expand(base), w), sp.Poly(w + 3, w)).as_expr()),
    0,
)
quotient = sp.Poly(sp.div(sp.Poly(sp.expand(base), w), sp.Poly(w + 3, w))[0], w)
coefficients = [sp.nsimplify(value) for value in quotient.all_coeffs()]
expected_descending = [
    sp.Rational(value, 52672757760000)
    for value in (
        25, 5069, 481435, 28845643, 1156390886, 30889085506,
        567494128662, 7287805570806, 63219692542293,
        348617270894625, 1180302849086175,
        2353083881439375, 2248473972937500,
    )
]
check("D_{2,2} quotient coefficient vector", coefficients, expected_descending)

# The y=0 expansion is not a valid coefficient bottom rung.
check(
    "[y] D_{1,2} = -8/5",
    sp.nsimplify(sp.Poly(sp.expand(D_squared(1, 2)), y).coeff_monomial(y)),
    sp.Rational(-8, 5),
)
check(
    "[y](D_{1,4} - D_{1,2}) = -88/45",
    sp.nsimplify(
        sp.Poly(sp.expand(D_squared(1, 4) - D_squared(1, 2)), y).coeff_monomial(y)
    ),
    sp.Rational(-88, 45),
)
check(
    "every D_{2,2} quotient coefficient is positive",
    all(value > 0 for value in coefficients),
    True,
)

# ------------------------------------ the partial-summation counterexample
witness = [
    Fraction(1, 2), Fraction(1), Fraction(2), Fraction(39, 10),
    Fraction(41, 10), Fraction(21, 5), Fraction(81, 10), Fraction(41, 5),
    Fraction(83, 10), Fraction(161, 10),
]


def minor_of(sequence: dict, p: int, q: int):
    a, b, c, d = cell(p, q)
    if max(a, b, c, d) not in sequence:
        return None
    return sequence[a] * sequence[b] - sequence[c] * sequence[d]


# The witness is 0-indexed.  The 1-indexed reading gives D_S(2,1) = -47/20
# with two negative original cells, so it is not the sequence in the report.
f = {i: value for i, value in enumerate(witness)}
partial = {}
running = Fraction(0)
for index in sorted(f):
    running += f[index]
    partial[index] = running

check("witness original cell (1,1) = 1/10", minor_of(f, 1, 1), Fraction(1, 10))
check("witness original cell (1,2) = 21/100", minor_of(f, 1, 2), Fraction(21, 100))
check("witness original cell (2,1) = 21/100", minor_of(f, 2, 1), Fraction(21, 100))
check("witness original cell (2,2) = 2/5", minor_of(f, 2, 2), Fraction(2, 5))
check(
    "witness is strictly increasing",
    all(witness[i] < witness[i + 1] for i in range(len(witness) - 1)),
    True,
)
check("partial sums have D_S(2,1) = -9/5", minor_of(partial, 2, 1), Fraction(-9, 5))
check(
    "partial sums preserve the (1,1) cell",
    minor_of(partial, 1, 1) >= minor_of(f, 1, 1) > 0,
    True,
)

# ------------------------------------ the H all-parameter state certificate
def positive_coefficient_summary(expression, variables):
    polynomial = sp.Poly(sp.expand(expression), *variables, domain=sp.QQ)
    coefficients = polynomial.coeffs()
    return (
        len(coefficients),
        min(coefficients),
        bool(coefficients and all(value > 0 for value in coefficients)),
    )


def h_state_map(index, center, state):
    return sp.cancel(
        (
            (center - 4) * (center - 2)
            + (center - 2 * index - 4) * state
        )
        / (center + state + 2 * index - 2)
    )


def h_boundary_value(index, center, state):
    error = center * (center - 2) / (4 * index + state)
    rho = 2 * index / (2 * index + center + error)
    return sp.cancel(
        (2 * index + 1)
        / center
        * ((index + 1) * (1 - rho**2) - center * rho)
    )


def h_bernstein_coefficients(expression, variable, degree):
    polynomial = sp.Poly(sp.expand(expression), variable, domain="EX")
    power = [
        polynomial.coeff_monomial(variable**k)
        for k in range(degree + 1)
    ]
    return [
        sp.factor(
            sum(
                power[k]
                * sp.Rational(
                    sp.binomial(j, k),
                    sp.binomial(degree, k),
                )
                for k in range(j + 1)
            )
        )
        for j in range(degree + 1)
    ]


def verify_h_dimension_step_certificate():
    center, r, zeta, m = sp.symbols(
        "center r zeta m", nonnegative=True
    )

    for parity in (0, 1):
        difference = sp.cancel(
            h_boundary_value(2 * r + parity, center, 4 - center)
            - h_boundary_value(r, center, center - 4)
        )
        numerator, denominator = difference.as_numer_denom()
        substitutions = {center: zeta + 3, r: m + 1}

        check(
            f"H dimension step [3,4], parity {parity}: "
            "numerator coefficients",
            positive_coefficient_summary(
                numerator.subs(substitutions), (zeta, m)
            ),
            (38, sp.Integer(8), True),
        )
        check(
            f"H dimension step [3,4], parity {parity}: "
            "denominator coefficients",
            positive_coefficient_summary(
                denominator.subs(substitutions), (zeta, m)
            ),
            (35, sp.Integer(2), True),
        )

    h, basis = sp.symbols("h basis", nonnegative=True)
    for parity in (0, 1):
        difference = sp.cancel(
            h_boundary_value(2 * r + parity, h + 4, h)
            - h_boundary_value(r, h + 4, 0)
        )
        numerator, denominator = difference.as_numer_denom()
        bernstein = h_bernstein_coefficients(numerator, h, 7)
        table = sum(
            sp.expand(35 * coefficient.subs(r, m + 1))
            * basis**index
            for index, coefficient in enumerate(bernstein)
        )

        check(
            f"H dimension step [4,5], parity {parity}: "
            "Bernstein coefficients",
            (sp.degree(numerator, h),)
            + positive_coefficient_summary(table, (basis, m)),
            (7, 64, sp.Integer(35840), True),
        )
        check(
            f"H dimension step [4,5], parity {parity}: "
            "denominator coefficients",
            positive_coefficient_summary(
                denominator.subs(r, m + 1), (h, m)
            ),
            (45, sp.Integer(1), True),
        )

    root_sum = sp.symbols("root_sum", nonnegative=True)
    for parity, expected_terms in ((0, 194), (1, 221)):
        large = 2 * r + parity
        small_parameter = 2 * r + 1
        large_parameter = 2 * large - 1
        total = small_parameter + large_parameter

        left = sp.cancel(
            root_sum
            * (root_sum + 2 * large_parameter)
            / (2 * (root_sum + total))
        )
        right = sp.cancel(
            root_sum
            * (root_sum + 2 * small_parameter)
            / (2 * (root_sum + total))
        )

        difference = sp.cancel(
            h_boundary_value(large, h + 4, right)
            - h_boundary_value(r, h + 4, left)
        )
        numerator, denominator = sp.together(
            difference
        ).as_numer_denom()

        level = sp.cancel(left * (left + 2 * small_parameter))
        relation = sp.together(h**2 + 2 * h - level).as_numer_denom()[0]

        quotient, remainder = sp.Poly(
            numerator, h, domain="EX"
        ).div(sp.Poly(relation, h, domain="EX"))
        quotient = quotient.as_expr()
        remainder = remainder.as_expr()

        remainder_numerator, remainder_denominator = sp.cancel(
            remainder
        ).as_numer_denom()
        core = sp.cancel(
            remainder_numerator / (2 * r if parity == 0 else 2)
        )

        check(
            f"H dimension step [5,infinity), parity {parity}: "
            "reduction identities",
            (
                sp.factor(left + right - root_sum),
                sp.factor(
                    left * (left + 2 * small_parameter)
                    - right * (right + 2 * large_parameter)
                ),
                sp.factor(numerator - quotient * relation - remainder),
                sp.denom(core),
            ),
            (0, 0, 0, 1),
        )

        substitutions = {h: zeta + 1, r: m + 1}
        check(
            f"H dimension step [5,infinity), parity {parity}: "
            "remainder coefficients",
            positive_coefficient_summary(
                core.subs(substitutions),
                (zeta, root_sum, m),
            ),
            (expected_terms, sp.Integer(1), True),
        )
        check(
            f"H dimension step [5,infinity), parity {parity}: "
            "denominator coefficients",
            (
                positive_coefficient_summary(
                    denominator.subs(substitutions),
                    (zeta, root_sum, m),
                ),
                positive_coefficient_summary(
                    remainder_denominator.subs(r, m + 1),
                    (root_sum, m),
                ),
            ),
            (
                (415, sp.Integer(1), True),
                (6, sp.Integer(1), True),
            ),
        )


def h_adjacent_ratio(index, center, state):
    return sp.cancel(
        (
            center**2
            + center * state
            + 3 * center * index
            - 3 * center
            + state * index
            - state
            + 4 * index**2
            - 2 * index
            + 2
        )
        / ((index + 1) * (center + state + 4 * index - 2))
    )


def h_two_step_ratio(index, center, state):
    adjacent = h_adjacent_ratio(index, center, state)
    return sp.cancel(((center - 1) * adjacent + index + 1) / (index + 2))


def verify_h_base_strip():
    n, center, state, r, zeta, h, s, slack = sp.symbols(
        "n center state r zeta h s slack",
        nonnegative=True,
    )

    rho = sp.symbols("rho", nonnegative=True)

    def rho_step(index, value):
        return sp.cancel(index / (center + index * value))

    def lower_barrier(index):
        return sp.cancel(
            (4 * index + 2 - center) / (4 * index + 2 + center)
        )

    barrier_residual = sp.cancel(
        rho_step(n + 2, rho_step(n + 1, lower_barrier(n)))
        - lower_barrier(n + 2)
    )
    declared_residual = sp.cancel(
        center
        * (center - 3)
        * (center - 2) ** 2
        / (
            (center + 4 * n + 10)
            * (
                center**3
                + 3 * center**2 * n
                + center**2
                + 5 * center * n**2
                + 9 * center * n
                + 4 * center
                + 4 * n**3
                + 14 * n**2
                + 14 * n
                + 4
            )
        )
    )
    q_from_rho = sp.cancel(
        (2 * n + 1)
        / center
        * ((n + 1) * (1 - rho**2) - center * rho)
    )
    declared_q_gap = sp.cancel(
        2
        * center
        * ((4 - center) * n + 2)
        / (center + 4 * n + 2) ** 2
    )
    check(
        "H q-cap [3,4]: parity bases, residual, and barrier identity",
        (
            sp.factor(
                -lower_barrier(0) - (center - 2) / (center + 2)
            ),
            sp.factor(
                1 / center
                - lower_barrier(1)
                - (center - 3)
                * (center - 2)
                / (center * (center + 6))
            ),
            sp.factor(barrier_residual - declared_residual),
            sp.factor(
                sp.diff(q_from_rho, rho)
                + (2 * n + 1)
                * (center + 2 * (n + 1) * rho)
                / center
            ),
            sp.factor(
                1
                - q_from_rho.subs(rho, lower_barrier(n))
                - declared_q_gap
            ),
        ),
        (0, 0, 0, 0, 0),
    )

    series_variable = sp.symbols("series_variable")
    logarithmic_derivative = (
        (center - 2) / (2 * (1 + series_variable))
        + center / (2 * (1 - series_variable))
    )
    contiguous_series_identity = sp.factor(
        center * (1 + series_variable) / (1 - series_variable)
        - (center - 2)
        * (1 - series_variable)
        / (1 + series_variable)
        - 2 * (2 * series_variable * logarithmic_derivative + 1)
    )
    s_current, s_next = sp.symbols(
        "s_current s_next", positive=True
    )
    q_current = (
        2 * (2 * n + 1) + (center - 2) * s_current
    ) / center
    q_next = (
        2 * (2 * n + 3) + (center - 2) * s_next
    ) / center
    weighted_gap = sp.cancel((n + 1) * q_current - n * q_next)
    declared_weighted_gap = (
        2
        + (center - 2) * ((n + 1) * s_current - n * s_next)
    ) / center
    quotient = sp.cancel(q_next / q_current)
    k_from_quotient = sp.cancel((quotient - 1) / (quotient + 1))
    check(
        "H q-cap [4,infinity): contiguous identity and quotient algebra",
        (
            contiguous_series_identity,
            sp.factor(weighted_gap - declared_weighted_gap),
            sp.factor(
                1
                - (2 * n + 1) * k_from_quotient
                - 2 * weighted_gap / (q_current + q_next)
            ),
        ),
        (0, 0, 0),
    )

    adjacent = h_adjacent_ratio(n, center, state)
    two_step = h_two_step_ratio(n, center, state)

    check(
        "H base strip: two-step ratio reconstruction",
        sp.factor(
            two_step
            - adjacent
            * h_adjacent_ratio(
                n + 1,
                center,
                h_state_map(n, center, state),
            )
        ),
        0,
    )
    check(
        "H base strip: ratio derivatives",
        (
            sp.factor(
                sp.diff(adjacent, state)
                - 2
                * n
                * (center - 2)
                / ((n + 1) * (center + state + 4 * n - 2) ** 2)
            ),
            sp.factor(
                sp.diff(two_step, state)
                - 2
                * n
                * (center - 2)
                * (center - 1)
                / (
                    (n + 1)
                    * (n + 2)
                    * (center + state + 4 * n - 2) ** 2
                )
            ),
        ),
        (0, 0),
    )

    radius = 4 - center
    check(
        "H base strip [3,4]: state trap endpoints",
        (
            sp.factor(
                sp.diff(h_state_map(n, center, state), state)
                + 4
                * n
                * (n + 1)
                / (center + state + 2 * n - 2) ** 2
            ),
            sp.factor(h_state_map(n, center, radius) + radius),
            sp.factor(
                h_state_map(n, center, -radius)
                - radius * (n + 3 - center) / (n + center - 3)
            ),
        ),
        (0, 0, 0),
    )

    bound34 = sp.cancel(
        h_two_step_ratio(2 * r, center, center - 4)
        / h_adjacent_ratio(r, center, 4 - center)
        - 1
    )
    declared34 = sp.cancel(
        zeta
        * (zeta + 1)
        * (2 * r + zeta + 1)
        / (2 * (r + 1) * (4 * r + zeta) * (2 * r + zeta + 2))
    )
    check(
        "H base strip [3,4]: lower-bound factorization",
        sp.factor(bound34.subs(center, zeta + 3) - declared34),
        0,
    )

    endpoint_even = sp.cancel(
        h_two_step_ratio(4 * s, 3, 1)
        / h_adjacent_ratio(2 * s, 3, 1)
        - 1
    )
    endpoint_odd = sp.cancel(
        h_two_step_ratio(4 * s + 2, 3, 1)
        / h_adjacent_ratio(2 * s + 1, 3, -1)
        - 1
    )
    check(
        "H base strip at center 3: alternating state and margins",
        (
            sp.factor(h_state_map(n, 3, -1) - 1),
            sp.factor(h_state_map(n, 3, 1) + 1),
            sp.factor(
                endpoint_even
                - 1 / (4 * (2 * s + 1) ** 2 * (8 * s + 1))
            ),
            sp.factor(
                endpoint_odd - 2 / ((4 * s + 5) * (8 * s + 5))
            ),
        ),
        (0, 0, 0, 0),
    )

    center45 = h + 4
    second = sp.cancel(
        h_state_map(
            n + 1,
            center45,
            h_state_map(n, center45, state),
        )
    )
    check(
        "H base strip [4,5]: two-step state trap",
        (
            sp.factor(
                sp.diff(second, state)
                - 4
                * n
                * (n + 1) ** 2
                * (n + 2)
                / (
                    state * (h + 2)
                    + h**2
                    + 2 * h * n
                    + 4 * h
                    + 2 * n**2
                    + 6 * n
                    + 4
                )
                ** 2
            ),
            sp.factor(
                second.subs(state, 0)
                - h**2
                * (h + 2)
                / (
                    h**2
                    + 2 * h * n
                    + 4 * h
                    + 2 * n**2
                    + 6 * n
                    + 4
                )
            ),
            sp.factor(
                h
                - second.subs(state, h)
                - 2
                * h
                * (h + 1)
                * (n + 1)
                / (h**2 + h * n + 3 * h + n**2 + 3 * n + 2)
            ),
            sp.factor(
                h_state_map(1, center45, h) - h**2 / (h + 2)
            ),
        ),
        (0, 0, 0, 0),
    )

    energy = (
        (8 - 8 * h) * r**2
        + (4 * h**2 + 12 * h + 12) * r
        + h**3
        + 5 * h**2
        + 8 * h
        + 4
    )
    bound45 = sp.cancel(
        h_two_step_ratio(2 * r, h + 4, 0)
        / h_adjacent_ratio(r, h + 4, h)
        - 1
    )
    declared45 = sp.cancel(
        (h + 2)
        * energy
        / (
            2
            * (2 * r + 1)
            * (h + 8 * r + 2)
            * (h**2 + 2 * h * r + 4 * h + 2 * r**2 + 5 * r + 3)
        )
    )
    check(
        "H base strip [4,5]: lower-bound factorization",
        sp.factor(bound45 - declared45),
        0,
    )

    first_energy_part = sp.Poly(
        (8 * (1 - h) * r**2).subs({h: 1 - slack, r: s + 1}),
        slack,
        s,
    ).coeffs()
    check(
        "H base strip [4,5]: energy is positive",
        (
            bool(
                first_energy_part
                and all(value >= 0 for value in first_energy_part)
            ),
            positive_coefficient_summary(
                ((4 * h**2 + 12 * h + 12) * r).subs(r, s + 1),
                (h, s),
            )[2],
            positive_coefficient_summary(
                h**3 + 5 * h**2 + 8 * h + 4,
                (h,),
            )[2],
        ),
        (True, True, True),
    )

    moving_root, fixed_root = sp.symbols(
        "moving_root fixed_root", positive=True
    )
    root_index = sp.symbols("root_index", nonnegative=True)

    def relation_remainder(expression, relation):
        numerator = sp.together(expression).as_numer_denom()[0]
        return sp.factor(
            sp.Poly(numerator, h, domain="EX")
            .rem(sp.Poly(relation, h, domain="EX"))
            .as_expr()
        )

    moving_relation = (
        h**2
        + 2 * h
        - moving_root * (moving_root + 4 * n - 2)
    )
    moving_image = h_state_map(n, h + 4, moving_root)
    declared_image = (
        moving_root
        * (moving_root + h + 2 * n - 2)
        / (moving_root + h + 2 * n + 2)
    )
    next_root_residual = (
        8
        * moving_root
        * (
            (h - 1) * (h + 2 * n + 2)
            + (h + 1) * moving_root
        )
        / (h + 2 * n + moving_root + 2) ** 2
    )
    fixed_relation = (
        h**2 + 2 * h - fixed_root * (fixed_root + 4 * n + 2)
    )
    root_level = h * (h + 2)
    explicit_root = (
        -(2 * root_index + 1)
        + sp.sqrt((2 * root_index + 1) ** 2 + root_level)
    )
    rationalized_root = root_level / (
        sp.sqrt((2 * root_index + 1) ** 2 + root_level)
        + 2 * root_index
        + 1
    )
    root_denominator = (
        sp.sqrt((2 * root_index + 1) ** 2 + root_level)
        + 2 * root_index
        + 1
    )
    residual_numerator, residual_denominator = sp.cancel(
        next_root_residual.subs({h: zeta + 1, n: s + 1})
    ).as_numer_denom()
    check(
        "H base strip [5,infinity): moving-root trap identities",
        (
            sp.simplify(explicit_root - rationalized_root),
            sp.factor(
                sp.diff(root_denominator, root_index)
                - 2
                - 2
                * (2 * root_index + 1)
                / sp.sqrt((2 * root_index + 1) ** 2 + root_level)
            ),
            sp.factor(1 + h * (h + 2) - (h + 1) ** 2),
            relation_remainder(
                h_state_map(n, h + 4, fixed_root) - fixed_root,
                fixed_relation,
            ),
            relation_remainder(
                moving_image - declared_image,
                moving_relation,
            ),
            relation_remainder(
                moving_image**2
                + (4 * n + 6) * moving_image
                - h * (h + 2)
                - next_root_residual,
                moving_relation,
            ),
            positive_coefficient_summary(
                residual_numerator, (zeta, s, moving_root)
            ),
            positive_coefficient_summary(
                residual_denominator, (zeta, s, moving_root)
            ),
        ),
        (
            0,
            0,
            0,
            0,
            0,
            0,
            (5, sp.Integer(8), True),
            (10, sp.Integer(1), True),
        ),
    )

    q = sp.symbols("q", positive=True)
    left_increment = q / (4 * r + 1)
    right_increment = q / (4 * r + 3)
    addition = sp.cancel(
        (left_increment + right_increment)
        / (1 + left_increment * right_increment)
        - q / (2 * r + 1)
    )
    declared_margin = q * (1 - q**2) / (
        (2 * r + 1) * ((4 * r + 1) * (4 * r + 3) + q**2)
    )
    check(
        "H dimension step: final hyperbolic-addition margin",
        sp.factor(addition - declared_margin),
        0,
    )


verify_h_dimension_step_certificate()
verify_h_base_strip()

# ---------------- endpoint and half-integer all-parameter proof certificates
def quotient_minor(p, q, parities, branch):
    a, b, c, d = cell(p, q)
    return sp.cancel(
        branch(a, parities[0]) * branch(b, parities[1])
        - branch(c, parities[2]) * branch(d, parities[3])
    )


def verify_endpoint_argument():
    k = sp.symbols("k", integer=True, positive=True)
    previous_central_ratio = 2 * k / (2 * k - 1)
    next_central_ratio = (2 * k + 1) / (2 * k + 2)

    def even(index):
        return 4 * index + 1

    def odd(index):
        return 2 * (2 * index + 1)

    check(
        "endpoint closed forms satisfy the recurrence and bases",
        (
            sp.factor(
                (2 * k + 1) * odd(k)
                - 2 * even(k)
                - 2 * k * odd(k - 1) * previous_central_ratio
            ),
            sp.factor(
                (2 * k + 2) * even(k + 1) * next_central_ratio
                - 2 * odd(k)
                - (2 * k + 1) * even(k)
            ),
            even(0) - 1,
            odd(0) - 2,
        ),
        (0, 0, 0, 0),
    )

    n = sp.symbols("n", integer=True, positive=True)
    smooth_argument = (2 * n + 1) / (2 * n - 1)
    alpha_argument = 4 * n**2 / (4 * n**2 - 1)

    check(
        "endpoint ratios and logarithmic decomposition",
        (
            sp.factor(
                even(k) / (odd(k - 1) * previous_central_ratio)
                - (4 * k + 1) / (4 * k)
            ),
            sp.factor(odd(k) / even(k) - (4 * k + 2) / (4 * k + 1)),
            sp.factor(
                ((2 * n + 1) / (2 * n)) ** 2
                - smooth_argument / alpha_argument
            ),
            sp.factor(
                (2 * n / (2 * n - 1)) ** 2
                - smooth_argument * alpha_argument
            ),
        ),
        (0, 0, 0, 0),
    )

    next_alpha_argument = 4 * (n + 1) ** 2 / (4 * (n + 1) ** 2 - 1)
    check(
        "endpoint alpha is positive and strictly decreasing",
        (
            sp.factor(
                alpha_argument
                - 1
                - 1 / ((2 * n - 1) * (2 * n + 1))
            ),
            sp.factor(
                alpha_argument
                - next_alpha_argument
                - 4 / ((2 * n - 1) * (2 * n + 1) * (2 * n + 3))
            ),
        ),
        (0, 0),
    )

    p, q = sp.symbols("p q", integer=True, positive=True)
    A, B, C, D = cell(p, q)
    half = sp.Rational(1, 2)
    dominance_gap = (
        4 * (B + 1) ** 2
        - 1
        - 2 * (C + half) * (D + half)
    )
    positive_decomposition = (
        2 * p * q * (p - 1) * (q - 1)
        + p * (q - 1)
        + q * (p - 1)
        + sp.Rational(5, 2)
    )

    check(
        "endpoint smooth-gap and domination identities",
        (
            sp.factor(
                (A + half) * (B + half)
                - (C + half) * (D + half)
                - half
            ),
            sp.factor(
                (A + half)
                * (B + half)
                / ((C + half) * (D + half))
                - 1
                - 1 / (2 * (C + half) * (D + half))
            ),
            sp.factor(dominance_gap - positive_decomposition),
        ),
        (0, 0, 0),
    )


def R1(index, even):
    if even:
        return (8 * index**2 + 8 * index + 3) / (3 * (2 * index + 1))
    return 2 * (2 * index + 1) / 3


def verify_three_halves_argument():
    k = sp.symbols("k", integer=True, positive=True)
    previous_central_ratio = 2 * k / (2 * k - 1)
    next_central_ratio = (2 * k + 1) / (2 * k + 2)

    def half_even(index):
        return 4 * index + 1

    def half_odd(index):
        return 2 * (2 * index + 1)

    def three_even(index):
        return (32 * index**2 + 16 * index + 3) / 3

    def three_odd(index):
        return 4 * (2 * index + 1) * (4 * index + 3) / 3

    check(
        "three-halves closed forms satisfy the recurrence and bases",
        (
            sp.factor(
                (2 * k + 1) * three_odd(k)
                - 4 * three_even(k)
                - 2 * k * three_odd(k - 1) * previous_central_ratio
            ),
            sp.factor(
                (2 * k + 2) * three_even(k + 1) * next_central_ratio
                - 4 * three_odd(k)
                - (2 * k + 1) * three_even(k)
            ),
            three_even(0) - 1,
            three_odd(0) - 4,
        ),
        (0, 0, 0, 0),
    )

    even_index = 2 * k
    odd_index = 2 * k + 1
    check(
        "R1 parity branches",
        (
            sp.factor(three_even(k) / half_even(k) - R1(even_index, True)),
            sp.factor(three_odd(k) / half_odd(k) - R1(odd_index, False)),
        ),
        (0, 0),
    )

    r, s = sp.symbols("r s", nonnegative=True)
    u, v = sp.symbols("u v", nonnegative=True)

    P_oo = (
        3072 * r**3 * s**3
        + 7168 * r**3 * s**2
        + 5120 * r**3 * s
        + 1024 * r**3
        + 7168 * r**2 * s**3
        + 18560 * r**2 * s**2
        + 14848 * r**2 * s
        + 3456 * r**2
        + 5120 * r * s**3
        + 14848 * r * s**2
        + 13288 * r * s
        + 3512 * r
        + 1024 * s**3
        + 3456 * s**2
        + 3512 * s
        + 1041
    )
    D_oo = (
        9
        * (8 * r * s + 4 * r + 8 * s + 5)
        * (8 * r * s + 8 * r + 4 * s + 5)
        * (8 * r * s + 8 * r + 8 * s + 9)
    )

    P_ee = (
        3072 * r**3 * s**3
        + 2048 * r**3 * s**2
        + 2048 * r**2 * s**3
        + 3200 * r**2 * s**2
        + 768 * r**2 * s
        + 768 * r * s**2
        + 488 * r * s
        + 48 * r
        + 48 * s
        + 9
    )
    D_ee = (
        9
        * (8 * r * s + 1)
        * (8 * r * s + 4 * r + 1)
        * (8 * r * s + 4 * s + 1)
    )

    P_oe = (
        5120 * r**3 * s**3
        + 2048 * r**3 * s**2
        + 13312 * r**2 * s**3
        + 8064 * r**2 * s**2
        + 768 * r**2 * s
        + 11264 * r * s**3
        + 9216 * r * s**2
        + 1752 * r * s
        + 48 * r
        + 3072 * s**3
        + 3200 * s**2
        + 936 * s
        + 63
    )
    D_oe = (
        9
        * (8 * r * s + 4 * s + 1)
        * (8 * r * s + 8 * s + 1)
        * (8 * r * s + 4 * r + 8 * s + 5)
    )

    check(
        "R1 odd-odd polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    2 * r + 1,
                    2 * s + 1,
                    (True, False, True, True),
                    R1,
                )
                - P_oo / D_oo
            ),
            positive_coefficient_summary(P_oo, (r, s)),
            positive_coefficient_summary(D_oo, (r, s)),
        ),
        (
            0,
            (16, sp.Integer(1024), True),
            (16, sp.Integer(2025), True),
        ),
    )

    check(
        "R1 even-even polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    2 * r,
                    2 * s,
                    (False, True, True, True),
                    R1,
                )
                - P_ee / D_ee
            ),
            positive_coefficient_summary(P_ee, (r, s)),
            positive_coefficient_summary(
                D_ee.subs({r: u + 1, s: v + 1}, simultaneous=True),
                (u, v),
            ),
        ),
        (
            0,
            (10, sp.Integer(9), True),
            (16, sp.Integer(4608), True),
        ),
    )

    check(
        "R1 odd-even polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    2 * r + 1,
                    2 * s,
                    (True, True, False, True),
                    R1,
                )
                - P_oe / D_oe
            ),
            positive_coefficient_summary(P_oe, (r, s)),
            positive_coefficient_summary(D_oe.subs(s, v + 1), (r, v)),
        ),
        (
            0,
            (13, sp.Integer(48), True),
            (16, sp.Integer(2304), True),
        ),
    )


def r2_e(index):
    return 8 * index**2 + 8 * index + 3


def r2_o(index):
    return 2 * index + 1


def r2_E(index):
    return (2 * index + 1) * (16 * index**2 + 16 * index + 15)


def r2_O(index):
    return 16 * index**2 + 16 * index + 13


def R2(index, even):
    if even:
        return r2_E(index) / (5 * r2_e(index))
    return r2_O(index) / (10 * r2_o(index))


def r2_q_polynomials(U, V):
    Q_oo = (
        256 * U**5
        + U**4 * (4096 * V**2 + 4352 * V + 1920)
        + U**3
        * (20480 * V**3 + 41472 * V**2 + 22528 * V + 5200)
        + U**2
        * (
            36864 * V**4
            + 118272 * V**3
            + 116992 * V**2
            + 39728 * V
            + 5928
        )
        + U
        * (
            28672 * V**5
            + 128256 * V**4
            + 198656 * V**3
            + 122736 * V**2
            + 25392 * V
            + 2674
        )
        + 8192 * V**6
        + 47360 * V**5
        + 102272 * V**4
        + 99856 * V**3
        + 39752 * V**2
        + 3450 * V
        + 579
    )

    Q_ee = (
        U**3 * (4096 * V**3 + 5120 * V**2 + 2048 * V + 384)
        + U**2
        * (
            16384 * V**4
            + 37888 * V**3
            + 27136 * V**2
            + 7808 * V
            + 960
        )
        + U
        * (
            20480 * V**5
            + 71680 * V**4
            + 83456 * V**3
            + 39104 * V**2
            + 6624 * V
            + 360
        )
        + 8192 * V**6
        + 39168 * V**5
        + 66176 * V**4
        + 48656 * V**3
        + 13720 * V**2
        + 474 * V
        + 81
    )

    Q_oe = (
        V**6
        * (
            8192 * U**6
            + 77824 * U**5
            + 303104 * U**4
            + 618496 * U**3
            + 696320 * U**2
            + 409600 * U
            + 98304
        )
        + V**5
        * (
            20480 * U**6
            + 214784 * U**5
            + 916992 * U**4
            + 2038784 * U**3
            + 2487296 * U**2
            + 1576960 * U
            + 405504
        )
        + V**4
        * (
            16384 * U**6
            + 198656 * U**5
            + 966016 * U**4
            + 2417152 * U**3
            + 3285504 * U**2
            + 2299904 * U
            + 647168
        )
        + V**3
        * (
            4096 * U**6
            + 66560 * U**5
            + 406016 * U**4
            + 1224048 * U**3
            + 1949664 * U**2
            + 1565888 * U
            + 496640
        )
        + V**2
        * (
            5120 * U**5
            + 57856 * U**4
            + 250048 * U**3
            + 514984 * U**2
            + 503872 * U
            + 186528
        )
        + V
        * (
            2048 * U**4
            + 18048 * U**3
            + 56416 * U**2
            + 72822 * U
            + 32004
        )
        + 384 * U**3
        + 2496 * U**2
        + 4968 * U
        + 2727
    )
    return Q_oo, Q_ee, Q_oe


def verify_five_halves_argument():
    k = sp.symbols("k", integer=True, positive=True)
    previous_central_ratio = 2 * k / (2 * k - 1)
    next_central_ratio = (2 * k + 1) / (2 * k + 2)

    def three_even(index):
        return (32 * index**2 + 16 * index + 3) / 3

    def three_odd(index):
        return 4 * (2 * index + 1) * (4 * index + 3) / 3

    def five_even(index):
        return (
            (4 * index + 1)
            * (64 * index**2 + 32 * index + 15)
            / 15
        )

    def five_odd(index):
        return (
            2
            * (2 * index + 1)
            * (64 * index**2 + 96 * index + 45)
            / 15
        )

    check(
        "five-halves closed forms satisfy the recurrence and bases",
        (
            sp.factor(
                (2 * k + 1) * five_odd(k)
                - 6 * five_even(k)
                - 2 * k * five_odd(k - 1) * previous_central_ratio
            ),
            sp.factor(
                (2 * k + 2) * five_even(k + 1) * next_central_ratio
                - 6 * five_odd(k)
                - (2 * k + 1) * five_even(k)
            ),
            five_even(0) - 1,
            five_odd(0) - 6,
        ),
        (0, 0, 0, 0),
    )

    even_index = 2 * k
    odd_index = 2 * k + 1
    check(
        "R2 parity branches",
        (
            sp.factor(
                five_even(k) / three_even(k) - R2(even_index, True)
            ),
            sp.factor(five_odd(k) / three_odd(k) - R2(odd_index, False)),
        ),
        (0, 0),
    )

    p, q, n = sp.symbols("p q n", integer=True, positive=True)
    a, b, c, d = cell(p, q)

    def affine(index):
        return 2 * (2 * index + 1) / 5

    check(
        "R2 affine margin and denominator factors",
        (
            sp.factor(
                affine(a) * affine(b)
                - affine(c) * affine(d)
                - sp.Rational(8, 25)
            ),
            positive_coefficient_summary(r2_e(n), (n,)),
            positive_coefficient_summary(r2_o(n), (n,)),
        ),
        (
            0,
            (3, sp.Integer(3), True),
            (2, sp.Integer(1), True),
        ),
    )

    r, s = sp.symbols("r s", nonnegative=True)
    U, V = sp.symbols("U V", nonnegative=True)
    Q_oo, Q_ee, _ = r2_q_polynomials(U, V)

    p_oo = 2 * r + 1
    q_oo = 2 * s + 1
    a, b, c, d = cell(p_oo, q_oo)
    Pi_oo = r2_o(b) * r2_e(a) * r2_e(c) * r2_e(d)
    check(
        "R2 odd-odd Q polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    p_oo,
                    q_oo,
                    (True, False, True, True),
                    R2,
                )
                - sp.Rational(8, 25)
                - 9
                * Q_oo.subs({U: p_oo + q_oo, V: p_oo * q_oo})
                / (50 * Pi_oo)
            ),
            positive_coefficient_summary(Q_oo, (U, V)),
        ),
        (0, (26, sp.Integer(256), True)),
    )

    p_ee = 2 * r
    q_ee = 2 * s
    a, b, c, d = cell(p_ee, q_ee)
    Pi_ee = r2_o(a) * r2_e(b) * r2_e(c) * r2_e(d)
    check(
        "R2 even-even Q polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    p_ee,
                    q_ee,
                    (False, True, True, True),
                    R2,
                )
                - sp.Rational(8, 25)
                - 9
                * Q_ee.subs({U: p_ee + q_ee, V: p_ee * q_ee})
                / (50 * Pi_ee)
            ),
            positive_coefficient_summary(Q_ee, (U, V)),
        ),
        (0, (22, sp.Integer(81), True)),
    )

    u, v = sp.symbols("u v", nonnegative=True)
    p_oe = u + 1
    q_oe = v
    a, b, c, d = cell(p_oe, q_oe)
    Pi_oe = r2_o(c) * r2_e(a) * r2_e(b) * r2_e(d)
    Q_oe = r2_q_polynomials(u, v)[2]
    check(
        "R2 odd-even Q polynomial identity and positivity",
        (
            sp.factor(
                quotient_minor(
                    p_oe,
                    q_oe,
                    (True, True, False, True),
                    R2,
                )
                - sp.Rational(8, 25)
                - 9 * Q_oe / (50 * Pi_oe)
            ),
            positive_coefficient_summary(Q_oe, (u, v)),
        ),
        (0, (43, sp.Integer(384), True)),
    )


def endpoint_cell_ratio(p, q):
    a, b, c, d = cell(p, q)
    half = Fraction(1, 2)
    return (
        T_rat(half, a)
        * T_rat(half, b)
        / (T_rat(half, c) * T_rat(half, d))
    )


def verify_qsharp_values():
    expected = {
        7: Fraction(218009, 63339654),
        8: Fraction(-107, 9088200),
        9: Fraction(4176826, 1906449531),
        10: Fraction(-55171, 3139181892),
        11: Fraction(3568693, 2353538250),
        12: Fraction(-20509, 1188820360),
    }
    for p, value in expected.items():
        check(
            f"endpoint E_{{{p},1}} - E_{{{p},3}}",
            endpoint_cell_ratio(p, 1) - endpoint_cell_ratio(p, 3),
            value,
        )


verify_endpoint_argument()
verify_three_halves_argument()
verify_five_halves_argument()
verify_qsharp_values()

# ------------------------------------------------- the scalar transfer system
# c_n = T(x,n) - T(x,n-1), u_n = n c_n, z_n = u_n / u_{n-1}.
def c_seq(xv: Fraction, N: int) -> list[Fraction]:
    return [T_rat(xv, n) - T_rat(xv, n - 1) for n in range(1, N)]


def z_seq(xv: Fraction, N: int) -> dict[int, Fraction]:
    h = 2 * xv
    z = {2: h}
    for n in range(2, N):
        z[n + 1] = 1 / z[n] + h / n
    return z


DIMENSIONS = (Fraction(1, 2), Fraction(3, 5), Fraction(1), Fraction(7, 4),
              Fraction(3))

for xv in DIMENSIONS:
    h = 2 * xv
    c = c_seq(xv, 26)
    z = z_seq(xv, 30)
    check(
        f"x={xv}: (n+1)c_(n+1) = h c_n + (n-1)c_(n-1)",
        all(
            (n + 1) * c[n] == h * c[n - 1] + (n - 1) * c[n - 2]
            for n in range(2, 24)
        ),
        True,
    )
    check(
        f"x={xv}: h T(x,n) = u_n + u_(n+1)",
        all(
            h * T_rat(xv, n) == n * c[n - 1] + (n + 1) * c[n]
            for n in range(1, 24)
        ),
        True,
    )
    check(
        f"x={xv}: T_n/T_(n-1) = z_n (1+z_(n+1))/(1+z_n)",
        all(
            T_rat(xv, n) / T_rat(xv, n - 1)
            == z[n] * (1 + z[n + 1]) / (1 + z[n])
            for n in range(2, 25)
        ),
        True,
    )
    check(
        f"x={xv}: two-step map z_(n+2) = z_n/(1+(h/n)z_n) + h/(n+1)",
        all(
            z[n + 2] == z[n] / (1 + h * z[n] / n) + h / (n + 1)
            for n in range(2, 25)
        ),
        True,
    )

# The two exact orbits bracketing the first open strip.
z_endpoint = z_seq(Fraction(1, 2), 60)
check(
    "h=1 orbit: z_(2m)=1 and z_(2m+1)=(2m+1)/(2m)",
    all(
        z_endpoint[2 * m] == 1
        and z_endpoint[2 * m + 1] == Fraction(2 * m + 1, 2 * m)
        for m in range(1, 25)
    ),
    True,
)
check(
    "h=1 parity swing is exactly 1/(n-1), i.e. order n^-h with h=1",
    all(
        z_endpoint[2 * m + 1] - z_endpoint[2 * m] == Fraction(1, 2 * m)
        for m in range(1, 25)
    ),
    True,
)
check(
    "h=2 orbit: c_n = 2 for every n>=1",
    all(T_rat(Fraction(1), n) - T_rat(Fraction(1), n - 1) == 2
        for n in range(1, 25)),
    True,
)
z_two = z_seq(Fraction(1), 40)
check(
    "h=2 orbit: z_n = n/(n-1), no parity oscillation",
    all(z_two[n] == Fraction(n, n - 1) for n in range(2, 35)),
    True,
)

# The window form of the target.
def window(z: dict[int, Fraction], p: int, q: int) -> Fraction:
    a, b, c_, d = cell(p, q)
    value = Fraction(1)
    for j in range(c_ + 1, a + 1):
        value *= z[j]
    for j in range(b + 1, d + 1):
        value /= z[j]
    return value * (1 + z[a + 1]) * (1 + z[b + 1]) / (
        (1 + z[c_ + 1]) * (1 + z[d + 1])
    )


for xv in DIMENSIONS:
    z = z_seq(xv, 90)
    check(
        f"x={xv}: window product equals T_aT_b/T_cT_d",
        all(
            window(z, p, q)
            == T_rat(xv, cell(p, q)[0]) * T_rat(xv, cell(p, q)[1])
            / (T_rat(xv, cell(p, q)[2]) * T_rat(xv, cell(p, q)[3]))
            for p in range(1, 5)
            for q in range(1, 5)
        ),
        True,
    )

# --------------------------------------------- the Green-kernel trace is odd H
def tau(n: int) -> Fraction:
    return sum(Fraction(1, 2 * j + 1) for j in range((n + 1) // 2))


for n in range(1, 10):
    Bn = (sp.eye(n) - K(n) * K(n)).inv()
    check(
        f"tr B_{n} = sum_(r<ceil({n}/2)) 1/(2r+1)",
        sp.nsimplify(sp.trace(Bn)),
        sp.Rational(tau(n).numerator, tau(n).denominator),
    )
a_, b_, c_, d_ = cell(1, 2)
check(
    "tau_6 + tau_2 - tau_3 - tau_4 = -2/15",
    tau(a_) + tau(b_) - tau(c_) - tau(d_),
    Fraction(-2, 15),
)
check(
    "and ab times that is the [y] coefficient -8/5",
    a_ * b_ * (tau(a_) + tau(b_) - tau(c_) - tau(d_)),
    Fraction(-8, 5),
)

# ------------------------------------- the eight-index symmetric-function law
pp, qq, ss = sp.symbols("pp qq ss", positive=True)


def cell_sym(P, Q):
    return (P + 1) * (Q + 1), P * Q, P * (Q + 1), Q * (P + 1)


a1, b1, c1, d1 = cell_sym(pp, qq)
a3, b3, c3, d3 = cell_sym(pp, qq + 2)
LEFT = [a1, b1, c3, d3]
RIGHT = [c1, d1, a3, b3]
check("e_1(L) = e_1(R)",
      sp.expand(sp.symmetric_poly(1, LEFT) - sp.symmetric_poly(1, RIGHT)), 0)
check("e_4(L) = e_4(R)",
      sp.expand(sp.symmetric_poly(4, LEFT) - sp.symmetric_poly(4, RIGHT)), 0)
check("e_2(L) - e_2(R) = 2(2p+1)",
      sp.expand(sp.symmetric_poly(2, LEFT) - sp.symmetric_poly(2, RIGHT)
                - 2 * (2 * pp + 1)), 0)
check("e_3(L) - e_3(R) = 2p(p+1)(2q+3)",
      sp.expand(sp.symmetric_poly(3, LEFT) - sp.symmetric_poly(3, RIGHT)
                - 2 * pp * (pp + 1) * (2 * qq + 3)), 0)
check(
    "prod(s+l) - prod(s+r) = 2s((2p+1)s + p(p+1)(2q+3))",
    sp.expand(
        sp.prod([ss + v for v in LEFT]) - sp.prod([ss + v for v in RIGHT])
        - 2 * ss * ((2 * pp + 1) * ss + pp * (pp + 1) * (2 * qq + 3))
    ),
    0,
)

# ------------------------------------------------ the closure lemma (W) census
def W_margin(xv: Fraction, p: int, q: int) -> Fraction:
    a1_, b1_, c1_, d1_ = cell(p, q)
    a2_, b2_, c2_, d2_ = cell(p, q + 2)
    return (
        T_rat(xv, a1_) * T_rat(xv, b1_) * T_rat(xv, c2_) * T_rat(xv, d2_)
        - T_rat(xv, c1_) * T_rat(xv, d1_) * T_rat(xv, a2_) * T_rat(xv, b2_)
    )


W_GRID = (Fraction(1, 2), Fraction(11, 20), Fraction(3, 5), Fraction(3, 4),
          Fraction(9, 10), Fraction(1), Fraction(11, 10), Fraction(3, 2),
          Fraction(2), Fraction(5, 2), Fraction(7, 3), Fraction(4),
          Fraction(15, 2), Fraction(20))
w_failures = [
    (xv, p, q)
    for xv in W_GRID
    for p in range(1, 9)
    for q in range(2, 9)
    if W_margin(xv, p, q) <= 0
]
check(f"(W) holds on all {len(W_GRID) * 8 * 7} cells with q>=2", w_failures, [])

# The q>=2 restriction is sharp: q=1 fails, and only at the endpoint.
check(
    "(W) FAILS at q=1, x=1/2, p=8, 10, 12 (mixed parity)",
    [p for p in (8, 10, 12) if W_margin(Fraction(1, 2), p, 1) < 0],
    [8, 10, 12],
)
check(
    "(W) holds at q=1, x=1/2, for odd p=7,9,11 (same parity)",
    [p for p in (7, 9, 11) if W_margin(Fraction(1, 2), p, 1) > 0],
    [7, 9, 11],
)
check(
    "the q=1 failure is an endpoint phenomenon: p=8 is positive off x=1/2",
    all(
        W_margin(xv, 8, 1) > 0
        for xv in (Fraction(3, 5), Fraction(3, 4), Fraction(1),
                   Fraction(3, 2), Fraction(2))
    ),
    True,
)

# -------------------------------------- universal dimension-step sanity grid
def dimension_quotient(xv: Fraction, n: int) -> Fraction:
    return T_rat(xv + 1, n) / T_rat(xv, n)


DIMENSION_STEP_GRID = (
    Fraction(1, 2), Fraction(2, 3), Fraction(1),
    Fraction(11, 10), Fraction(3, 2), Fraction(5),
)
check(
    "universal dimension step on the exact regression grid",
    all(
        dimension_quotient(xv, cell(p, q)[0])
        * dimension_quotient(xv, cell(p, q)[1])
        > dimension_quotient(xv, cell(p, q)[2])
        * dimension_quotient(xv, cell(p, q)[3])
        for xv in DIMENSION_STEP_GRID
        for p in range(1, 9)
        for q in range(1, 9)
    ),
    True,
)

# ------------------------- closure-anchor interpolation route exact falsifier
w_symbol = sp.symbols("w_symbol")
p_symbol = sp.symbols("p_symbol")
p_values = [sp.Poly(1, p_symbol), sp.Poly(p_symbol, p_symbol)]
for n in range(1, 24):
    p_values.append(
        sp.Poly(
            p_symbol * p_values[n].as_expr()
            + n**2 * p_values[n - 1].as_expr(),
            p_symbol,
        )
    )

normalized_rows = []
for n, polynomial in enumerate(p_values):
    expression = polynomial.as_expr()
    if n % 2:
        expression /= p_symbol
    expression = sp.Poly(sp.expand(expression), p_symbol)
    normalized_rows.append(
        sp.expand(
            sum(
                coefficient * (w_symbol + 4) ** (degree // 2)
                for (degree,), coefficient in expression.terms()
            )
            / sp.factorial(n)
        )
    )

side_left = sp.prod(normalized_rows[index] for index in (16, 9, 18, 20))
side_right = sp.prod(normalized_rows[index] for index in (12, 12, 24, 15))
side_derivative = sp.factor(
    (
        sp.diff(side_left, w_symbol) / side_left
        - sp.diff(side_right, w_symbol) / side_right
    ).subs(w_symbol, 0)
)
check(
    "W side-ratio derivative at (p,q,w)=(3,3,0)",
    side_derivative,
    sp.Rational(
        -15_735_955_899_430_519_108,
        54_938_841_295_981_420_122_345,
    ),
)

# --------------------------------------- transfer envelope and exact decay law
s_symbol = sp.symbols("s_symbol", positive=True)
n_symbol = sp.symbols("n_symbol", integer=True, positive=True)
xi = sp.symbols("xi", positive=True)
alpha_symbol = sp.symbols("alpha_symbol", positive=True)
h_symbol = sp.symbols("h_symbol", positive=True)

lower_weight = ((1 + s_symbol) / (1 - s_symbol)) ** xi
lower_boundary = s_symbol ** (n_symbol - 1) * (1 - s_symbol**2) * lower_weight
lower_derivative = s_symbol ** (n_symbol - 2) * lower_weight * (
    n_symbol - 1 + 2 * xi * s_symbol - (n_symbol + 1) * s_symbol**2
)
check(
    "lower transfer integration-by-parts identity",
    sp.simplify(sp.diff(lower_boundary, s_symbol) - lower_derivative),
    0,
)

upper_weight = ((1 - s_symbol) / (1 + s_symbol)) ** alpha_symbol
upper_boundary = (
    s_symbol ** (n_symbol - 1) * (1 - s_symbol) * upper_weight
)
upper_derivative = upper_weight * (
    (n_symbol - 1) * s_symbol ** (n_symbol - 2)
    - n_symbol * s_symbol ** (n_symbol - 1)
    - 2
    * alpha_symbol
    * s_symbol ** (n_symbol - 1)
    / (1 + s_symbol)
)
check(
    "upper transfer integration-by-parts identity",
    sp.simplify(sp.diff(upper_boundary, s_symbol) - upper_derivative),
    0,
)

beta_symbol = (
    sp.gamma(xi + 1)
    / sp.gamma(1 - xi)
    * sp.gamma(n_symbol + 1 - xi)
    / sp.gamma(n_symbol + 1 + xi)
)
check(
    "beta_n/beta_(n-1) = (n-x)/(n+x)",
    sp.simplify(
        sp.expand_func(beta_symbol / beta_symbol.subs(n_symbol, n_symbol - 1))
        - (n_symbol - xi) / (n_symbol + xi)
    ),
    0,
)
check(
    "beta_(n-1)(1/2) = 1/(2n-1)",
    sp.simplify(
        sp.expand_func(
            beta_symbol.subs(
                {n_symbol: n_symbol - 1, xi: sp.Rational(1, 2)}
            )
        )
        - 1 / (2 * n_symbol - 1)
    ),
    0,
)

gamma_symbol = (
    (xi - 1)
    / (2 * n_symbol)
    * sp.gamma(xi + 1)
    / sp.gamma(2 - xi)
    * sp.gamma(n_symbol + 2 - xi)
    / sp.gamma(n_symbol + xi + 1)
)
check(
    "gamma_n/gamma_(n-1) product ratio",
    sp.simplify(
        sp.expand_func(
            gamma_symbol / gamma_symbol.subs(n_symbol, n_symbol - 1)
        )
        - (n_symbol - 1)
        / n_symbol
        * (n_symbol + 1 - xi)
        / (n_symbol + xi)
    ),
    0,
)

lower_b_constant = (
    2 ** (-2 * xi) * sp.gamma(xi + 1) / sp.gamma(1 - xi)
)
lower_e_constant = (
    2 ** (1 - 2 * xi) * sp.gamma(xi + 1) / sp.gamma(1 - xi)
)
check(
    "two consecutive lower ratios give the exact transfer constant",
    sp.simplify(2 * lower_b_constant - lower_e_constant),
    0,
)
check(
    "marginal transfer half-amplitude is 1/2",
    sp.simplify(lower_e_constant.subs(xi, sp.Rational(1, 2))),
    sp.Rational(1, 2),
)
upper_b_constant = (
    (xi - 1)
    * 2 ** (-2 * xi)
    * sp.gamma(xi + 1)
    / sp.gamma(2 - xi)
)
upper_e_constant = (
    (xi - 1)
    * 2 ** (1 - 2 * xi)
    * sp.gamma(xi + 1)
    / sp.gamma(2 - xi)
)
check(
    "two consecutive upper ratios give the exact transfer constant",
    sp.simplify(2 * upper_b_constant - upper_e_constant),
    0,
)

lower_x_envelope = (
    sp.Rational(3, 2)
    * (1 + xi) ** (2 * xi)
    * (1 + 2 ** (2 * xi))
)
lower_h_envelope = (
    sp.Rational(3, 2)
    * (1 + h_symbol / 2) ** h_symbol
    * (1 + 2**h_symbol)
)
check(
    "C_-(h) substitution",
    sp.simplify(
        lower_x_envelope.subs(xi, h_symbol / 2) - lower_h_envelope
    ),
    0,
)
upper_x_envelope = (
    sp.Rational(8, 7)
    * (xi - 1)
    / 2
    * (1 + xi) ** (2 * xi - 1)
    * (1 + sp.Rational(3, 2) ** (2 * xi))
)
upper_h_envelope = (
    2
    * (h_symbol - 2)
    / 7
    * (1 + h_symbol / 2) ** (h_symbol - 1)
    * (1 + sp.Rational(3, 2) ** h_symbol)
)
check(
    "C_+(h) substitution",
    sp.simplify(
        upper_x_envelope.subs(xi, h_symbol / 2) - upper_h_envelope
    ),
    0,
)

t_constant = (
    2 ** (-2 * xi - 1)
    * sp.gamma(xi + 1) ** 2
    * sp.sin(sp.pi * xi)
    / sp.pi
)
t_constant_beta = (
    xi
    * 2 ** (-2 * xi - 1)
    * sp.gamma(xi + 1)
    / sp.gamma(1 - xi)
)
reflection_substitution = (
    sp.pi * xi / (sp.gamma(xi + 1) * sp.gamma(1 - xi))
)
check(
    "T-coordinate decay constant agrees by reflection",
    sp.simplify(
        (t_constant - t_constant_beta).subs(
            sp.sin(sp.pi * xi), reflection_substitution
        )
    ),
    0,
)

# ------------------------------------------ full first closure coefficient
def poly_add(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    degree = max(len(left), len(right))
    return [
        (left[index] if index < len(left) else Fraction(0))
        + (right[index] if index < len(right) else Fraction(0))
        for index in range(degree)
    ]


def poly_scale(values: list[Fraction], factor: Fraction) -> list[Fraction]:
    return [factor * value for value in values]


def poly_multiply(
    left: list[Fraction], right: list[Fraction]
) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def poly_product(rows: list[list[Fraction]]) -> list[Fraction]:
    result = [Fraction(1)]
    for row in rows:
        result = poly_multiply(result, row)
    return result


def divide_by_u_minus_one(values: list[Fraction]) -> list[Fraction]:
    quotient = []
    for index in range(len(values) - 1):
        quotient.append(
            -values[0] if index == 0 else quotient[-1] - values[index]
        )
    check(
        "division by u-1 reconstructs the closure polynomial",
        poly_multiply([Fraction(-1), Fraction(1)], quotient),
        values,
    )
    return quotient


def odd_cycle_rows(maximum_n: int) -> list[list[Fraction]]:
    maximum_m = maximum_n // 2 + 1
    even = [[Fraction(1)]]
    odd = [[Fraction(1)]]
    for m in range(maximum_m):
        even.append(
            poly_scale(
                poly_add(
                    [Fraction(0)] + odd[m],
                    poly_scale(even[m], Fraction(2 * m + 1)),
                ),
                Fraction(1, 2 * m + 2),
            )
        )
        odd.append(
            poly_scale(
                poly_add(even[m + 1], poly_scale(odd[m], Fraction(2 * m + 2))),
                Fraction(1, 2 * m + 3),
            )
        )
    return [
        even[n // 2] if n % 2 == 0 else odd[n // 2]
        for n in range(maximum_n + 1)
    ]


def closure_corners(p: int, q: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        (
            (p + 1) * (q + 1),
            p * q,
            p * (q + 3),
            (p + 1) * (q + 2),
        ),
        (
            p * (q + 1),
            (p + 1) * q,
            (p + 1) * (q + 3),
            p * (q + 2),
        ),
    )


def zeta_cycle(n: int) -> Fraction:
    m = n // 2
    if n % 2 == 0:
        return Fraction(1, 2) * sum(
            (Fraction(1, 2 * index + 1) for index in range(m)),
            Fraction(0),
        )
    return Fraction(1, 2) * sum(
        (Fraction(1, 2 * index + 3) for index in range(m)),
        Fraction(0),
    )


def kappa_cycle(r: int, q: int) -> Fraction:
    return (
        zeta_cycle(r * (q + 1))
        + zeta_cycle(r * (q + 2))
        - zeta_cycle(r * q)
        - zeta_cycle(r * (q + 3))
    )


def delta_one_cycle(p: int, q: int) -> Fraction:
    left, right = closure_corners(p, q)
    return sum((zeta_cycle(n) for n in left), Fraction(0)) - sum(
        (zeta_cycle(n) for n in right), Fraction(0)
    )


cycle_rows = odd_cycle_rows(45)
for n, row in enumerate(cycle_rows):
    check(f"odd-cycle Q_{n}(1)=1", sum(row, Fraction(0)), Fraction(1))
    check(
        f"odd-cycle Q_{n}'(1)=zeta_{n}",
        sum(
            (Fraction(degree) * value for degree, value in enumerate(row)),
            Fraction(0),
        ),
        zeta_cycle(n),
    )

delta_controls = {
    (1, 3): Fraction(2, 77),
    (2, 2): Fraction(44, 4095),
    (3, 3): Fraction(59_903, 5_870_865),
    (1, 2): Fraction(-1, 45),
    (2, 3): Fraction(-13, 1309),
}
for (p, q), expected in delta_controls.items():
    check(f"delta_1({p},{q})", delta_one_cycle(p, q), expected)
    check(
        f"delta_1({p},{q}) = kappa_(p+1)-kappa_p",
        delta_one_cycle(p, q),
        kappa_cycle(p + 1, q) - kappa_cycle(p, q),
    )

p_mixed, q_mixed = sp.symbols(
    "p_mixed q_mixed", integer=True, positive=True
)
check(
    "first mixed-parity Cauchy margin",
    sp.factor(
        q_mixed / (q_mixed + 1)
        + q_mixed * p_mixed / (p_mixed * (q_mixed + 2) + 1)
        - 1
    ),
    (
        p_mixed * (q_mixed**2 - 2) - 1
    )
    / (
        (q_mixed + 1)
        * (p_mixed * (q_mixed + 2) + 1)
    ),
)
check(
    "second mixed-parity Cauchy margin",
    1 + (q_mixed + 1) / (q_mixed + 2) - 1,
    (q_mixed + 1) / (q_mixed + 2),
)

same_parity_cells = [
    (p, q)
    for p in range(1, 15)
    for q in range(2, 17)
    if p % 2 == q % 2
]
check("same-parity delta_1 exact census has 105 cells", len(same_parity_cells), 105)
check(
    "same-parity delta_1 is positive on the exact census",
    all(delta_one_cycle(p, q) > 0 for p, q in same_parity_cells),
    True,
)

mixed_parity_cells = [
    (p, q)
    for p in range(1, 15)
    for q in range(2, 17)
    if p % 2 != q % 2
]
check("mixed-parity delta_1 exact census has 105 cells",
      len(mixed_parity_cells), 105)
check(
    "mixed-parity delta_1 is negative on the exact census",
    all(delta_one_cycle(p, q) < 0 for p, q in mixed_parity_cells),
    True,
)


def int_poly_multiply(left: list[int], right: list[int]) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def int_poly_product(rows: list[list[int]]) -> list[int]:
    result = [1]
    for row in rows:
        result = int_poly_multiply(result, row)
    return result


def int_poly_shift(values: list[int], offset: int) -> list[int]:
    result = [0] * len(values)
    for degree, value in enumerate(values):
        for power in range(degree + 1):
            result[power] += (
                value * math.comb(degree, power) * offset ** (degree - power)
            )
    return result


def int_poly_derivative(values: list[int]) -> list[int]:
    return [degree * values[degree] for degree in range(1, len(values))]


def int_poly_subtract(left: list[int], right: list[int]) -> list[int]:
    length = max(len(left), len(right))
    result = [
        (left[index] if index < len(left) else 0)
        - (right[index] if index < len(right) else 0)
        for index in range(length)
    ]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def integer_cycle_rows(maximum_n: int) -> list[list[int]]:
    rows = [[1], [1]]
    for n in range(1, maximum_n):
        transported = [0] + rows[n] if n % 2 else rows[n]
        inherited = [n * n * value for value in rows[n - 1]]
        degree = max(len(transported), len(inherited))
        rows.append(
            [
                (transported[index] if index < len(transported) else 0)
                + (inherited[index] if index < len(inherited) else 0)
                for index in range(degree)
            ]
        )
    return rows


integer_rows = integer_cycle_rows(285)
factorials = [math.factorial(n) for n in range(286)]
check(
    "integer odd-cycle rows satisfy Q_n(1)=1 through n=285",
    all(sum(row) == factorials[n] for n, row in enumerate(integer_rows)),
    True,
)

raw_digest = hashlib.sha256()
raw_cells = 0
raw_coefficients = 0
raw_bad_cells: list[tuple[int, int]] = []
raw_division_bad: list[tuple[int, int]] = []
raw_controls: dict[tuple[int, int], tuple[Fraction, Fraction]] = {}
for p, q in same_parity_cells:
    left_indices, right_indices = closure_corners(p, q)
    left_integer = int_poly_product(
        [integer_rows[index] for index in left_indices]
    )
    right_integer = int_poly_product(
        [integer_rows[index] for index in right_indices]
    )
    left_denominator = math.prod(factorials[index] for index in left_indices)
    right_denominator = math.prod(
        factorials[index] for index in right_indices
    )
    denominator_gcd = math.gcd(left_denominator, right_denominator)
    left_scale = right_denominator // denominator_gcd
    right_scale = left_denominator // denominator_gcd
    degree = max(len(left_integer), len(right_integer))
    cleared = [
        left_scale
        * (left_integer[index] if index < len(left_integer) else 0)
        - right_scale
        * (right_integer[index] if index < len(right_integer) else 0)
        for index in range(degree)
    ]
    running = 0
    quotient: list[int] = []
    for value in cleared[:-1]:
        running -= value
        quotient.append(running)
    if running != cleared[-1]:
        raw_division_bad.append((p, q))
    if (
        len(quotient) != 2 * p * q + 3 * p + q + 1
        or any(value <= 0 for value in quotient)
    ):
        raw_bad_cells.append((p, q))

    common_denominator = (
        left_denominator * right_denominator // denominator_gcd
    )
    if (p, q) == (1, 3):
        raw_controls[(p, q)] = (
            Fraction(quotient[0], common_denominator),
            Fraction(quotient[-1], common_denominator),
        )

    raw_digest.update(f"{p},{q},{len(quotient)},".encode("ascii"))
    for value in quotient:
        raw_digest.update(str(value).encode("ascii"))
        raw_digest.update(b",")
    raw_digest.update(b"\n")
    raw_cells += 1
    raw_coefficients += len(quotient)

check("raw survival census has 105 cells", raw_cells, 105)
check("raw survival census exact divisions", raw_division_bad, [])
check("raw survival census has no nonpositive coefficient", raw_bad_cells, [])
check("raw survival census coefficient count", raw_coefficients, 17_661)
check(
    "raw survival controls at (1,3)",
    raw_controls[(1, 3)],
    (
        Fraction(273, 131_072),
        Fraction(1, 1_738_201_006_080_000),
    ),
)
check(
    "raw survival canonical digest",
    raw_digest.hexdigest(),
    "1860046c7348673dedef46ffbac49cc454c147f2f20483a7c1221cf046d47806",
)


def fraction_poly_evaluate(values: list[Fraction], argument: Fraction) -> Fraction:
    result = Fraction(0)
    for value in reversed(values):
        result = result * argument + value
    return result


def scale_ratio(r: int, q: int, u_value: int) -> Fraction:
    return (
        fraction_poly_evaluate(cycle_rows[r * (q + 1)], Fraction(u_value))
        * fraction_poly_evaluate(cycle_rows[r * (q + 2)], Fraction(u_value))
        / (
            fraction_poly_evaluate(cycle_rows[r * q], Fraction(u_value))
            * fraction_poly_evaluate(
                cycle_rows[r * (q + 3)], Fraction(u_value)
            )
        )
    )


check(
    "scale quotient fence below u=4",
    scale_ratio(2, 2, 2) - scale_ratio(1, 2, 2),
    Fraction(-425_373_584, 59_875_414_893),
)
check(
    "scale quotient positive control at u=4",
    scale_ratio(2, 2, 4) - scale_ratio(1, 2, 4),
    Fraction(344, 127_575),
)
check(
    "scale quotient q=1 fence at u=4",
    scale_ratio(9, 1, 4) - scale_ratio(8, 1, 4),
    Fraction(-2_782, 195_240_325),
)


def raw_tail_polynomial(p: int, q: int) -> list[Fraction]:
    left, right = closure_corners(p, q)
    return divide_by_u_minus_one(
        poly_add(
            poly_product([cycle_rows[n] for n in left]),
            poly_scale(
                poly_product([cycle_rows[n] for n in right]),
                Fraction(-1),
            ),
        )
    )


lower_atom = divide_by_u_minus_one(
    poly_add(
        poly_multiply(cycle_rows[4], cycle_rows[5]),
        poly_scale(poly_multiply(cycle_rows[3], cycle_rows[6]), Fraction(-1)),
    )
)
check(
    "lower two-block atom at (1,3)",
    lower_atom,
    [
        Fraction(-17, 960),
        Fraction(139, 8640),
        Fraction(13, 8640),
        Fraction(1, 8640),
    ],
)
check("[u^4] G at out-of-range (1,1)", raw_tail_polynomial(1, 1)[4],
      Fraction(-1, 24_192))

def shift_u_to_y(row: list[Fraction]) -> list[Fraction]:
    shifted = [Fraction(0)]
    for degree, coefficient in enumerate(row):
        shifted = poly_add(
            shifted,
            poly_scale(
                [Fraction(sp.binomial(degree, power))
                 for power in range(degree + 1)],
                coefficient,
            ),
        )
    return shifted


def affine_fraction_shift(
    row: list[Fraction], offset: int, slope: int
) -> list[Fraction]:
    """Return coefficients of P(offset+slope*z)."""

    shifted = [Fraction(0)] * len(row)
    for degree, coefficient in enumerate(row):
        for power in range(degree + 1):
            shifted[power] += (
                coefficient
                * math.comb(degree, power)
                * offset ** (degree - power)
                * slope**power
            )
    return shifted


left_22, right_22 = closure_corners(2, 2)
left_y = poly_product([shift_u_to_y(cycle_rows[n]) for n in left_22])
right_y = poly_product([shift_u_to_y(cycle_rows[n]) for n in right_22])
check("delta_2(2,2)", left_y[2] - right_y[2],
      Fraction(558_067, 37_837_800))
check(
    "second log derivative at (2,2)",
    2 * left_y[2] - left_y[1] ** 2
    - 2 * right_y[2] + right_y[1] ** 2,
    Fraction(-4_804_463, 147_567_420),
)


# ------------------------------------ second same-parity threshold tail U_2
def shifted_numerator_certificate(expression, substitutions, variables):
    numerator = sp.cancel(expression).as_numer_denom()[0]
    polynomial = sp.Poly(
        sp.expand(numerator.subs(substitutions, simultaneous=True)),
        *variables,
        domain=sp.QQ,
    )
    payload = "".join(
        f"{','.join(map(str, powers))}:{coefficient}\n"
        for powers, coefficient in polynomial.terms()
    ).encode("ascii")
    return (
        (
            len(polynomial.coeffs()),
            min(polynomial.coeffs()),
            all(value > 0 for value in polynomial.coeffs()),
        ),
        hashlib.sha256(payload).hexdigest(),
    )


tail_p, tail_q = sp.symbols("tail_p tail_q", positive=True)
tail_u, tail_v = sp.symbols("tail_u tail_v", nonnegative=True)

tail_i_q = (
    1 / tail_q**2
    - 1 / (tail_q + 1) ** 2
    - 1 / (tail_q + 2) ** 2
    + 1 / (tail_q + 3) ** 2
)
tail_i_simple = 12 / (
    (tail_q - 1) * tail_q * (tail_q + 1) * (tail_q + 2)
)
check(
    "same-parity U_2 logarithmic beta bound",
    shifted_numerator_certificate(
        tail_i_simple - tail_i_q,
        {tail_q: tail_v + 2},
        (tail_v,),
    ),
    (
        (5, sp.Integer(48), True),
        "a9be8dd2b2e49085f993c246f0d2f61bd38f3a0774e83863a6bcb23e6f3dfc9b",
    ),
)


def first_moment_j_upper(tail_r, tail_a):
    tail_x = tail_a + 1
    return sp.Rational(1, 2) * (
        1 / tail_x**2
        + sp.Rational(1, 2) / tail_x
        - sp.Rational(1, 2) / (tail_x + 2 * (tail_r - 1))
    )


tail_delta_upper = 1 / (2 * tail_p * tail_q * (tail_q + 1))
tail_smooth_upper = 1 / (
    2
    * tail_p**2
    * (tail_q - 1)
    * tail_q
    * (tail_q + 1)
    * (tail_q + 2)
)
tail_odd_delta_margin = sp.factor(
    tail_delta_upper
    - tail_smooth_upper
    - first_moment_j_upper(tail_p, tail_p * tail_q)
)
tail_even_delta_margin = sp.factor(
    tail_delta_upper
    - tail_smooth_upper
    - first_moment_j_upper(
        tail_p + 1, (tail_p + 1) * (tail_q + 1)
    )
)
check(
    "same-parity U_2 odd first-moment upper certificate",
    shifted_numerator_certificate(
        tail_odd_delta_margin,
        {tail_p: tail_u + 1, tail_q: tail_v + 3},
        (tail_u, tail_v),
    ),
    (
        (25, sp.Integer(1), True),
        "55f9183f6ea1a8c83bb9a8a115fe092527c040654ffc2205a8956138d0ecabcb",
    ),
)
check(
    "same-parity U_2 even first-moment upper certificate",
    shifted_numerator_certificate(
        tail_even_delta_margin,
        {tail_p: tail_u + 2, tail_q: tail_v + 2},
        (tail_u, tail_v),
    ),
    (
        (29, sp.Integer(1), True),
        "f077648cb997e07a3c3eac514d738a17b6edce3ba083c52d77efcf9afe6f6bc8",
    ),
)


def endpoint_smooth_ratio(tail_q_value):
    return (
        (2 * (tail_p + 1) * (tail_q_value + 1) + 1)
        * (2 * tail_p * tail_q_value + 1)
        / (
            (2 * tail_p * (tail_q_value + 1) + 1)
            * (2 * (tail_p + 1) * tail_q_value + 1)
        )
    )


tail_n = tail_p * tail_q + 1
tail_exp_eta = (
    1 - sp.Rational(1, 4) / (tail_n + 2 * tail_p) ** 2
) / (1 - sp.Rational(1, 4) / tail_n**2)
tail_x_ratio = sp.factor(
    endpoint_smooth_ratio(tail_q)
    / endpoint_smooth_ratio(tail_q + 2)
    / tail_exp_eta
)
tail_scale = tail_p * (tail_p + 1) * tail_q * (tail_q + 1)
tail_x_margin = sp.factor(
    (tail_x_ratio - 1) / tail_x_ratio
    - (tail_p + 1) / tail_scale**2
)
check(
    "same-parity U_2 endpoint X certificate, odd bulk",
    shifted_numerator_certificate(
        tail_x_margin,
        {tail_p: tail_u + 3, tail_q: tail_v + 3},
        (tail_u, tail_v),
    ),
    (
        (100, sp.Integer(64), True),
        "ebf3b80e5e75419b3d2f43d627fc1d341302d1f8b54c98c8f0a66a9965d816fe",
    ),
)
check(
    "same-parity U_2 endpoint X certificate, even bulk",
    shifted_numerator_certificate(
        tail_x_margin,
        {tail_p: tail_u + 4, tail_q: tail_v + 2},
        (tail_u, tail_v),
    ),
    (
        (100, sp.Integer(64), True),
        "f4042625ede4025ae29dec45a30afcc364420eeca2928219d412b1fed88aac40",
    ),
)

tail_r = sp.symbols("tail_r", integer=True, positive=True)
tail_rho_p1 = sp.factor(
    (4 * tail_r + 3)
    * (4 * tail_r + 9)
    * (8 * tail_r + 9)
    * (8 * tail_r + 13)
    / (
        (4 * tail_r + 5)
        * (4 * tail_r + 7)
        * (8 * tail_r + 5)
        * (8 * tail_r + 17)
    )
)
tail_delta_p1 = 2 / ((4 * tail_r + 3) * (4 * tail_r + 7))
tail_scale_p1 = 2 * (2 * tail_r + 1) * (2 * tail_r + 2)
tail_p1_declared = (
    2
    * (
        2048 * tail_r**4
        + 7296 * tail_r**3
        + 8928 * tail_r**2
        + 4332 * tail_r
        + 679
    )
    / (
        (4 * tail_r + 3)
        * (4 * tail_r + 5)
        * (4 * tail_r + 7)
        * (8 * tail_r + 5)
        * (8 * tail_r + 17)
    )
)
tail_p1_margin = sp.factor(
    tail_scale_p1 * (tail_rho_p1 - 1) - tail_delta_p1
)
check(
    "same-parity U_2 p=1 exact row",
    (
        sp.factor(tail_p1_margin - tail_p1_declared),
        shifted_numerator_certificate(
            tail_p1_margin,
            {tail_r: tail_u + 1},
            (tail_u,),
        ),
    ),
    (
        0,
        (
            (5, sp.Integer(4096), True),
            "3b8c1788ada83c852302c5c04ec7f2d0d0d508e3e19c69fadb8000eec6afa4ca",
        ),
    ),
)

tail_rho_p2 = sp.factor(
    (2 * tail_r + 1) ** 2
    * (3 * tail_r + 4)
    * (4 * tail_r + 5)
    * (6 * tail_r + 1)
    * (8 * tail_r + 1)
    * (8 * tail_r + 13)
    * (12 * tail_r + 13)
    / (
        (2 * tail_r + 3) ** 2
        * (3 * tail_r + 1)
        * (4 * tail_r + 1)
        * (6 * tail_r + 7)
        * (8 * tail_r + 5)
        * (8 * tail_r + 9)
        * (12 * tail_r + 1)
    )
)
tail_delta_p2 = sp.factor(
    4
    * (
        72 * tail_r**4
        + 192 * tail_r**3
        + 166 * tail_r**2
        + 54 * tail_r
        + 11
    )
    / (
        3
        * (2 * tail_r + 1)
        * (2 * tail_r + 3)
        * (4 * tail_r + 1)
        * (4 * tail_r + 5)
        * (6 * tail_r + 1)
        * (6 * tail_r + 7)
    )
)
tail_scale_p2 = 6 * (2 * tail_r) * (2 * tail_r + 1)
tail_p2_numerator = 4 * (
    5_308_416 * tail_r**10
    + 31_684_608 * tail_r**9
    + 79_478_784 * tail_r**8
    + 108_897_984 * tail_r**7
    + 88_918_416 * tail_r**6
    + 44_487_496 * tail_r**5
    + 13_537_948 * tail_r**4
    + 2_350_994 * tail_r**3
    + 168_440 * tail_r**2
    - 9_726 * tail_r
    - 1_485
)
tail_p2_denominator = (
    3
    * (2 * tail_r + 1)
    * (2 * tail_r + 3) ** 2
    * (3 * tail_r + 1)
    * (4 * tail_r + 1)
    * (4 * tail_r + 5)
    * (6 * tail_r + 1)
    * (6 * tail_r + 7)
    * (8 * tail_r + 5)
    * (8 * tail_r + 9)
    * (12 * tail_r + 1)
)
tail_p2_margin = sp.factor(
    tail_scale_p2 * (tail_rho_p2 - 1) - tail_delta_p2
)
check(
    "same-parity U_2 p=2 exact row",
    (
        sp.factor(
            tail_p2_margin - tail_p2_numerator / tail_p2_denominator
        ),
        shifted_numerator_certificate(
            tail_p2_margin,
            {tail_r: tail_u + 1},
            (tail_u,),
        ),
    ),
    (
        0,
        (
            (11, sp.Integer(21_233_664), True),
            "cf1a845a5e4774545f19d7c0b2b7ad72f92ef774c8451893e3bd69554f1ea53a",
        ),
    ),
)


def odd_cycle_value_at_four(n: int) -> Fraction:
    m = n // 2
    central = Fraction(math.comb(2 * m, m), 4**m)
    return (
        (4 * m + 1) * central
        if n % 2 == 0
        else (2 * m + 1) * central
    )


def second_threshold_tail(p: int, q: int) -> Fraction:
    left, right = closure_corners(p, q)
    left_value = math.prod(odd_cycle_value_at_four(n) for n in left)
    right_value = math.prod(odd_cycle_value_at_four(n) for n in right)
    return left_value - right_value - 3 * delta_one_cycle(p, q)


check(
    "same-parity U_2(1,3)",
    second_threshold_tail(1, 3),
    Fraction(13_402_743, 10_092_544),
)
check(
    "same-parity U_2(2,2)",
    second_threshold_tail(2, 2),
    Fraction(361_955_358_061, 366_414_397_440),
)
check(
    "same-parity U_2 is positive on the 105-cell exact regression census",
    all(second_threshold_tail(p, q) > 0 for p, q in same_parity_cells),
    True,
)


# -------------------------------------- threshold tails and ladder LR order
formal_deltas = sp.symbols("formal_delta_1:8")
tail_w = sp.symbols("tail_w")
direct_tail_shift = sp.Poly(
    sp.expand(
        sum(
            formal_deltas[k - 1] * (3 + tail_w) ** (k - 1)
            for k in range(1, 8)
        )
    ),
    tail_w,
)
formal_tails = {
    r: sum(3**k * formal_deltas[k - 1] for k in range(r, 8))
    for r in range(1, 8)
}
check(
    "threshold-tail constant coefficient",
    direct_tail_shift.coeff_monomial(1),
    formal_tails[1] / 3,
)
check(
    "threshold-tail shifted coefficients",
    all(
        sp.simplify(
            direct_tail_shift.coeff_monomial(tail_w**j)
            - sp.Rational(1, 3 ** (j + 1))
            * sum(
                sp.binomial(r - 2, j - 1) * formal_tails[r]
                for r in range(j + 1, 8)
            )
        )
        == 0
        for j in range(1, 7)
    ),
    True,
)


def threshold_law(row: list[Fraction]) -> list[Fraction]:
    value_at_four = fraction_poly_evaluate(row, Fraction(4))
    return poly_scale(affine_fraction_shift(row, 1, 3), 1 / value_at_four)


threshold_normalizations_ok = True
threshold_masses_ok = True
threshold_pgf_recurrences_ok = True
threshold_tail_recurrences_ok = True
threshold_recurrence_gates = 0
for m in range(19):
    central = Fraction(math.comb(2 * m, m), 4**m)
    even_value = fraction_poly_evaluate(cycle_rows[2 * m], Fraction(4))
    odd_value = fraction_poly_evaluate(cycle_rows[2 * m + 1], Fraction(4))
    threshold_normalizations_ok &= even_value == (4 * m + 1) * central
    threshold_normalizations_ok &= odd_value == (2 * m + 1) * central

    even_law = threshold_law(cycle_rows[2 * m])
    odd_law = threshold_law(cycle_rows[2 * m + 1])
    next_even_law = threshold_law(cycle_rows[2 * m + 2])
    threshold_masses_ok &= sum(even_law, Fraction(0)) == 1
    threshold_masses_ok &= sum(odd_law, Fraction(0)) == 1

    if m >= 1:
        previous_odd_law = threshold_law(cycle_rows[2 * m - 1])
        declared_odd = poly_add(
            poly_scale(
                even_law, Fraction(4 * m + 1, (2 * m + 1) ** 2)
            ),
            poly_scale(
                previous_odd_law,
                Fraction(4 * m * m, (2 * m + 1) ** 2),
            ),
        )
        threshold_pgf_recurrences_ok &= declared_odd == odd_law
        threshold_recurrence_gates += 1

    declared_even = poly_add(
        poly_scale(even_law, Fraction(4 * m + 1, 4 * m + 5)),
        poly_scale(
            poly_multiply([Fraction(1), Fraction(3)], odd_law),
            Fraction(1, 4 * m + 5),
        ),
    )
    threshold_pgf_recurrences_ok &= declared_even == next_even_law
    threshold_recurrence_gates += 1

    def survival(values: list[Fraction], threshold: int) -> Fraction:
        return sum(values[threshold:], Fraction(0))

    for threshold in range(1, len(next_even_law)):
        declared_tail = (
            (4 * m + 1) * survival(even_law, threshold)
            + survival(odd_law, threshold)
            + 3 * survival(odd_law, threshold - 1)
        ) / (4 * m + 5)
        threshold_tail_recurrences_ok &= (
            survival(next_even_law, threshold) == declared_tail
        )
        threshold_recurrence_gates += 1

check(
    "threshold Q_n(4) parity formulas through m=18",
    threshold_normalizations_ok,
    True,
)
check(
    "threshold laws are normalized through m=18",
    threshold_masses_ok,
    True,
)
check(
    "positive threshold PGF recurrences through m=18",
    threshold_pgf_recurrences_ok,
    True,
)
check(
    f"positive threshold survival recurrence ({threshold_recurrence_gates} gates)",
    threshold_tail_recurrences_ok,
    True,
)

cemetery_cells = ((1, 2), (1, 3), (2, 2), (2, 3), (4, 5))
cemetery_identity_ok = True
cemetery_anchor_ok = True
cemetery_tail_gates = 0
cemetery_controls: tuple[Fraction, Fraction, Fraction] | None = None
for p, q in cemetery_cells:
    left_indices, right_indices = closure_corners(p, q)
    left_z = poly_product(
        [affine_fraction_shift(cycle_rows[index], 1, 1)
         for index in left_indices]
    )
    right_z = poly_product(
        [affine_fraction_shift(cycle_rows[index], 1, 1)
         for index in right_indices]
    )
    left_at_four = fraction_poly_evaluate(left_z, Fraction(3))
    right_at_four = fraction_poly_evaluate(right_z, Fraction(3))
    cemetery_anchor_ok &= left_at_four > right_at_four
    theta_value = right_at_four / left_at_four
    left_law = poly_scale(
        poly_product(
            [affine_fraction_shift(cycle_rows[index], 1, 3)
             for index in left_indices]
        ),
        1 / left_at_four,
    )
    right_law = poly_scale(
        poly_product(
            [affine_fraction_shift(cycle_rows[index], 1, 3)
             for index in right_indices]
        ),
        1 / right_at_four,
    )
    degree = max(len(left_z), len(right_z)) - 1
    for threshold in range(1, degree + 1):
        weighted_delta_tail = sum(
            Fraction(3**index)
            * (
                (left_z[index] if index < len(left_z) else 0)
                - (right_z[index] if index < len(right_z) else 0)
            )
            for index in range(threshold, degree + 1)
        )
        probability_gap = left_at_four * (
            sum(left_law[threshold:], Fraction(0))
            - theta_value * sum(right_law[threshold:], Fraction(0))
        )
        cemetery_identity_ok &= weighted_delta_tail == probability_gap
        cemetery_tail_gates += 1
    if (p, q) == (1, 3):
        threshold = 3
        undiluted = (
            sum(left_law[threshold:], Fraction(0))
            - sum(right_law[threshold:], Fraction(0))
        )
        diluted = (
            sum(left_law[threshold:], Fraction(0))
            - theta_value * sum(right_law[threshold:], Fraction(0))
        )
        cemetery_controls = theta_value, undiluted, diluted

check("finite cemetery anchors in five cells", cemetery_anchor_ok, True)
check(
    f"cemetery-tail probability identity ({cemetery_tail_gates} tails)",
    cemetery_identity_ok,
    True,
)
check(
    "cemetery mass is load-bearing at (p,q,r)=(1,3,3)",
    cemetery_controls,
    (
        Fraction(825, 833),
        Fraction(-33_616_166_912, 43_755_186_234_375),
        Fraction(628_087_384, 95_465_860_875),
    ),
)

n_p, n_next, w_p, w_next, u_symbol = sp.symbols(
    "n_p n_next w_p w_next u_symbol", nonzero=True
)
c_p = (n_p - w_p) / (u_symbol - 1)
c_next = (n_next - w_next) / (u_symbol - 1)
abstract_closure = n_next * w_p - w_next * n_p
check(
    "abstract consecutive-scale factorization",
    sp.simplify(
        abstract_closure
        - w_p * w_next * (n_next / w_next - n_p / w_p)
    ),
    0,
)
check(
    "abstract C quotient factorization",
    sp.simplify(
        abstract_closure / (u_symbol - 1)
        - (w_p * c_next - w_next * c_p)
    ),
    0,
)


# Bounded mixed-parity coefficient-ratio probe after shifting to u=2.
# R_n=n! Q_n is used below; its positive per-index scaling preserves every
# A/B coefficient minor and every Wronskian sign.
mixed_lr_digest = hashlib.sha256()
mixed_lr_cells = 0
mixed_lr_minors = 0
mixed_lr_wronskians = 0
mixed_lr_maximum_degree = 0
mixed_lr_degree_ok = True
mixed_lr_anchor_ok = True
mixed_lr_minors_ok = True
mixed_lr_wronskians_ok = True
for p in range(1, 11):
    for q in range(2, 12):
        if p % 2 == q % 2:
            continue
        left_indices, right_indices = closure_corners(p, q)
        left = int_poly_product(
            [int_poly_shift(integer_rows[index], 2) for index in left_indices]
        )
        right = int_poly_product(
            [int_poly_shift(integer_rows[index], 2) for index in right_indices]
        )
        degree = 2 * p * q + 3 * p + q + 1
        mixed_lr_degree_ok &= len(left) == degree + 1
        mixed_lr_degree_ok &= len(right) == degree + 1

        minors = [
            left[index + 1] * right[index]
            - left[index] * right[index + 1]
            for index in range(degree)
        ]
        mixed_lr_minors_ok &= all(value > 0 for value in minors)
        mixed_lr_minors += len(minors)

        wronskian = int_poly_subtract(
            int_poly_multiply(int_poly_derivative(left), right),
            int_poly_multiply(left, int_poly_derivative(right)),
        )
        mixed_lr_degree_ok &= len(wronskian) == 2 * degree - 1
        mixed_lr_wronskians_ok &= all(value > 0 for value in wronskian)
        mixed_lr_wronskians += len(wronskian)
        mixed_lr_maximum_degree = max(
            mixed_lr_maximum_degree, len(wronskian) - 1
        )

        left_at_four = math.prod(
            Fraction(
                sum(value * 4**degree_index
                    for degree_index, value in enumerate(integer_rows[index])),
                factorials[index],
            )
            for index in left_indices
        )
        right_at_four = math.prod(
            Fraction(
                sum(value * 4**degree_index
                    for degree_index, value in enumerate(integer_rows[index])),
                factorials[index],
            )
            for index in right_indices
        )
        mixed_lr_anchor_ok &= left_at_four > right_at_four

        mixed_lr_digest.update(f"{p},{q},{degree},".encode("ascii"))
        for value in minors:
            mixed_lr_digest.update(str(value).encode("ascii"))
            mixed_lr_digest.update(b",")
        mixed_lr_digest.update(b"|")
        for value in wronskian:
            mixed_lr_digest.update(str(value).encode("ascii"))
            mixed_lr_digest.update(b",")
        mixed_lr_digest.update(b"\n")
        mixed_lr_cells += 1

first_left_indices, first_right_indices = closure_corners(1, 2)
first_left = poly_product(
    [affine_fraction_shift(cycle_rows[index], 2, 1)
     for index in first_left_indices]
)
first_right = poly_product(
    [affine_fraction_shift(cycle_rows[index], 2, 1)
     for index in first_right_indices]
)
check(
    "MIXED-LR_2 normalized first minor at (1,2)",
    first_left[1] * first_right[0] - first_left[0] * first_right[1],
    Fraction(3_696_009_788_814_959, 49_533_891_379_200_000),
)
check("MIXED-LR_2 probe has 50 cells", mixed_lr_cells, 50)
check("MIXED-LR_2 common degrees", mixed_lr_degree_ok, True)
check("MIXED-LR_2 finite u=4 anchors", mixed_lr_anchor_ok, True)
check(
    "MIXED-LR_2 adjacent minors",
    (mixed_lr_minors, mixed_lr_minors_ok),
    (4_800, True),
)
check(
    "MIXED-LR_2 Wronskian coefficients",
    (mixed_lr_wronskians, mixed_lr_wronskians_ok),
    (9_550, True),
)
check(
    "MIXED-LR_2 total signs and maximum degree",
    (mixed_lr_minors + mixed_lr_wronskians, mixed_lr_maximum_degree),
    (14_350, 522),
)
check(
    "MIXED-LR_2 canonical reconstruction digest",
    mixed_lr_digest.hexdigest(),
    "d0e82f20c391d484ecb31b208ca975f31d9a3bcd880349b3868a80aa54ac1468",
)


def positive_green_block(n: int) -> sp.Matrix:
    if n == 1:
        return sp.zeros(0)
    green = (sp.eye(n) - K(n) * K(n)).inv()
    indices = [index for index in range(n) if index % 2 == n % 2]
    return green.extract(indices, indices)


def threshold_resolvent(block: sp.Matrix, c_value: sp.Rational) -> sp.Matrix:
    if block.rows == 0:
        return sp.zeros(0)
    return sp.simplify(
        block * (sp.eye(block.rows) + c_value * block).inv()
    )


def determinant_coefficients(matrix: sp.Matrix) -> list[sp.Rational]:
    variable = sp.symbols("determinant_variable")
    if matrix.rows == 0:
        return [sp.Rational(1)]
    polynomial = sp.Poly(
        sp.expand((sp.eye(matrix.rows) + variable * matrix).det()),
        variable,
    )
    return [
        sp.Rational(polynomial.coeff_monomial(variable**degree))
        for degree in range(matrix.rows + 1)
    ]


f4_coefficients = determinant_coefficients(
    threshold_resolvent(positive_green_block(4), sp.Rational(3))
)
f6_coefficients = determinant_coefficients(
    threshold_resolvent(positive_green_block(6), sp.Rational(3))
)
check(
    "threshold-shift F_4 coefficients",
    f4_coefficients,
    [sp.Rational(1), sp.Rational(22, 81), sp.Rational(1, 81)],
)
check(
    "threshold-shift F_6 coefficients",
    f6_coefficients,
    [
        sp.Rational(1),
        sp.Rational(103, 325),
        sp.Rational(67, 2925),
        sp.Rational(1, 2925),
    ],
)
check(
    "threshold-shift first LR cross-margin",
    f6_coefficients[1] * f4_coefficients[0]
    - f6_coefficients[0] * f4_coefficients[1],
    sp.Rational(1193, 26325),
)
check(
    "threshold-shift second LR cross-margin",
    f6_coefficients[2] * f4_coefficients[1]
    - f6_coefficients[1] * f4_coefficients[2],
    sp.Rational(547, 236925),
)

for n in range(1, 9):
    old_block = positive_green_block(n)
    new_block = positive_green_block(n + 2)
    dimension = old_block.rows + 1
    embedded = sp.zeros(dimension)
    if old_block.rows:
        embedded[: old_block.rows, : old_block.rows] = old_block
    vector = sp.ones(dimension, 1)
    alpha_value = sp.Rational(1, (n + 1) * (n + 2))
    check(
        f"positive Green rank-one birth at n={n}",
        sp.simplify(
            new_block - embedded - alpha_value * vector * vector.T
        ) == sp.zeros(dimension),
        True,
    )
    for u0_value in (2, 4, 7):
        c_value = sp.Rational(u0_value - 1)
        old_resolvent = threshold_resolvent(old_block, c_value)
        new_resolvent = threshold_resolvent(new_block, c_value)
        embedded_resolvent = sp.zeros(dimension)
        if old_resolvent.rows:
            embedded_resolvent[
                : old_resolvent.rows, : old_resolvent.rows
            ] = old_resolvent
        inverse_vector = (
            sp.eye(dimension) + c_value * embedded
        ).inv() * vector
        denominator = 1 + c_value * alpha_value * (
            vector.T
            * (sp.eye(dimension) + c_value * embedded).inv()
            * vector
        )[0]
        correction = (
            alpha_value
            * inverse_vector
            * inverse_vector.T
            / denominator
        )
        check(
            f"threshold resolvent rank-one birth n={n}, u0={u0_value}",
            sp.simplify(
                new_resolvent - embedded_resolvent - correction
            ) == sp.zeros(dimension),
            True,
        )
        old_coefficients = determinant_coefficients(old_resolvent)
        new_coefficients = determinant_coefficients(new_resolvent)
        ratios = [
            new_coefficients[index] / old_coefficients[index]
            for index in range(len(old_coefficients))
        ]
        check(
            f"threshold coefficient ratios increase n={n}, u0={u0_value}",
            all(
                ratios[index + 1] >= ratios[index]
                for index in range(len(ratios) - 1)
            ),
            True,
        )


# ------------------------------------------- thm:inc shifted-basis escape
# Negative-results register: in v = x-1 the first normalized increment has
# all nonnegative coefficients, so the positive-at-x=1 exclusion does not
# extend to variables vanishing at x=1.
v_shift = sp.Symbol("v")


def _delta_cell(p: int, q: int, at):
    a, b, c, d = cell(p, q)
    return sp.expand(T_sym(at, a) * T_sym(at, b) - T_sym(at, c) * T_sym(at, d))


_increment = sp.expand(
    sp.expand_func(
        _delta_cell(1, 2, 1 + v_shift) - _delta_cell(1, 1, 1 + v_shift)
    )
)
_displayed = sp.expand(
    sp.Rational(2, 45)
    * v_shift
    * (v_shift + 1)
    * (
        4 * v_shift**6
        + 24 * v_shift**5
        + 74 * v_shift**4
        + 116 * v_shift**3
        + 105 * v_shift**2
        + 97 * v_shift
        + 75
    )
)
check(
    "thm:inc escape witness, increment expansion in v = x-1",
    sp.expand(_increment - _displayed),
    0,
)
check(
    "thm:inc escape witness, all coefficients nonnegative",
    all(c >= 0 for c in sp.Poly(_increment, v_shift).all_coeffs()),
    True,
)


# ---------------------------------------------------------------- verdict
print()
if FAILURES:
    print(f"FAIL: {len(FAILURES)} anchor(s) did not reproduce")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)
print(f"PASS: every manuscript anchor reproduced exactly ({CHECKS} exact checks)")
sys.exit(0)
