#!/usr/bin/env python3
"""Build the exact proof certificate for the two compact dimension-step strips."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import sympy as sp


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "dimension_step_compact_certificate.json"
SCHEMA = "dimension-step-compact-v1"


def fraction_text(value: sp.Expr | Fraction | int) -> str:
    rational = sp.Rational(value)
    return f"{int(rational.p)}/{int(rational.q)}"


def polynomial_record(expression: sp.Expr, variables: tuple[sp.Symbol, ...]) -> dict[str, object]:
    poly = sp.Poly(sp.expand(expression), *variables, domain=sp.QQ)
    terms = poly.terms()
    payload = "\n".join(
        f"{','.join(map(str, monomial))}:{fraction_text(coefficient)}"
        for monomial, coefficient in terms
    )
    coefficients = [sp.Rational(coefficient) for _, coefficient in terms]
    integer_content = sp.gcd_list(
        [int(coefficient) for coefficient in coefficients if coefficient.q == 1]
    ) if coefficients and all(coefficient.q == 1 for coefficient in coefficients) else None
    return {
        "variables": [str(variable) for variable in variables],
        "degrees": list(poly.degree_list()),
        "term_count": len(terms),
        "minimum": fraction_text(min(coefficients)),
        "maximum": fraction_text(max(coefficients)),
        "strictly_positive": bool(coefficients and all(value > 0 for value in coefficients)),
        "nonnegative": bool(coefficients and all(value >= 0 for value in coefficients)),
        "zero_count": 0,
        "integer_content": int(integer_content) if integer_content is not None else None,
        "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }


def vector_record(values: list[sp.Expr], labels: list[str]) -> dict[str, object]:
    rationals = [sp.Rational(value) for value in values]
    payload = "\n".join(
        f"{label}:{fraction_text(value)}" for label, value in zip(labels, rationals)
    )
    return {
        "entry_count": len(rationals),
        "minimum": fraction_text(min(rationals)),
        "maximum": fraction_text(max(rationals)),
        "strictly_positive": all(value > 0 for value in rationals),
        "nonnegative": all(value >= 0 for value in rationals),
        "zero_count": sum(value == 0 for value in rationals),
        "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }


def bernstein_coefficients(expression: sp.Expr, variable: sp.Symbol, degree: int) -> list[sp.Expr]:
    poly = sp.Poly(sp.expand(expression), variable, domain="EX")
    power = [poly.coeff_monomial(variable**k) for k in range(degree + 1)]
    return [
        sp.factor(sum(
            power[k] * sp.Rational(sp.binomial(j, k), sp.binomial(degree, k))
            for k in range(j + 1)
        ))
        for j in range(degree + 1)
    ]


def tensor_bernstein(
    expression: sp.Expr, specifications: tuple[tuple[sp.Symbol, int], ...]
) -> list[tuple[tuple[int, ...], sp.Expr]]:
    rows: list[tuple[tuple[int, ...], sp.Expr]] = [((), expression)]
    for variable, degree in specifications:
        expanded: list[tuple[tuple[int, ...], sp.Expr]] = []
        for prefix, value in rows:
            for index, coefficient in enumerate(bernstein_coefficients(value, variable, degree)):
                expanded.append((prefix + (index,), coefficient))
        rows = expanded
    return rows


def bernstein_power_record(
    rows: list[tuple[tuple[int, ...], sp.Expr]], parameter: sp.Symbol
) -> dict[str, object]:
    values: list[sp.Expr] = []
    labels: list[str] = []
    for basis, expression in rows:
        poly = sp.Poly(sp.expand(expression), parameter, domain=sp.QQ)
        if poly.is_zero:
            values.append(sp.Integer(0))
            labels.append(f"{','.join(map(str, basis))}:0")
            continue
        for degree in range(poly.degree() + 1):
            values.append(poly.nth(degree))
            labels.append(f"{','.join(map(str, basis))}:{degree}")
    record = vector_record(values, labels)
    record["basis_count"] = len(rows)
    return record


def state_map(index: sp.Expr, center: sp.Expr, state: sp.Expr) -> sp.Expr:
    index, center, state = map(sp.sympify, (index, center, state))
    return sp.cancel(
        ((center - 4) * (center - 2) + (center - 2 * index - 4) * state)
        / (center + state + 2 * index - 2)
    )


def defect_factor(index: sp.Expr, center: sp.Expr, state: sp.Expr) -> sp.Expr:
    index, center, state = map(sp.sympify, (index, center, state))
    a_value = center**2 + center * state + 4 * center * index - 2 * center + 4 * state * index + 16 * index**2
    b_value = (
        center**2 + center * state + 3 * center * index - 3 * center
        + state * index - state + 4 * index**2 - 2 * index + 2
    )
    c_value = center + state + 4 * index - 2
    d_value = (
        center**3 + center**2 * state + 5 * center**2 * index - center**2
        + 3 * center * state * index + center * state + 12 * center * index**2
        + 2 * center * index - 2 * center + 4 * state * index**2
        + 4 * state * index + 16 * index**3 + 16 * index**2
    )
    return sp.cancel((index + 1) * a_value * b_value / (index * c_value * d_value))


def pair_factor(index: sp.Expr, center: sp.Expr, state: sp.Expr) -> sp.Expr:
    return sp.cancel(
        defect_factor(index, center, state)
        * defect_factor(index + 1, center, state_map(index, center, state))
    )


def step_two_product_lower(start: sp.Expr, count: sp.Expr) -> sp.Expr:
    if count == 0:
        return sp.Integer(1)
    return sp.cancel(
        (start + 2 * count) / start
        * (2 * start + 1) / (2 * (start + 2 * count) + 1)
    )


def quadratic_a(index: sp.Expr) -> sp.Expr:
    return 2 * index**2 + 2 * index + 1


def step_two_product_upper(start: sp.Expr, count: sp.Expr) -> sp.Expr:
    return sp.cancel(
        (start + 2 * count) / start
        * (2 * (start + 2 * count) + 1) / (2 * start + 1)
        * quadratic_a(start) / quadratic_a(start + 2 * count)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    C, n, d, z, u, s = sp.symbols("C n d z u s")
    E = defect_factor(n, C, d)
    H = pair_factor(n, C, d)

    derivative_numerator = sp.together(sp.diff(H, d)).as_numer_denom()[0]
    derivative_core = sp.cancel(derivative_numerator / (4 * n * (C - 2) * (n + 2)))
    derivative_core_shifted = sp.expand(
        derivative_core.subs({C: 3 + z, d: (3 + z) - 4 + u})
    )
    pair_state_record = polynomial_record(derivative_core_shifted, (z, u, n))

    center_records: dict[str, object] = {}
    for name, expression, degree in (
        ("lower_pair", pair_factor(n, C, C - 4), 8),
        ("upper_pair", pair_factor(n, C, 4 - C), 4),
        ("lower_single", defect_factor(n, C, C - 4), 6),
    ):
        numerator = sp.together(sp.diff(expression, C)).as_numer_denom()[0]
        shifted = sp.expand(numerator.subs({C: 3 + z, n: 1 + s}))
        center_records[name] = bernstein_power_record(
            [((index,), value) for index, value in enumerate(bernstein_coefficients(shifted, z, degree))],
            s,
        )

    lower_kernel = (n + 2) * (2 * n + 1) / (n * (2 * n + 5))
    upper_kernel = (
        (n + 2) * (2 * n + 5) * (2 * n**2 + 2 * n + 1)
        / (n * (2 * n + 1) * (2 * n**2 + 10 * n + 13))
    )
    lower_single = 2 * (n + 1) * (2 * n + 1) * (2 * n + 3) / (n * (8 * n**2 + 24 * n + 19))
    upper_single = (n + 1) * (4 * n + 1) ** 2 / (n * (16 * n**2 + 24 * n - 1))
    kernel_identities = {
        "lower_pair": sp.factor(pair_factor(n, 3, -1) - lower_kernel) == 0,
        "upper_pair": sp.factor(pair_factor(n, 4, 0) - upper_kernel) == 0,
        "lower_single": sp.factor(defect_factor(n, 3, -1) - lower_single) == 0,
    }

    endpoint_upper = sp.factor(defect_factor(n, C, 4 - C))
    endpoint_upper_expected = sp.cancel(
        (n + 1) * (C + 2 * n - 1) * (C + 8 * n**2 + 8 * n)
        / (n * (2 * n + 1) * (C**2 + 4 * C * n + C + 8 * n**2 + 8 * n))
    )
    endpoint_upper_derivative = sp.factor(
        sp.together(sp.diff(endpoint_upper, C)).as_numer_denom()[0]
    )
    endpoint_upper_derivative_expected = -2 * (C + 4 * n) * (n + 1) ** 2 * (4 * C * n - C - 12 * n)
    upper_single_identity = {
        "factorization": sp.factor(endpoint_upper - endpoint_upper_expected) == 0,
        "derivative": sp.factor(endpoint_upper_derivative - endpoint_upper_derivative_expected) == 0,
        "maximum_value": sp.factor(
            endpoint_upper.subs(C, 12 * n / (4 * n - 1)) - upper_single
        ) == 0,
    }

    r, p, t, w = sp.symbols("r p t w")
    even_b = 2 * p * r
    even_c = p * (2 * r + 1)
    even_a_minus_one = (p + 1) * (2 * r + 1) - 1
    even_margin = sp.cancel(
        step_two_product_lower(even_b, r)
        / (upper_single.subs(n, even_a_minus_one) * step_two_product_upper(even_c, r))
        - 1
    )
    even_numerator = sp.together(even_margin).as_numer_denom()[0]
    even_shifted = sp.expand(even_numerator.subs(p, 2 * r + w).subs(r, 1 + t) / 2)
    even_record = polynomial_record(even_shifted, (t, w))

    odd_b = p * (2 * r + 1)
    odd_c = 2 * p * (r + 1)
    odd_d_minus_one = (p + 1) * (2 * r + 1) - 1
    odd_margin = sp.cancel(
        lower_single.subs(n, odd_d_minus_one)
        * step_two_product_lower(odd_b, r)
        / step_two_product_upper(odd_c, r + 1)
        - 1
    )
    odd_numerator = sp.together(odd_margin).as_numer_denom()[0]
    odd_shifted = sp.expand(odd_numerator.subs(p, 2 * r + 1 + w))
    odd_record = polynomial_record(odd_shifted, (r, w))

    h, v = sp.symbols("h v")
    G = sp.cancel(
        (d * (h**2 - 2 * h * n + 2 * n**2 + 2 * n) + h**2 * (h + 2))
        / (d * (h + 2) + h**2 + 2 * h * n + 4 * h + 2 * n**2 + 6 * n + 4)
    )
    trap_residual = sp.factor(
        h / (n + 2) - G.subs(d, h / n)
    )
    trap_residual_expected = -h**2 * (h - 2) * (n + 1) / (
        (n + 2) * (h**2 + 2 * h * n + 2 * h + 2 * n**2 + 4 * n)
    )
    g_derivative_numerator = sp.together(sp.diff(G, h)).as_numer_denom()[0]
    g_scaled = sp.cancel(g_derivative_numerator.subs(d, h * v / n) * n**2 / h)
    g_shifted = sp.expand(g_scaled.subs(n, 1 + s))
    g_record = bernstein_power_record(tensor_bernstein(g_shifted, ((h, 3), (v, 2))), s)

    e_center_numerator = sp.together(sp.diff(E, C)).as_numer_denom()[0]
    e_center_shifted = sp.expand(e_center_numerator.subs({C: 4 + h, d: u, n: 1 + s}))
    e_center_record = bernstein_power_record(
        tensor_bernstein(e_center_shifted, ((h, 6), (u, 4))), s
    )

    m = sp.symbols("m", integer=True, positive=True)
    orbit_even = sp.Rational(3, 1) / (4 * (2 * m) + 1)
    orbit_odd = sp.Rational(3, 1) / (4 * (2 * m + 1) - 1)
    orbit_identities = {
        "odd_to_even": sp.factor(
            state_map(2 * m + 1, 5, orbit_odd) - sp.Rational(3, 1) / (4 * (2 * m + 2) + 1)
        ) == 0,
        "even_to_odd": sp.factor(
            state_map(2 * m, 5, orbit_even) - sp.Rational(3, 1) / (4 * (2 * m + 1) - 1)
        ) == 0,
    }

    def r4(index: sp.Expr) -> sp.Expr:
        return (2 * index**2 + 2 * index + 1) / (index * (2 * index + 1))

    def r5_odd(index: sp.Expr) -> sp.Expr:
        return (16 * index**2 + 16 * index + 13) / (10 * index * (2 * index + 1))

    def r5_even(index: sp.Expr) -> sp.Expr:
        return (2 * index + 1) * (16 * index**2 + 16 * index + 15) / (
            5 * index * (8 * index**2 + 8 * index + 3)
        )

    endpoint_ratio_identities = {
        "C4": sp.factor(
            defect_factor(n, 4, 0) - r4(n) / r4(n + 1)
        ) == 0,
        "C5_odd": sp.factor(
            defect_factor(2 * m + 1, 5, orbit_odd)
            - r5_odd(2 * m + 1) / r5_even(2 * m + 2)
        ) == 0,
        "C5_even": sp.factor(
            defect_factor(2 * m, 5, orbit_even)
            - r5_even(2 * m) / r5_odd(2 * m + 1)
        ) == 0,
    }

    endpoint_parity_records: dict[str, object] = {}
    for eta in (0, 1):
        for omega in (0, 1):
            q_value = 2 * r + eta
            p_value = q_value + 2 * v + omega
            a_value = (p_value + 1) * (q_value + 1)
            b_value = p_value * q_value
            c_value = p_value * (q_value + 1)
            d_value = (p_value + 1) * q_value
            if eta == 0 and omega == 0:
                r5_c, r5_a = r5_even(c_value), r5_odd(a_value)
            elif eta == 0:
                r5_c, r5_a = r5_odd(c_value), r5_even(a_value)
            else:
                r5_c, r5_a = r5_even(c_value), r5_even(a_value)
            margin = sp.cancel(
                (r4(b_value) / r4(d_value)) / (r5_c / r5_a) - 1
            )
            numerator = sp.together(margin).as_numer_denom()[0]
            if eta == 0:
                shifted = sp.expand(numerator.subs(r, 1 + t))
                variables = (t, v)
            else:
                shifted = sp.expand(numerator)
                variables = (r, v)
            endpoint_parity_records[f"eta_{eta}_omega_{omega}"] = polynomial_record(
                shifted, variables
            )

    one_factor_lower = defect_factor(4, 3, -1) * defect_factor(5, 3, -1)
    one_factor_upper = (
        defect_factor(6, 3, 1) * defect_factor(7, 3, 1) * defect_factor(8, 3, 1)
    )
    pair_control = sp.cancel(
        pair_factor(4, 3, -1) / (pair_factor(6, 3, 1) * defect_factor(8, 3, 1)) - 1
    )
    controls = {
        "one_factor_lower": fraction_text(one_factor_lower),
        "one_factor_upper": fraction_text(one_factor_upper),
        "one_factor_ratio_minus_one": fraction_text(one_factor_lower / one_factor_upper - 1),
        "pair_ratio_minus_one": fraction_text(pair_control),
    }

    all_records = [
        pair_state_record,
        *center_records.values(),
        even_record,
        odd_record,
        g_record,
        e_center_record,
        *endpoint_parity_records.values(),
    ]
    identities = {
        **kernel_identities,
        **{f"upper_single_{name}": value for name, value in upper_single_identity.items()},
        "trap_residual": sp.factor(trap_residual - trap_residual_expected) == 0,
        **{f"orbit_{name}": value for name, value in orbit_identities.items()},
        **{f"endpoint_ratio_{name}": value for name, value in endpoint_ratio_identities.items()},
    }
    passed = (
        all(identities.values())
        and all(record["strictly_positive"] for record in all_records if record is not g_record)
        and g_record["nonnegative"]
        and g_record["zero_count"] == 1
        and controls["one_factor_ratio_minus_one"] == "-18113657/8184328857"
        and controls["pair_ratio_minus_one"] == "1/113"
    )

    result = {
        "schema": SCHEMA,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification": "all-parameter proof on two compact center strips",
        "worker_contract": "one serial worker; exact SymPy rational arithmetic",
        "identities": identities,
        "strip_3_4": {
            "pair_state_derivative": pair_state_record,
            "center_derivatives": center_records,
            "even_block_numerator": even_record,
            "odd_block_numerator": odd_record,
        },
        "strip_4_5": {
            "state_monotonicity": g_record,
            "defect_center_monotonicity": e_center_record,
            "endpoint_parity_numerators": endpoint_parity_records,
        },
        "controls": controls,
        "passed": passed,
        "dispositions": [
            "DIMENSION_STEP_PROVED_ON_X_IN_HALF_TO_ONE",
            "DIMENSION_STEP_PROVED_ON_X_IN_ONE_TO_THREE_HALVES",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"passed={passed}")
    print(f"output={args.output}")
    print(f"producer_sha256={result['producer_sha256']}")
    for disposition in result["dispositions"]:
        print(f"disposition={disposition}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
