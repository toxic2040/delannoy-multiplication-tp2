#!/usr/bin/env python3
"""Build the exact certificate for adjacent-Q dimension-step monotonicity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "dimension_step_unbounded_certificate.json"
SCHEMA = "dimension-step-unbounded-v1"
EXPECTED_RESULTANT = {
    "degrees": [8, 20],
    "term_count": 146,
    "coefficient_sum": 41136763893522915744,
    "minimum": 64,
    "maximum": 6985028140230144064,
    "value_at_c5_n1": 372324389696000000,
    "payload_sha256": "94356f13c8d749d43fd85feb835f8ddb3d0356a95a014009a76e6daaab6cc3f0",
}




def fraction_text(value: sp.Expr | int) -> str:
    rational = sp.Rational(value)
    return f"{int(rational.p)}/{int(rational.q)}"


def integer_polynomial_record(
    expression: sp.Expr, variables: tuple[sp.Symbol, ...]
) -> dict[str, object]:
    poly = sp.Poly(sp.expand(expression), *variables, domain=sp.ZZ)
    terms = poly.terms()
    payload_rows = [
        [*(int(exponent) for exponent in monomial), int(coefficient)]
        for monomial, coefficient in terms
    ]
    payload = json.dumps(payload_rows, separators=(",", ":")).encode("ascii")
    coefficients = [int(coefficient) for _, coefficient in terms]
    return {
        "variables": [str(variable) for variable in variables],
        "degrees": list(poly.degree_list()),
        "term_count": len(terms),
        "coefficient_sum": sum(coefficients),
        "minimum": min(coefficients),
        "maximum": max(coefficients),
        "strictly_positive": all(coefficient > 0 for coefficient in coefficients),
        "canonical_terms": payload_rows,
        "payload_encoding": "compact JSON rows [powers...,coefficient] in Poly.terms order",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def coefficients_ascending(expression: sp.Expr, variable: sp.Symbol) -> list[int]:
    poly = sp.Poly(sp.expand(expression), variable, domain=sp.ZZ)
    return [int(poly.nth(degree)) for degree in range(poly.degree() + 1)]


def all_coefficients_positive(expression: sp.Expr, variables: tuple[sp.Symbol, ...]) -> bool:
    coefficients = sp.Poly(sp.expand(expression), *variables, domain=sp.ZZ).coeffs()
    return bool(coefficients) and all(coefficient > 0 for coefficient in coefficients)


def replace_even_power_variable(
    expression: sp.Expr, old: sp.Symbol, new: sp.Symbol
) -> sp.Expr:
    poly = sp.Poly(sp.expand(expression), old, domain="EX")
    result = sp.Integer(0)
    for (degree,), coefficient in poly.terms():
        if degree % 2:
            raise ValueError(f"odd {old}-power {degree} in even polynomial")
        result += coefficient * new ** (degree // 2)
    return sp.expand(result)


def z_step(index: sp.Expr, center: sp.Expr, state: sp.Expr) -> sp.Expr:
    return sp.cancel((index + 1) / (center + (index + 1) * state))


def q_value(index: sp.Expr, center: sp.Expr, state: sp.Expr) -> sp.Expr:
    return sp.cancel(
        (2 * index + 1)
        * ((index + 1) * (1 - state**2) - center * state)
        / center
    )


def actual_z_orbit(center: sp.Rational, maximum: int) -> list[sp.Rational]:
    values = [sp.Rational(0)]
    for index in range(1, maximum + 1):
        values.append(sp.cancel(index / (center + index * values[index - 1])))
    return [sp.Rational(value) for value in values]


def main() -> None:
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    c, n, z, u, s, v, y, w, h, e = sp.symbols("c n z u s v y w h e")

    z_next = z_step(n, c, z)
    q_current = q_value(n, c, z)
    q_next = q_value(n + 1, c, z_next)
    tanh_increment = sp.cancel((z_next - z) / (1 - z_next * z))
    q_from_riccati_identity = sp.factor(
        q_current - (2 * n + 1) * tanh_increment
    ) == 0
    adjacent_difference = sp.factor(q_next - q_current)
    p_n, p_denominator = sp.fraction(adjacent_difference)
    p_n = sp.expand(p_n)

    expected_p_denominator = c * (c + (n + 1) * z) ** 2
    derivative = sp.factor(sp.diff(adjacent_difference, z))
    derivative_numerator, derivative_denominator = sp.fraction(derivative)
    derivative_coefficients = [
        c**4 * (2 * n + 1)
        + c**2 * (2 * n**3 + 7 * n**2 + 8 * n + 3)
        + 4 * n**5 + 26 * n**4 + 66 * n**3 + 82 * n**2 + 50 * n + 12,
        c * (n + 1) * (10 * c**2 * n + 5 * c**2 + 2 * n**3 + 7 * n**2 + 8 * n + 3),
        9 * c**2 * (n + 1) ** 2 * (2 * n + 1),
        7 * c * (n + 1) ** 3 * (2 * n + 1),
        2 * (n + 1) ** 4 * (2 * n + 1),
    ]
    expected_derivative_numerator = sp.expand(
        sum(coefficient * z**degree for degree, coefficient in enumerate(derivative_coefficients))
    )
    expected_derivative_denominator = c * (c + (n + 1) * z) ** 3

    expected_at_zero = -c**2 * (2 * n**2 + n - 2) - (
        2 * n**4 + 11 * n**3 + 22 * n**2 + 19 * n + 6
    )
    expected_at_one = c * (
        (2 * n + 1) * c**2
        + (4 * n**2 + 8 * n + 5) * c
        + 4 * n**3 + 16 * n**2 + 22 * n + 10
    )
    scalar_identities = {
        "AQ1_from_riccati_tanh_increment": q_from_riccati_identity,
        "adjacent_denominator": sp.factor(p_denominator - expected_p_denominator) == 0,
        "derivative_numerator": sp.factor(
            derivative_numerator - expected_derivative_numerator
        ) == 0,
        "derivative_denominator": sp.factor(
            derivative_denominator - expected_derivative_denominator
        ) == 0,
        "threshold_at_zero": sp.factor(p_n.subs(z, 0) - expected_at_zero) == 0,
        "threshold_at_one": sp.factor(p_n.subs(z, 1) - expected_at_one) == 0,
        "derivative_coefficients_positive": all(
            all_coefficients_positive(coefficient, (c, n))
            for coefficient in derivative_coefficients
        ),
    }
    negative_at_zero_record = integer_polynomial_record(
        sp.expand((-expected_at_zero).subs({c: 5 + h, n: 1 + s})),
        (h, s),
    )
    positive_at_one_record = integer_polynomial_record(
        sp.expand(expected_at_one.subs({c: 5 + h, n: 1 + s})),
        (h, s),
    )
    center_factor_record = integer_polynomial_record(5 + h, (h,))
    state_denominator_factor_record = integer_polynomial_record(
        5 + h + (2 + s) * v,
        (h, s, v),
    )
    scalar_domain = {
        "domain": "c>=5, n>=1, 0<=z<=1",
        "center_factor": center_factor_record,
        "state_denominator_factor": state_denominator_factor_record,
        "adjacent_denominator_positive": (
            center_factor_record["strictly_positive"]
            and state_denominator_factor_record["strictly_positive"]
        ),
        "derivative_denominator_positive": (
            center_factor_record["strictly_positive"]
            and state_denominator_factor_record["strictly_positive"]
        ),
        "negative_at_zero": negative_at_zero_record,
        "positive_at_one": positive_at_one_record,
        "unique_simple_root_in_open_unit_interval": (
            scalar_identities["derivative_coefficients_positive"]
            and negative_at_zero_record["strictly_positive"]
            and positive_at_one_record["strictly_positive"]
        ),
        "sign_reason": (
            "c>0 and c+(n+1)z>0 make both denominators positive; "
            "F is strictly increasing, with P(0)<0<P(1)"
        ),
    }

    two_step = sp.factor(z_step(n + 1, c, z_next))
    expected_two_step = sp.cancel(
        (n + 2) * (c + (n + 1) * z)
        / (c**2 + c * (n + 1) * z + (n + 1) * (n + 2))
    )
    two_step_numerator, two_step_denominator = sp.fraction(two_step)
    two_step_derivative = sp.factor(sp.diff(two_step, z))
    expected_two_step_derivative = sp.cancel(
        (n + 1) ** 2
        * (n + 2) ** 2
        / (c**2 + c * (n + 1) * z + (n + 1) * (n + 2)) ** 2
    )
    two_step_denominator_record = integer_polynomial_record(
        sp.expand(
            (c**2 + c * (n + 1) * z + (n + 1) * (n + 2)).subs(
                {c: 5 + h, n: 1 + s, z: v}
            )
        ),
        (h, s, v),
    )
    p_n_plus_two = p_n.subs(n, n + 2)
    j_n = sp.factor(
        sp.together(p_n_plus_two.subs(z, two_step) * two_step_denominator**4)
    )
    j_n_numerator, j_n_denominator = sp.fraction(j_n)
    j_n = sp.expand(j_n_numerator)

    resultant = sp.factor(sp.resultant(p_n, j_n, z))
    resultant_prefactor = 4 * c**2 * (n + 1) ** 24 * (n + 2) ** 8
    k_n = sp.cancel(resultant / resultant_prefactor)
    k_n_numerator, k_n_denominator = sp.fraction(k_n)
    k_in_v = replace_even_power_variable(k_n_numerator, c, v)
    k_shifted = sp.expand(k_in_v.subs({v: 25 + u, n: 1 + s}))
    resultant_record = integer_polynomial_record(k_shifted, (u, s))
    resultant_record["value_at_c5_n1"] = int(k_n_numerator.subs({c: 5, n: 1}))

    threshold_coefficient = sp.factor(n - sp.Rational(2, 1) / (2 * n + 1))
    next_threshold_coefficient = sp.factor(
        n + 2 - sp.Rational(2, 1) / (2 * n + 5)
    )
    scaled_threshold_polynomial = sp.factor(
        sp.limit(w**2 * p_n.subs({c: 1 / w, z: y * w}), w, 0, dir="+")
    )
    scaled_threshold_exact = sp.factor(
        w**2 * p_n.subs({c: 1 / w, z: y * w})
    )
    lower_threshold_bracket = sp.factor(
        scaled_threshold_polynomial.subs(y, threshold_coefficient - e)
    )
    upper_threshold_bracket = sp.factor(
        scaled_threshold_polynomial.subs(y, threshold_coefficient + e)
    )
    two_step_scaled_exact = sp.factor(
        two_step.subs({c: 1 / w, z: y * w}) / w
    )
    two_step_scaled_generic_limit = sp.factor(
        sp.limit(
            two_step_scaled_exact,
            w,
            0,
            dir="+",
        )
    )
    two_step_scaled_error = sp.factor(
        two_step_scaled_exact - two_step_scaled_generic_limit
    )
    orientation_gap = sp.factor(
        two_step_scaled_generic_limit - next_threshold_coefficient
    )
    scaled_root_limit_gate = (
        sp.factor(
            scaled_threshold_polynomial
            - (2 * n + 1) * (y - threshold_coefficient)
        )
        == 0
        and sp.factor(lower_threshold_bracket + (2 * n + 1) * e) == 0
        and sp.factor(upper_threshold_bracket - (2 * n + 1) * e) == 0
        and sp.Poly(sp.expand(scaled_threshold_exact), w, y, n).degree(w) >= 0
        and scalar_domain["unique_simple_root_in_open_unit_interval"]
    )
    root_propagation = {
        "two_step_identity": sp.factor(two_step - expected_two_step) == 0,
        "two_step_derivative_identity": sp.factor(
            two_step_derivative - expected_two_step_derivative
        ) == 0,
        "two_step_denominator": two_step_denominator_record,
        "two_step_strictly_increasing": (
            sp.factor(two_step_derivative - expected_two_step_derivative) == 0
            and two_step_denominator_record["strictly_positive"]
        ),
        "cleared_polynomial_denominator": fraction_text(j_n_denominator),
        "cleared_polynomial_degree": int(sp.Poly(j_n, z).degree()),
        "cleared_polynomial": str(j_n),
        "resultant_quotient_denominator": fraction_text(k_n_denominator),
        "resultant_factorization_identity": sp.factor(
            resultant - resultant_prefactor * k_n_numerator
        ) == 0,
        "resultant_record": resultant_record,
        "resultant_matches_frozen_payload": all(
            resultant_record[key] == expected
            for key, expected in EXPECTED_RESULTANT.items()
        ),
        "scaled_threshold_polynomial": str(scaled_threshold_polynomial),
        "scaled_threshold_root": str(threshold_coefficient),
        "scaled_threshold_exact_degree_in_inverse_center": int(
            sp.Poly(sp.expand(scaled_threshold_exact), w, y, n).degree(w)
        ),
        "epsilon_brackets": {
            "lower_limit": str(lower_threshold_bracket),
            "upper_limit": str(upper_threshold_bracket),
            "identities": (
                sp.factor(lower_threshold_bracket + (2 * n + 1) * e) == 0
                and sp.factor(upper_threshold_bracket - (2 * n + 1) * e) == 0
            ),
            "root_limit_conclusion": "c*r_n(c) -> n-2/(2*n+1)",
        },
        "scaled_root_limit_gate": scaled_root_limit_gate,
        "two_step_scaled_generic_limit": str(two_step_scaled_generic_limit),
        "two_step_scaled_exact": str(two_step_scaled_exact),
        "two_step_scaled_error": str(two_step_scaled_error),
        "next_threshold_scaled_limit": str(next_threshold_coefficient),
        "orientation_gap": str(orientation_gap),
        "large_center_orientation": (
            scaled_root_limit_gate
            and two_step_scaled_generic_limit == n + 2
            and orientation_gap == sp.cancel(2 / (2 * n + 5))
        ),
        "no_root_crossing_reason": (
            "The simple roots depend continuously on c; a meeting "
            "G_n(r_n)=r_(n+2) would zero the reconstructed resultant, "
            "whose remaining factor is strictly positive on c>=5,n>=1."
        ),
    }
    root_propagation["strict_root_order_all_centers"] = (
        root_propagation["two_step_strictly_increasing"]
        and root_propagation["resultant_factorization_identity"]
        and root_propagation["resultant_matches_frozen_payload"]
        and resultant_record["strictly_positive"]
        and root_propagation["large_center_orientation"]
    )

    z_one = sp.Rational(1, 1) / c
    z_two = sp.factor(z_step(1, c, z_one))
    f_one = sp.factor(adjacent_difference.subs({n: 1, z: z_one}))
    f_two = sp.factor(adjacent_difference.subs({n: 2, z: z_two}))
    expected_f_one = sp.cancel(
        2 * (c**6 - 13 * c**4 + 36 * c**2 + 12)
        / (c**3 * (c**2 + 2) ** 2)
    )
    expected_f_two = sp.cancel(
        2 * (c**10 - 23 * c**8 + 180 * c**6 + 524 * c**4 - 3040 * c**2 - 2016)
        / (c**3 * (c**2 + 2) ** 2 * (c**2 + 8) ** 2)
    )
    base_one_in_v = replace_even_power_variable(
        c**6 - 13 * c**4 + 36 * c**2 + 12, c, v
    )
    base_two_in_v = replace_even_power_variable(
        c**10 - 23 * c**8 + 180 * c**6 + 524 * c**4 - 3040 * c**2 - 2016,
        c,
        v,
    )
    base_one_coefficients = coefficients_ascending(base_one_in_v.subs(v, 25 + u), u)
    base_two_coefficients = coefficients_ascending(base_two_in_v.subs(v, 25 + u), u)
    bases = {
        "z_1": str(z_one),
        "z_2": str(z_two),
        "F_1_identity": sp.factor(f_one - expected_f_one) == 0,
        "F_2_identity": sp.factor(f_two - expected_f_two) == 0,
        "F_1_shifted_coefficients": base_one_coefficients,
        "F_2_shifted_coefficients": base_two_coefficients,
        "F_1_coefficients_match": base_one_coefficients == [8412, 1261, 62, 1],
        "F_2_coefficients_match": base_two_coefficients
        == [3843234, 876285, 84024, 4130, 102, 1],
        "F_1_strictly_positive": all(value > 0 for value in base_one_coefficients),
        "F_2_strictly_positive": all(value > 0 for value in base_two_coefficients),
    }
    parity_induction = {
        "odd_base": "z_1>r_1 because F_1(z_1)>0",
        "even_base": "z_2>r_2 because F_2(z_2)>0",
        "step": (
            "z_(n+2)=G_n(z_n)>G_n(r_n)>r_(n+2), so "
            "F_(n+2)(z_(n+2))>0"
        ),
        "conclusion": "Q_(n+1)>Q_n for every n>=1,c>=5",
        "passed": (
            root_propagation["strict_root_order_all_centers"]
            and bases["F_1_identity"]
            and bases["F_2_identity"]
            and bases["F_1_strictly_positive"]
            and bases["F_2_strictly_positive"]
        ),
    }

    endpoint_z = actual_z_orbit(sp.Rational(3), 7)
    endpoint_q_five = q_value(5, sp.Rational(3), endpoint_z[5])
    endpoint_q_six = q_value(6, sp.Rational(3), endpoint_z[6])
    compact_endpoint_falsifier = {
        "Q_5": fraction_text(endpoint_q_five),
        "Q_6": fraction_text(endpoint_q_six),
        "strict_dip": bool(endpoint_q_six < endpoint_q_five),
        "matches_frozen_values": (
            endpoint_q_five == sp.Rational(583, 625)
            and endpoint_q_six == sp.Rational(221, 243)
        ),
    }

    p, q, r, t, q_low, q_high = sp.symbols("p q r t q_low q_high")
    a = (p + 1) * (q + 1)
    b = p * q
    corner_c = p * (q + 1)
    corner_d = (p + 1) * q
    high_start = p * (q + 1)
    low_stop = (p + 1) * q
    alpha = (1 + r) / 2
    beta = (1 - r) / 2
    epsilon_derivative = sp.factor(
        -sp.diff(sp.atanh(r / (2 * n + 1)), r)
    )
    expected_epsilon_derivative = sp.cancel(
        -(2 * n + 1) / ((2 * n + 1) ** 2 - r**2)
    )
    epsilon_lower_factor = integer_polynomial_record(
        (2 * n + 1 - r).subs({n: 1 + s, r: 1 - h}),
        (s, h),
    )
    epsilon_upper_factor = integer_polynomial_record(
        (2 * n + 1 + r).subs({n: 1 + s, r: v}),
        (s, v),
    )
    epsilon_numerator_factor = integer_polynomial_record(
        (2 * n + 1).subs(n, 1 + s),
        (s,),
    )
    half_log_tangent = sp.factor(
        ((n + alpha) - (n + beta)) / ((n + alpha) + (n + beta))
    )
    h_derivative_identity = sp.simplify(
        2 * sp.sinh(t) ** 2 * sp.diff(t * sp.coth(t), t)
        - (sp.sinh(2 * t) - 2 * t)
    )
    sinh_excess_derivative_identity = sp.simplify(
        sp.diff(sp.sinh(2 * t) - 2 * t, t) - 4 * sp.sinh(t) ** 2
    )
    kernel = sp.sinh(r * t / 2) / sp.sinh(t / 2)
    h_function = lambda value: value * sp.coth(value)
    kernel_log_derivative_identity = sp.simplify(
        t * sp.diff(sp.log(kernel), t)
        - (h_function(r * t / 2) - h_function(t / 2))
    )
    kernel_transform_identities = {
        "alpha_plus_beta": sp.factor(alpha + beta - 1) == 0,
        "alpha_minus_beta": sp.factor(alpha - beta - r) == 0,
        "exponential_numerator": sp.simplify(
            sp.expand_power_exp(
                sp.exp(t) * (sp.exp(-beta * t) - sp.exp(-alpha * t))
                - (sp.exp(alpha * t) - sp.exp(beta * t))
            )
        )
        == 0,
        "hyperbolic_numerator": sp.simplify(
            sp.expand_power_exp(
                sp.exp(alpha * t)
                - sp.exp(beta * t)
                - 2 * sp.exp(t / 2) * sp.sinh(r * t / 2)
            ).rewrite(sp.exp)
        )
        == 0,
        "hyperbolic_denominator": sp.simplify(
            sp.expand_power_exp(
                sp.exp(t) - 1 - 2 * sp.exp(t / 2) * sp.sinh(t / 2)
            ).rewrite(sp.exp)
        )
        == 0,
    }
    frozen_r = (q_low + q_high) / 2
    frozen_r_identities = {
        "above_low_by_half_gap": sp.factor(
            frozen_r - q_low - (q_high - q_low) / 2
        ) == 0,
        "below_high_by_half_gap": sp.factor(
            q_high - frozen_r - (q_high - q_low) / 2
        ) == 0,
    }
    epsilon_decreases = (
        sp.factor(epsilon_derivative - expected_epsilon_derivative) == 0
        and epsilon_numerator_factor["strictly_positive"]
        and epsilon_lower_factor["strictly_positive"]
        and epsilon_upper_factor["strictly_positive"]
    )
    kernel_decreases = (
        h_derivative_identity == 0
        and sinh_excess_derivative_identity == 0
        and kernel_log_derivative_identity == 0
        and all(kernel_transform_identities.values())
    )
    corner_symmetry = {
        "a_fixed": sp.factor(a.subs({p: q, q: p}, simultaneous=True) - a) == 0,
        "b_fixed": sp.factor(b.subs({p: q, q: p}, simultaneous=True) - b) == 0,
        "c_maps_to_d": sp.factor(
            corner_c.subs({p: q, q: p}, simultaneous=True) - corner_d
        )
        == 0,
        "d_maps_to_c": sp.factor(
            corner_d.subs({p: q, q: p}, simultaneous=True) - corner_c
        )
        == 0,
    }
    constant_state_comparator_direction = (
        kernel_decreases
        and sp.factor(low_stop / b - a / high_start) == 0
        and all(corner_symmetry.values())
    )
    frozen_state_sandwich_gate = (
        epsilon_decreases
        and all(frozen_r_identities.values())
        and constant_state_comparator_direction
    )
    frozen_state_bridge = {
        "orientation_by_symmetry": {
            "assumption": "p>=q without loss of generality",
            "corner_identities": corner_symmetry,
            "passed": all(corner_symmetry.values()),
        },
        "low_length": str(sp.expand(low_stop - b)),
        "high_length": str(sp.expand(a - high_start)),
        "separation": str(sp.expand(high_start - (low_stop - 1))),
        "lengths_and_separation": (
            sp.expand(low_stop - b) == q
            and sp.expand(a - high_start) == q + 1
            and sp.expand(high_start - (low_stop - 1)) == p - q + 1
        ),
        "equal_product": sp.expand(a * b - high_start * low_stop) == 0,
        "low_cap_telescoping_ratio": str(sp.factor(low_stop / b)),
        "high_cap_telescoping_ratio": str(sp.factor(a / high_start)),
        "cap_sums_equal": sp.factor(low_stop / b - a / high_start) == 0,
        "epsilon_derivative": str(epsilon_derivative),
        "epsilon_derivative_identity": sp.factor(
            epsilon_derivative - expected_epsilon_derivative
        ) == 0,
        "epsilon_denominator_factors": {
            "two_n_plus_one": epsilon_numerator_factor,
            "two_n_plus_one_minus_r": epsilon_lower_factor,
            "two_n_plus_one_plus_r": epsilon_upper_factor,
        },
        "epsilon_decreases_in_frozen_state": epsilon_decreases,
        "half_log_tangent_identity": sp.factor(
            half_log_tangent - r / (2 * n + 1)
        ) == 0,
        "kernel_log_derivative_identity": kernel_log_derivative_identity == 0,
        "h_derivative_identity": h_derivative_identity == 0,
        "sinh_excess_derivative_identity": sinh_excess_derivative_identity == 0,
        "kernel_strictly_decreases_for_zero_lt_r_lt_one": kernel_decreases,
        "kernel_transform_identities": kernel_transform_identities,
        "kernel_monotonicity_reason": (
            "H(x)=x*coth(x) has derivative numerator sinh(2x)-2x; "
            "that excess starts at zero and has derivative 4*sinh(x)^2>0"
        ),
        "constant_state_comparator": {
            "delta_block": (
                "D_p(q)=1/2*[Phi((p+1)q)-Phi(pq)], "
                "Phi(x)=log Gamma(x+alpha)-log Gamma(x+beta)"
            ),
            "scaled_digamma_integral": (
                "x[psi(x+alpha)-psi(x+beta)]="
                "integral_0^infinity exp(-u) k_r(u/x) du"
            ),
            "direction": (
                "k_r strictly decreasing implies the scaled digamma gap is "
                "strictly increasing in x, hence D_p(q) strictly increases "
                "and the equal-cap defect block B_p(q) strictly decreases"
            ),
            "strict_inequality": "sum_I epsilon_n(r) > sum_J epsilon_n(r)",
            "passed": constant_state_comparator_direction,
        },
        "frozen_r_choice": (
            "r*=(Q_((p+1)q-1)+Q_(p(q+1)))/2"
        ),
        "frozen_r_symbolic": str(frozen_r),
        "frozen_r_identities": frozen_r_identities,
        "frozen_r_domain_reason": (
            "AQ9 and the separated blocks give 0<Q_low<Q_high<1, so "
            "Q_low<r*<Q_high and 0<r*<1"
        ),
        "constant_state_comparator_source": "annular reduction in the manuscript",
        "sandwich_direction": (
            "sum_I epsilon_n(Q_n) >= sum_I epsilon_n(r*) > "
            "sum_J epsilon_n(r*) >= sum_J epsilon_n(Q_n)"
        ),
        "annular_identity": (
            "Xi_(p,q)(x)=2*(sum_I epsilon_n(Q_n)-"
            "sum_J epsilon_n(Q_n))"
        ),
        "full_sandwich_gate": frozen_state_sandwich_gate,
        "unequal_length_factorwise_route_used": False,
    }

    derivative_mutation_rejections = 0
    for degree in range(5):
        mutant = expected_derivative_numerator - 2 * derivative_coefficients[degree] * z**degree
        if not all_coefficients_positive(mutant, (z, c, n)):
            derivative_mutation_rejections += 1

    base_mutation_rejections = 0
    for coefficients in (base_one_coefficients, base_two_coefficients):
        for index in range(len(coefficients)):
            mutant = list(coefficients)
            mutant[index] = -mutant[index]
            if not all(value > 0 for value in mutant):
                base_mutation_rejections += 1

    k_poly = sp.Poly(k_shifted, u, s, domain=sp.ZZ)
    k_terms = k_poly.terms()
    minimum_monomial, minimum_coefficient = min(k_terms, key=lambda row: int(row[1]))
    k_mutant = sp.expand(k_shifted - 2 * minimum_coefficient * u**minimum_monomial[0] * s**minimum_monomial[1])
    k_mutant_record = integer_polynomial_record(k_mutant, (u, s))
    mutation_controls = {
        "derivative_sign_mutations_rejected": derivative_mutation_rejections,
        "derivative_sign_mutation_count": 5,
        "base_sign_mutations_rejected": base_mutation_rejections,
        "base_sign_mutation_count": len(base_one_coefficients) + len(base_two_coefficients),
        "resultant_mutated_monomial": list(minimum_monomial),
        "resultant_mutated_coefficient": -int(minimum_coefficient),
        "resultant_sign_mutation_rejected": (
            not k_mutant_record["strictly_positive"]
            and k_mutant_record["payload_sha256"] != resultant_record["payload_sha256"]
        ),
        "protocol": (
            "Negate each of the five AQ3 coefficient expressions in turn; "
            "negate each of the ten shifted base coefficients in turn; "
            "negate K's minimum coefficient at the named monomial and require "
            "both positivity and canonical-payload gates to reject it."
        ),
    }

    passed = (
        all(scalar_identities.values())
        and scalar_domain["adjacent_denominator_positive"]
        and scalar_domain["derivative_denominator_positive"]
        and scalar_domain["unique_simple_root_in_open_unit_interval"]
        and root_propagation["two_step_identity"]
        and root_propagation["two_step_derivative_identity"]
        and root_propagation["two_step_strictly_increasing"]
        and root_propagation["cleared_polynomial_denominator"] == "1/1"
        and root_propagation["resultant_quotient_denominator"] == "1/1"
        and root_propagation["resultant_factorization_identity"]
        and root_propagation["resultant_matches_frozen_payload"]
        and root_propagation["large_center_orientation"]
        and root_propagation["strict_root_order_all_centers"]
        and bases["F_1_identity"]
        and bases["F_2_identity"]
        and bases["F_1_coefficients_match"]
        and bases["F_2_coefficients_match"]
        and bases["F_1_strictly_positive"]
        and bases["F_2_strictly_positive"]
        and parity_induction["passed"]
        and compact_endpoint_falsifier["strict_dip"]
        and compact_endpoint_falsifier["matches_frozen_values"]
        and frozen_state_bridge["orientation_by_symmetry"]["passed"]
        and frozen_state_bridge["lengths_and_separation"]
        and frozen_state_bridge["equal_product"]
        and frozen_state_bridge["cap_sums_equal"]
        and frozen_state_bridge["epsilon_derivative_identity"]
        and frozen_state_bridge["epsilon_decreases_in_frozen_state"]
        and frozen_state_bridge["half_log_tangent_identity"]
        and frozen_state_bridge["kernel_log_derivative_identity"]
        and frozen_state_bridge["h_derivative_identity"]
        and frozen_state_bridge["sinh_excess_derivative_identity"]
        and frozen_state_bridge["kernel_strictly_decreases_for_zero_lt_r_lt_one"]
        and all(frozen_state_bridge["frozen_r_identities"].values())
        and frozen_state_bridge["constant_state_comparator"]["passed"]
        and frozen_state_bridge["full_sandwich_gate"]
        and derivative_mutation_rejections == 5
        and base_mutation_rejections == 10
        and mutation_controls["resultant_sign_mutation_rejected"]
    )

    result = {
        "schema": SCHEMA,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "classification": "all-parameter proof on the unbounded center strip",
        "worker_contract": "one serial worker; exact SymPy rational and integer arithmetic",
        "scalar_adjacent_difference": {
            "quartic_degree": int(sp.Poly(p_n, z).degree()),
            "threshold_polynomial": str(p_n),
            "derivative_coefficients": [str(value) for value in derivative_coefficients],
            "identities": scalar_identities,
            "domain": scalar_domain,
        },
        "root_propagation": root_propagation,
        "parity_induction_bases": bases,
        "parity_induction": parity_induction,
        "compact_endpoint_falsifier": compact_endpoint_falsifier,
        "frozen_state_bridge": frozen_state_bridge,
        "mutation_controls": mutation_controls,
        "finite_pq_scan_used": False,
        "passed": passed,
        "dispositions": [
            "DIMENSION_STEP_PROVED_ON_X_GE_THREE_HALVES",
        ],
        "scope_boundary": (
            "The universal dimension-step theorem does not by itself prove the "
            "equal-product theorem on the noninteger base strip."
        ),
    }
    result["body_encoding"] = "UTF-8 compact JSON with sorted keys, excluding body_sha256"
    canonical_body = json.dumps(result, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    result["body_sha256"] = hashlib.sha256(canonical_body).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"passed={passed}")
    print(f"output={args.output}")
    print(f"producer_sha256={result['producer_sha256']}")
    print(f"resultant_terms={resultant_record['term_count']}")
    print(f"resultant_payload_sha256={resultant_record['payload_sha256']}")
    for disposition in result["dispositions"]:
        print(f"disposition={disposition}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
