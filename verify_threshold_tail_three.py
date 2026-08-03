#!/usr/bin/env python3
"""Exact verifier for the third-threshold-tail material.

Standard library only, so this runs independently of the SymPy suite.

Coverage, in the order the manuscript states it:

  * the second factorial moments (eq:low-moments) against the parity
    recurrences (eq:moment-recurrence);
  * the logarithmic identity (eq:delta-two-log) and (eq:nabla-psi);
  * the elementary masses (eq:cd-masses) and comparisons (eq:cd-compare);
  * both telescopes (eq:telescope-odd), (eq:telescope-even) against a direct
    evaluation of nabla rho, and the absorbing atom (eq:atom-absorb) with its
    cleared all-index certificate;
  * Theorem "second-log-derivative sign" including (eq:delta-two-uniform);
  * the quantitative endpoint theorem: the 41-coefficient smooth-drop
    certificate, independent 68- and 29-coefficient parity-margin
    certificates, an exact 1,770-cell replay, and the exact p=2 limit 4/3;
  * the outer fences and the finite rectangle of the third-tail theorem,
    its 2,685-cell residual, and U_3 > 0 on every residual cell;

Exits nonzero on any failure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, reduce
from math import comb, gcd, lcm, log
import sys


FAILURES: list[str] = []
CHECKS = 0


def check(label: str, ok: bool) -> None:
    global CHECKS
    CHECKS += 1
    if not ok:
        FAILURES.append(label)
        print(f"[FAIL] {label}")


def report(label: str, value) -> None:
    print(f"  {label:<52} {value}")


# ------------------------------------------------------------------ corners

def corners(p: int, q: int):
    L = ((p + 1) * (q + 1), p * q, p * (q + 3), (p + 1) * (q + 2))
    R = (p * (q + 1), q * (p + 1), (p + 1) * (q + 3), p * (q + 2))
    return L, R


@cache
def q4(n: int) -> Fraction:
    """Q_n(4) in closed form."""
    m = n // 2
    c = Fraction(comb(2 * m, m), 4 ** m)
    return ((4 * m + 1) if n % 2 == 0 else (2 * m + 1)) * c


def ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


# ------------------------------------- independent sparse rational algebra

def _clean2(terms):
    return {monomial: Fraction(value) for monomial, value in terms.items() if value}


@dataclass(frozen=True)
class Poly2:
    """A rational polynomial in two variables, stored sparsely."""

    terms: dict[tuple[int, int], Fraction]

    def __post_init__(self):
        object.__setattr__(self, "terms", _clean2(self.terms))

    @staticmethod
    def coerce(value):
        if isinstance(value, Poly2):
            return value
        return Poly2({(0, 0): Fraction(value)})

    def __add__(self, other):
        other = self.coerce(other)
        result = dict(self.terms)
        for monomial, value in other.terms.items():
            result[monomial] = result.get(monomial, Fraction(0)) + value
        return Poly2(result)

    __radd__ = __add__

    def __neg__(self):
        return Poly2({monomial: -value for monomial, value in self.terms.items()})

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        result = {}
        for (i, j), left in self.terms.items():
            for (r, s), right in other.terms.items():
                monomial = (i + r, j + s)
                result[monomial] = result.get(monomial, Fraction(0)) + left * right
        return Poly2(result)

    __rmul__ = __mul__

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("polynomial exponent must be a nonnegative integer")
        result = Poly2.coerce(1)
        base = self
        while exponent:
            if exponent & 1:
                result *= base
            base *= base
            exponent //= 2
        return result

    def shift(self, p0, q0):
        """Return f(r+p0,s+q0), still indexed by the formal variables."""

        r = Poly2({(1, 0): 1})
        s = Poly2({(0, 1): 1})
        result = Poly2.coerce(0)
        for (i, j), value in self.terms.items():
            result += value * (r + p0) ** i * (s + q0) ** j
        return result


@dataclass(frozen=True)
class RationalFunction:
    num: Poly2
    den: Poly2

    @staticmethod
    def coerce(value):
        if isinstance(value, RationalFunction):
            return value
        return RationalFunction(Poly2.coerce(value), Poly2.coerce(1))

    def __add__(self, other):
        other = self.coerce(other)
        return RationalFunction(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )

    __radd__ = __add__

    def __neg__(self):
        return RationalFunction(-self.num, self.den)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        other = self.coerce(other)
        return RationalFunction(self.num * other.num, self.den * other.den)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.coerce(other)
        return RationalFunction(self.num * other.den, self.den * other.num)

    def __rtruediv__(self, other):
        return self.coerce(other) / self

    def equal(self, other):
        other = self.coerce(other)
        return self.num * other.den == other.num * self.den


def rat(num, den=1):
    return RationalFunction(Poly2.coerce(num), Poly2.coerce(den))


def primitive_summary(polynomial, p0, q0):
    shifted = polynomial.shift(p0, q0)
    ordered = [shifted.terms[key] for key in sorted(shifted.terms, reverse=True)]
    common_denominator = reduce(lcm, (value.denominator for value in ordered), 1)
    integers = [
        value.numerator * (common_denominator // value.denominator)
        for value in ordered
    ]
    common_factor = reduce(gcd, integers)
    primitive = [value // common_factor for value in integers]
    payload = ",".join(map(str, primitive)).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return len(primitive), min(primitive), max(primitive), digest


# ------------------------------------------------------------------ moments

def moments_recurrence(N: int):
    """zeta_n, xi_n by the parity recurrences."""
    z = [Fraction(0)] * (N + 1)
    x = [Fraction(0)] * (N + 1)
    for m in range(1, N // 2 + 1):
        n = 2 * m
        z[n] = (z[n - 1] + 1 + (n - 1) * z[n - 2]) / n
        x[n] = (x[n - 1] + z[n - 1] + (n - 1) * x[n - 2]) / n
        if n + 1 <= N:
            z[n + 1] = (z[n] + n * z[n - 1]) / (n + 1)
            x[n + 1] = (x[n] + n * x[n - 1]) / (n + 1)
    return z, x


def moments_closed(N: int):
    """sigma_n, upsilon_n, zeta_n, xi_n, varrho_n by the closed forms."""
    sig = [Fraction(0)] * (N + 1)
    ups = [Fraction(0)] * (N + 1)
    odd_h = Fraction(0)
    running = Fraction(0)
    for n in range(1, N + 1):
        if n & 1:
            odd_h += Fraction(1, n)
        sig[n] = odd_h
        if not (n & 1):
            running += odd_h / (n // 2)
        ups[n] = running
    z, x, rho = [], [], []
    for n in range(N + 1):
        e = n & 1
        z.append((sig[n] - e) / 2)
        x.append((ups[n] - (1 + 2 * e) * sig[n] + 3 * e) / 8)
        rho.append((sig[n] * sig[n] - ups[n]) / 4)
    return sig, ups, z, x, rho


def product_second(indices, zeta, xi):
    """The y^2 coefficient of a four-factor product."""

    return sum((xi[n] for n in indices), Fraction(0)) + sum(
        zeta[indices[i]] * zeta[indices[j]]
        for i in range(4)
        for j in range(i + 1, 4)
    )


def masses(limit: int):
    """The elementary masses c_m, d_m and the odd-harmonic sigma_{2m}."""
    s = [Fraction(0)] * (limit + 2)
    c = [Fraction(0)] * (limit + 2)
    d = [Fraction(0)] * (limit + 2)
    for m in range(1, limit + 2):
        s[m] = s[m - 1] + Fraction(1, 2 * m - 1)
        c[m] = (s[m] - Fraction(m, 2 * m - 1)) / (4 * m * (2 * m - 1))
        d[m] = (s[m] - Fraction(m, 2 * m + 1)) / (4 * m * (2 * m + 1))
    return s, c, d


# --------------------------------------------------- the cleared atom certificate

def _padd(f, g, scale=1):
    h = dict(f)
    for mon, val in g.items():
        h[mon] = h.get(mon, 0) + scale * val
        if not h[mon]:
            del h[mon]
    return h


def _pmul(f, g):
    h: dict[tuple[int, int, int], int] = {}
    for (i, j, k), a in f.items():
        for (r, s, t), b in g.items():
            mon = (i + r, j + s, k + t)
            h[mon] = h.get(mon, 0) + a * b
    return {mon: val for mon, val in h.items() if val}


def _pscale(f, c):
    return {mon: c * val for mon, val in f.items() if c * val}


def atom_certificate():
    """Cleared form of c_{m+P}+c_{m+Q} > c_m at m = 2PQ+j, P=A+1, Q=B+1, j=X+1."""
    ONE = {(0, 0, 0): 1}
    A = {(1, 0, 0): 1}
    B = {(0, 1, 0): 1}
    X = {(0, 0, 1): 1}
    P = _padd(A, ONE)
    Q = _padd(B, ONE)
    j = _padd(X, ONE)
    m = _padd(_pscale(_pmul(P, Q), 2), j)

    def twice_minus_one(f):
        return _padd(_pscale(f, 2), ONE, scale=-1)

    def f(t):
        mt = _padd(m, t)
        return _pmul(mt, twice_minus_one(mt))

    base = _pmul(m, twice_minus_one(m))
    return _padd(_pmul(base, _padd(f(P), f(Q))), _pmul(f(P), f(Q)), scale=-1)


# --------------------------------------- quantitative endpoint certificates

def _univariate_degree_lc(polynomial: Poly2):
    check("asymptotic polynomial is univariate", all(j == 0 for _, j in polynomial.terms))
    degree = max(i for i, _ in polynomial.terms)
    return degree, polynomial.terms[(degree, 0)]


def endpoint_symbolic_certificates():
    """Reconstruct every all-index gate in the endpoint-margin theorem."""

    p = Poly2({(1, 0): 1})
    q = Poly2({(0, 1): 1})
    B = p * q
    C = p * (q + 1)
    D = (p + 1) * q

    X = 2 * rat(C + Fraction(1, 2)) * rat(D + Fraction(1, 2))
    Y = 2 * rat(p * (q + 3) + Fraction(1, 2)) * rat(
        (p + 1) * (q + 2) + Fraction(1, 2)
    )
    d = Y - X
    d_closed = 8 * p * (p + 1) * q + 12 * p**2 + 16 * p + 2
    check("endpoint d = Y-X closed form", d.equal(d_closed))

    def smooth(row):
        b = p * row
        c = p * (row + 1)
        d0 = (p + 1) * row
        a = (p + 1) * (row + 1)
        return rat((2 * a + 1) * (2 * b + 1), (2 * c + 1) * (2 * d0 + 1))

    n = B + 1
    shifted_n = n + 2 * p
    exp_eta = (1 - 1 / rat(4 * shifted_n**2)) / (1 - 1 / rat(4 * n**2))
    smooth_difference = smooth(q) / smooth(q + 2) - exp_eta
    smooth_summary = primitive_summary(smooth_difference.num, 1, 2)
    check(
        "smooth-drop certificate summary",
        smooth_summary
        == (
            41,
            16,
            101_008,
            "e340cc6b97142dc696970ec56f4220e891db333fdcaab5993dfcbf994c939041",
        ),
    )
    check(
        "smooth-drop denominator is positive",
        all(value > 0 for value in smooth_difference.den.shift(1, 2).terms.values()),
    )

    target = d / (Y * (Y + 1))
    common_q = (q + 1) ** 2 * (q + 3) ** 2
    m_p = (q + 2) / rat(p**2 * common_q)
    m_p1 = (q + 2) / rat((p + 1) ** 2 * common_q)
    even_difference = m_p + m_p1 - target
    odd_difference = 2 * m_p - target
    even_summary = primitive_summary(even_difference.num, 2, 2)
    odd_summary = primitive_summary(odd_difference.num, 1, 3)
    check(
        "even endpoint-margin certificate summary",
        even_summary
        == (
            68,
            16,
            8_939_748_630,
            "46399ea23ca1e9fc163202c5a6b3ed96970d69fbb02343ae2ca75212fb3e25fa",
        ),
    )
    check(
        "odd endpoint-margin certificate summary",
        odd_summary
        == (
            29,
            16,
            737_944,
            "c077aedbcfb5940fa0e3dd90ec909a976fb18a7521a3538d93e1eabb3e76a2c6",
        ),
    )
    check(
        "even endpoint-margin denominator is positive",
        all(value > 0 for value in even_difference.den.shift(2, 2).terms.values()),
    )
    check(
        "odd endpoint-margin denominator is positive",
        all(value > 0 for value in odd_difference.den.shift(1, 3).terms.values()),
    )

    check(
        "(q+2)^3/(q+1)^2 > q+3 clearing",
        (q + 2) ** 3 - (q + 3) * (q + 1) ** 2 == q**2 + 5 * q + 5,
    )
    check(
        "even factored endpoint surplus",
        4 * (p**2 + (p + 1) ** 2) * (q + 3) - d_closed
        == 4 * q + 12 * p**2 + 8 * p + 10,
    )
    check(
        "odd factored endpoint surplus",
        8 * (p + 1) ** 2 * (q + 3) - d_closed
        == 8 * (p + 1) * q + 12 * p**2 + 32 * p + 22,
    )
    y_floor = 2 * p * (p + 1) * (q + 3) * (q + 2)
    check(
        "Y > 2p(p+1)(q+2)(q+3)",
        all(value > 0 for value in (Y - y_floor).num.shift(1, 2).terms.values()),
    )
    first_atom = Fraction(1, 4) * (
        1 / rat(C**2) - 1 / rat((C + 2 * p) ** 2)
    )
    check("first alpha-series atom equals m_p(q)", first_atom.equal(m_p))

    # Exact p=2, q=2r cancellation and asymptotic margin.
    r = Poly2({(1, 0): 1})
    rho_p2 = rat(
        (2 * r + 1) ** 2
        * (3 * r + 4)
        * (4 * r + 5)
        * (6 * r + 1)
        * (8 * r + 1)
        * (8 * r + 13)
        * (12 * r + 13),
        (2 * r + 3) ** 2
        * (3 * r + 1)
        * (4 * r + 1)
        * (6 * r + 7)
        * (8 * r + 5)
        * (8 * r + 9)
        * (12 * r + 1),
    )
    X2 = rat((8 * r + 5) * (12 * r + 1), 2)
    Y2 = rat((8 * r + 13) * (12 * r + 13), 2)
    margin_p2 = (rho_p2 - 1) / ((Y2 - X2) / (2 * Y2 * (Y2 + 1)))
    P2 = (
        (8 * r + 13)
        * (12 * r + 13)
        * (12 * r + 19)
        * (
            6144 * r**5
            + 22208 * r**4
            + 29232 * r**3
            + 16748 * r**2
            + 4148 * r
            + 545
        )
    )
    Q2 = (
        4
        * (2 * r + 3) ** 2
        * (3 * r + 1)
        * (4 * r + 1)
        * (6 * r + 7)
        * (8 * r + 5)
        * (12 * r + 1)
        * (48 * r + 41)
    )
    check("p=2 endpoint-margin closed form", margin_p2.equal(rat(P2, Q2)))
    p2_num_degree, p2_num_lc = _univariate_degree_lc(P2)
    p2_den_degree, p2_den_lc = _univariate_degree_lc(Q2)
    check(
        "p=2 endpoint-margin exact limit is 4/3",
        p2_num_degree == p2_den_degree and 3 * p2_num_lc == 4 * p2_den_lc,
    )
    margin_surplus = margin_p2 - Fraction(4, 3)
    check(
        "p=2 endpoint-margin exceeds its limit",
        all(value > 0 for value in margin_surplus.num.shift(1, 0).terms.values())
        and all(value > 0 for value in margin_surplus.den.shift(1, 0).terms.values()),
    )

    return smooth_summary, even_summary, odd_summary


def endpoint_quantities(p: int, q: int):
    xn = (2 * p * (q + 1) + 1) * (2 * q * (p + 1) + 1)
    yn = (2 * p * (q + 3) + 1) * (2 * (p + 1) * (q + 2) + 1)
    X = Fraction(xn, 2)
    Y = Fraction(yn, 2)
    target = (Y - X) / (Y * (Y + 1))
    common_q = (q + 1) ** 2 * (q + 3) ** 2
    m_p = Fraction(q + 2, p**2 * common_q)
    m_p1 = Fraction(q + 2, (p + 1) ** 2 * common_q)
    lower = 2 * m_p if p & 1 else m_p + m_p1
    return target, lower


def endpoint_ratio(p: int, q: int) -> Fraction:
    L, R = corners(p, q)
    left = right = Fraction(1)
    for n in L:
        left *= q4(n)
    for n in R:
        right *= q4(n)
    return left / right


# ------------------------------------------------------------------ the run

def main() -> int:
    print("threshold tail three — exact verifier")
    print()

    # ---- 1. the atom certificate --------------------------------------
    cert = atom_certificate()
    check("atom certificate has 55 terms", len(cert) == 55)
    check("atom certificate is strictly positive", all(v > 0 for v in cert.values()))
    check("atom certificate minimum coefficient is 4", min(cert.values()) == 4)
    g = 0
    for v in cert.values():
        g = gcd(g, v)
    check("atom certificate gcd is 1", g == 1)
    print("[1] cleared atom certificate")
    report("terms / min coefficient / gcd",
           f"{len(cert)} / {min(cert.values())} / {g}")

    # ---- 2. moments and the logarithmic identity ----------------------
    PMAX = QMAX = 40
    N = (PMAX + 1) * (QMAX + 3)
    zr, xr = moments_recurrence(N)
    sig, ups, zc, xc, rho = moments_closed(N)
    check("closed-form zeta matches the recurrence",
          all(zr[n] == zc[n] for n in range(N + 1)))
    check("closed-form xi matches the recurrence",
          all(xr[n] == xc[n] for n in range(N + 1)))
    psi = [zc[n] * zc[n] - 2 * xc[n] for n in range(N + 1)]
    check("psi_n = sigma_n/4 + varrho_n - eps_n/2",
          all(psi[n] == sig[n] / 4 + rho[n] - Fraction(n & 1, 2)
              for n in range(N + 1)))
    print(f"[2] second factorial moments, n <= {N}")

    # ---- 3. per-cell identities and bounds ----------------------------
    mass_limit = max(N, 2 * 24 * 24 + 2 * 24 + 24)
    s, cm, dm = masses(mass_limit)
    check("c_1 = 0 and c_2 = 1/36", cm[1] == 0 and cm[2] == Fraction(1, 36))
    check("c_m > c_{m+1} for m >= 2",
          all(cm[m] > cm[m + 1] for m in range(2, mass_limit)))
    check("d_m > c_{m+1} for m >= 1",
          all(dm[m] > cm[m + 1] for m in range(1, mass_limit)))
    check("d_m - c_{m+1} closed form",
          all(dm[m] - cm[m + 1]
              == (s[m] - Fraction(m, 2 * m + 1)) / (4 * m * (m + 1) * (2 * m + 1))
              for m in range(1, mass_limit)))

    cells = 0
    for p in range(1, PMAX + 1):
        for q in range(2, QMAX + 1):
            if (p - q) & 1:
                continue
            L, R = corners(p, q)

            def nab(f, L=L, R=R):
                return sum((f[n] for n in L), Fraction(0)) - sum(
                    (f[n] for n in R), Fraction(0)
                )

            d1 = nab(zc)
            d2 = product_second(L, zc, xc) - product_second(R, zc, xc)
            Sigma = sum((zc[n] for n in L + R), Fraction(0))

            check(f"one odd index on each side ({p},{q})",
                  sum(n & 1 for n in L) == 1 and sum(n & 1 for n in R) == 1)
            check(f"delta_2 = (Sigma delta_1 - nabla psi)/2 ({p},{q})",
                  d2 == (Sigma * d1 - nab(psi)) / 2)
            check(f"nabla psi = delta_1/2 + nabla varrho ({p},{q})",
                  nab(psi) == d1 / 2 + nab(rho))
            check(f"nabla varrho > 0 ({p},{q})", nab(rho) > 0)

            # the two telescopes
            if p & 1:
                b = (p * q - 1) // 2
                cc = p * (q + 1) // 2
                dd = q * (p + 1) // 2
                a = (p + 1) * (q + 1) // 2
                tel = (
                    sum(cm[cc + j] for j in range(1, p + 1))
                    + sum(cm[dd + j] for j in range(1, p + 2))
                    + sum(dm[b + j] for j in range(1, p + 1))
                    - sum(cm[a + j] for j in range(1, p + 2))
                )
            else:
                P, Q = p // 2, q // 2
                b = 2 * P * Q
                tel = (
                    -sum(cm[b + j] for j in range(1, 2 * P + 1))
                    + sum(cm[b + P + j] for j in range(1, 2 * P + 1))
                    + sum(cm[b + Q + j] for j in range(1, 2 * P + 2))
                    + sum(dm[b + P + Q + j] for j in range(1, 2 * P + 2))
                )
            check(f"telescope reproduces nabla varrho ({p},{q})", tel == nab(rho))

            # the theorem's two bounds
            check(f"delta_2 < (Sigma-1/2) delta_1 / 2 ({p},{q})",
                  d2 < (Sigma - Fraction(1, 2)) * d1 / 2)
            Nmax = (p + 1) * (q + 3)
            check(f"Sigma <= 3 + 2 log N ({p},{q})",
                  float(Sigma) <= 3 + 2 * log(Nmax) + 1e-12)
            check(f"delta_1 < 13/(12 p q^2) ({p},{q})",
                  d1 < Fraction(13, 12 * p * q * q))
            check(f"delta_2 < (log N + 5/4) delta_1 ({p},{q})",
                  float(d2) <= (log(Nmax) + 1.25) * float(d1) + 1e-15)
            cells += 1
    print(f"[3] per-cell identities and bounds on {cells} same-parity cells "
          f"p <= {PMAX}, q <= {QMAX}")

    # ---- 3b. the three displayed polynomial identities -----------------
    ONE = {(0, 0, 0): 1}
    v0 = {(1, 0, 0): 1}   # m, or t
    v1 = {(0, 1, 0): 1}   # B

    def lin(poly, a, b):
        """a*poly + b."""
        return _padd(_pscale(poly, a), _pscale(ONE, b))

    # (i) clearing c_m - c_{m+1} under sigma_{2m} >= (3m-2)/(2m-1)
    lhs = _padd(
        _padd(
            _pmul(_pmul(lin(v0, 3, -2), lin(v0, 4, 1)), lin(v0, 2, 1)),
            _pmul(_pmul(v0, lin(v0, 1, 1)), _pmul(lin(v0, 2, 1), lin(v0, 2, 1))),
            scale=-1,
        ),
        _pmul(_pmul(v0, v0), _pmul(lin(v0, 2, -1), lin(v0, 2, -1))),
    )
    rhs = _pscale({(3, 0, 0): 6, (2, 0, 0): -1, (1, 0, 0): -5, (0, 0, 0): -1}, 2)
    check("cleared c_m - c_{m+1} numerator is 2(6m^3-m^2-5m-1)", lhs == rhs)

    # (ii) m^2 - m(P+Q) - 3PQ at m = 2PQ+1, P = A+1, Q = B+1
    Pv, Qv = lin(v0, 1, 1), lin(v1, 1, 1)
    mv = _padd(_pscale(_pmul(Pv, Qv), 2), ONE)
    quad = _padd(
        _padd(_pmul(mv, mv), _pmul(mv, _padd(Pv, Qv)), scale=-1),
        _pscale(_pmul(Pv, Qv), 3),
        scale=-1,
    )
    expected_quad = {
        (1, 0, 0): 2, (2, 0, 0): 2, (0, 1, 0): 2, (0, 2, 0): 2,
        (1, 1, 0): 9, (2, 1, 0): 6, (1, 2, 0): 6, (2, 2, 0): 4,
    }
    check("quadratic at m=2PQ+1 expands to the displayed positive form",
          quad == expected_quad)
    check("that expansion is coefficientwise nonnegative",
          all(v > 0 for v in quad.values()))

    # (iii) 90 q^4 - 8(q+3)(q+4)^2 at q = t+2
    qv = lin(v0, 1, 2)
    diff = _padd(
        _pscale(_pmul(_pmul(qv, qv), _pmul(qv, qv)), 90),
        _pmul(lin(v0, 1, 5), _pmul(lin(v0, 1, 6), lin(v0, 1, 6))),
        scale=-8,
    )
    check("90 q^4 - 8(q+3)(q+4)^2 at q=t+2 is 90t^4+712t^3+2024t^2+2112t",
          diff == {(4, 0, 0): 90, (3, 0, 0): 712, (2, 0, 0): 2024, (1, 0, 0): 2112})
    print("[3b] the three displayed polynomial identities of the proof")

    # ---- 4. the absorbing atom on an exact grid ------------------------
    atom_cells = 0
    for P in range(1, 25):
        for Q in range(1, 25):
            for j in range(1, 2 * P + 1):
                m = 2 * P * Q + j
                lhs = Fraction(1, (m + P) * (2 * m + 2 * P - 1)) + Fraction(
                    1, (m + Q) * (2 * m + 2 * Q - 1)
                )
                check("atom inequality", lhs > Fraction(1, m * (2 * m - 1)))
                check("atom absorbs", cm[m + P] + cm[m + Q] > cm[m])
                atom_cells += 1
    print(f"[4] absorbing atom on {atom_cells} exact (P,Q,j) triples")

    # ---- 5. quantitative endpoint theorem -----------------------------
    smooth_summary, even_summary, odd_summary = endpoint_symbolic_certificates()
    endpoint_cells = 0
    best = None
    for p in range(1, 61):
        for q in range(2, 61):
            if (p - q) & 1:
                continue
            endpoint_cells += 1
            target, lower = endpoint_quantities(p, q)
            check(f"endpoint rational lower bound ({p},{q})", lower > target)
            endpoint = endpoint_ratio(p, q)
            check(f"quantitative endpoint gap ({p},{q})", endpoint - 1 > target / 2)
            margin = (endpoint - 1) / (target / 2)
            if best is None or margin < best[0]:
                best = (margin, p, q)
    check("endpoint replay holds 1,770 cells", endpoint_cells == 1770)
    check("endpoint replay minimum is at (2,60)", best[1:] == (2, 60))
    check(
        "endpoint replay minimum begins 1.4040357648",
        Fraction(14_040_357_648, 10_000_000_000)
        < best[0]
        < Fraction(14_040_357_649, 10_000_000_000),
    )
    print("[5] quantitative endpoint theorem — analytic gates before residual")
    report("smooth / even / odd certificate terms", "41 / 68 / 29")
    report("certificate summaries", (smooth_summary, even_summary, odd_summary))
    report("exact endpoint replay", f"{endpoint_cells:,} cells")
    report(
        "minimum (rho-1)/target at (2,60)",
        f"{float(best[0]):.10f}",
    )
    report("p=2 exact asymptotic limit", "4/3")

    # ---- 6. outer fences ----------------------------------------------
    check("q >= 10^4 constant fence",
          Fraction(48009, 4800) < 100 * Fraction(10 ** 12, 8 * 10003 * 10004 ** 2))
    check("p >= 32768 fence value",
          Fraction(13, 12) * (Fraction(453, 4 * 32768) + Fraction(9, 2048))
          == Fraction(4459, 524288))
    check("p >= 32768 fence is below 1/90", Fraction(4459, 524288) < Fraction(1, 90))
    check("q^4 fence at q = 2",
          Fraction(2 ** 4, 8 * 5 * 6 ** 2) == Fraction(1, 90))
    for q in range(1071, 10000):
        Kq = ceil_log2(q + 3)
        check("one-variable outer comparison",
              Fraction(q * q, 8 * (q + 3) * (q + 4) ** 2)
              > Fraction(13, 12 * q * q) * (Fraction(93, 4) + 9 * Kq))
    print("[6] outer fences, including 1071 <= q < 10000 exactly")

    # ---- 7. the finite rectangle and its residual ----------------------
    def tail_proves(p: int, q: int) -> bool:
        K = ceil_log2((p + 1) * (q + 3))
        xn = (2 * p * (q + 1) + 1) * (2 * q * (p + 1) + 1)
        yn = (2 * p * (q + 3) + 1) * (2 * (p + 1) * (q + 2) + 1)
        return 12 * p * q * q * (p * (p + 1) * q * (q + 3) * (yn - xn)) > 13 * (
            57 + 36 * K
        ) * yn * (yn + 2)

    residual = []
    rectangle = 0
    for q in range(2, 1071):
        p0 = 2 if q % 2 == 0 else 1
        for p in range(p0, 32768, 2):
            rectangle += 1
            if not tail_proves(p, q):
                residual.append((p, q))
    check("rectangle holds 17,513,961 cells", rectangle == 17_513_961)
    check("residual holds 2,685 cells", len(residual) == 2685)
    check("residual p <= 1736", max(p for p, _ in residual) == 1736)
    check("residual q <= 457", max(q for _, q in residual) == 457)
    max_n = max(max(corners(p, q)[0] + corners(p, q)[1]) for p, q in residual)
    check("residual max index 8685", max_n == 8685)
    print(f"[7] finite rectangle {rectangle:,} cells, residual {len(residual):,}, "
          f"max index {max_n}")

    # ---- 8. U_3 > 0 on every residual cell ----------------------------
    zres, xres = moments_recurrence(max_n)
    worst = None
    fails = 0
    for p, q in residual:
        L, R = corners(p, q)
        d1 = sum((zres[n] for n in L), Fraction(0)) - sum(
            (zres[n] for n in R), Fraction(0)
        )
        d2 = product_second(L, zres, xres) - product_second(R, zres, xres)
        left = right = Fraction(1)
        for n in L:
            left *= q4(n)
        for n in R:
            right *= q4(n)
        u2 = left - right - 3 * d1
        u3 = u2 - 9 * d2
        if u3 <= 0:
            fails += 1
        ratio = u2 / (9 * d2) if d2 > 0 else None
        if ratio is not None and (worst is None or ratio < worst[0]):
            worst = (ratio, p, q, d2, u3)
    check("no residual cell fails U_3 > 0", fails == 0)
    check("worst residual ratio cell is (1,3)", worst[1:3] == (1, 3))
    check("delta_2(1,3)", worst[3] == Fraction(23599, 582120))
    check("U_3(1,3)", worst[4] == Fraction(1020641999, 1059717120))
    check("worst residual ratio value",
          worst[0] == Fraction(1407288015, 386646016))
    L22, R22 = corners(2, 2)
    check(
        "delta_2(2,2)",
        product_second(L22, zres, xres) - product_second(R22, zres, xres)
        == Fraction(558067, 37837800),
    )
    print("[8] U_3 > 0 on all 2,685 residual cells, 0 exceptions")
    report("min U_2/(9 delta_2) at (1,3)", f"{worst[0]} = {float(worst[0]):.10f}")

    # ---- verdict -------------------------------------------------------
    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} check(s) did not reproduce")
        for item in FAILURES[:20]:
            print(f"  - {item}")
        return 1
    print(f"PASS: every third-tail check reproduced exactly ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
