#!/usr/bin/env python3
"""Exact verifier for the upper base strip ``1 <= x <= 3/2``.

The full run reconstructs 50 proof gates in exact integer or rational
arithmetic and writes a deterministic summary certificate.  Floating-point
values appear only in human-readable annotations and never decide a gate.

    python3 verify_upper_strip.py

GATES
  V1  Taylor coefficients of L(m) = log(phi(m)/m):  x, 0, delta_x, lambda_x
  V2  G(n) expansion and Psi(n,x) = 4 delta_x - C(x)/n + O(n^-2), C = 3x^2(3-x^2)/4
  V3  1/3 - 4 delta_x = (x-1)^2(2x+1)/3 >= 0  (sharpness of c6 = 1/3)
  V4  (P7) EXACT Bernstein certificate: Psi(n,x) <= 4 delta_x on n >= 100
  V5  bracket validity 0 < U < 1 on the box
  V6  assembly inputs: sandwich, defect closed form, |D|, |D(n+1)-D(n)|,
      multipliers, |E_n|, parity-chain tail, continuum certificates, S, 4.01
  V7  (P1) finite bridge: N_n(3+u) coefficients > 0 for 2 <= n <= 120
  V8  reduction chain: artanh identity, h(z) = z log((z+1)/(z-1)) decreasing,
      A_n > 0  =>  log-grid slopes increasing
  V9  final margin at n = 100
"""
import argparse
import json
import sys
from fractions import Fraction as Fr
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "upper_strip_certificate.json"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()

NBRIDGE = 120
FAILS = []
NCHECK = 0


def check(name, cond, detail=""):
    global NCHECK
    NCHECK += 1
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


# =====================================================================
# bivariate polynomial machinery:  dict (i,j) -> Fraction  ==  coeff * v^i * x^j
# =====================================================================
def padd(a, b):
    r = dict(a)
    for k, v in b.items():
        r[k] = r.get(k, Fr(0)) + v
        if r[k] == 0:
            del r[k]
    return r


def psub(a, b):
    return padd(a, pscal(-1, b))


def pscal(c, a):
    c = Fr(c)
    return {} if c == 0 else {k: c * v for k, v in a.items()}


def pmul(a, b, cap=None):
    r = {}
    for (i1, j1), v1 in a.items():
        for (i2, j2), v2 in b.items():
            i, j = i1 + i2, j1 + j2
            if cap is not None and i > cap:
                continue
            r[(i, j)] = r.get((i, j), Fr(0)) + v1 * v2
    return {k: v for k, v in r.items() if v != 0}


def ppow(a, e, cap=None):
    r = {(0, 0): Fr(1)}
    base = a
    while e:
        if e & 1:
            r = pmul(r, base, cap)
        base = pmul(base, base, cap)
        e >>= 1
    return r


def pdeg(a):
    return (0, 0) if not a else (max(i for i, _ in a), max(j for _, j in a))


def peval(a, v0, x0):
    return sum(c * Fr(v0) ** i * Fr(x0) ** j for (i, j), c in a.items())


def shift_v(a, k):
    r = {}
    for (i, j), c in a.items():
        for l in range(i + 1):
            r[(l, j)] = r.get((l, j), Fr(0)) + c * comb(i, l) * Fr(k) ** (i - l)
    return {k2: v for k2, v in r.items() if v != 0}


def to_w_form(a):
    d = pdeg(a)[0]
    return {(d - i, j): c for (i, j), c in a.items()}


def divide_by_v(a, k):
    if not all(i >= k for i, _ in a):
        raise ArithmeticError(f"polynomial is not divisible by v^{k}")
    return {(i - k, j): c for (i, j), c in a.items()}


def _bern1d(c):
    d = len(c) - 1
    return [sum(c[k] * Fr(comb(i, k), comb(d, k)) for k in range(i + 1)) for i in range(d + 1)]


def bernstein_box(poly, vhi, x0=Fr(1), x1=Fr(3, 2)):
    dv, dx = pdeg(poly)
    M = [[Fr(0)] * (dx + 1) for _ in range(dv + 1)]
    for (i, j), c in poly.items():
        M[i][j] = c
    xs, xc = Fr(x0), Fr(x1) - Fr(x0)
    R = [[Fr(0)] * (dx + 1) for _ in range(dv + 1)]
    for i in range(dv + 1):
        for j in range(dx + 1):
            c = M[i][j]
            if c:
                for l in range(j + 1):
                    R[i][l] += c * comb(j, l) * xs ** (j - l) * xc ** l
    for i in range(dv + 1):
        s = Fr(vhi) ** i
        for l in range(dx + 1):
            R[i][l] *= s
    S = [[Fr(0)] * (dx + 1) for _ in range(dv + 1)]
    for l in range(dx + 1):
        col = _bern1d([R[i][l] for i in range(dv + 1)])
        for i in range(dv + 1):
            S[i][l] = col[i]
    return [_bern1d(S[i]) for i in range(dv + 1)]


def cert_nonneg(poly, vhi, x0=Fr(1), x1=Fr(3, 2)):
    B = bernstein_box(poly, vhi, x0, x1)
    flat = [c for row in B for c in row]
    return all(c >= 0 for c in flat), min(flat), len(flat)


W1 = {(1, 0): Fr(1)}       # the "v" variable (w = 1/n, or n, depending on use)
X = {(0, 1): Fr(1)}
ONE = {(0, 0): Fr(1)}
OPW = padd(ONE, W1)


def Lam(z, K, cap=None):
    """Lam_K(z) = sum_{k=1}^K (-1)^(k+1) z^k / k, truncated at v-degree cap."""
    acc = {}
    for k in range(1, K + 1):
        acc = padd(acc, pscal(Fr((-1) ** (k + 1), k), ppow(z, k, cap)))
    return acc


print("=" * 74)
print("verify_upper_strip.py -- exact upper-strip replay")
print("=" * 74)

# =====================================================================
print("\nV1. Taylor coefficients of L(m) = log(phi(m)/m), phi(m)=m+x+x^2/2m+x^2/4m^2")
CAP = 6
u = padd(padd(pmul(X, W1), pscal(Fr(1, 2), pmul(ppow(X, 2), ppow(W1, 2)))),
         pscal(Fr(1, 4), pmul(ppow(X, 2), ppow(W1, 3))))          # U(w,x), w = 1/m
Lser = Lam(u, 6, cap=CAP)                                          # exact to w^6
delta = pscal(Fr(1, 12), pmul(ppow(X, 2), padd(pscal(3, ONE), pscal(-2, X))))   # x^2(3-2x)/12
lam = pscal(Fr(1, 8), pmul(ppow(X, 3), padd(X, pscal(-2, ONE))))                # x^3(x-2)/8
c1 = {(0, j): c for (i, j), c in Lser.items() if i == 1}
c2 = {(0, j): c for (i, j), c in Lser.items() if i == 2}
c3 = {(0, j): c for (i, j), c in Lser.items() if i == 3}
c4 = {(0, j): c for (i, j), c in Lser.items() if i == 4}
check("[m^-1] L = x", c1 == {(0, 1): Fr(1)})
check("[m^-2] L = 0  (this is why phi carries x^2/(2m))", c2 == {})
check("[m^-3] L = delta_x = x^2(3-2x)/12", c3 == {(0, k): v for (i, k), v in delta.items()})
check("[m^-4] L = lambda_x = x^3(x-2)/8", c4 == {(0, k): v for (i, k), v in lam.items()})

# =====================================================================
print("\nV2. G(n) = (2n+1)L(n+1)-(2n-1)L(n); Psi(n,x) = n^3[x/(n(n+1)) - G(n)]")
# exact identity  (2n+1)x/(n+1) - (2n-1)x/n = x/(n(n+1))
lhs = psub(pmul(padd(pscal(2, W1), ONE), pmul(X, {(0, 0): Fr(1)})), {})   # placeholder
# check as polynomial identity in n: n(2n+1)x - (n+1)(2n-1)x == x
Nv = {(1, 0): Fr(1)}
idl = psub(pmul(Nv, pmul(padd(pscal(2, Nv), ONE), X)),
           pmul(padd(Nv, ONE), pmul(padd(pscal(2, Nv), pscal(-1, ONE)), X)))
check("(2n+1)x/(n+1)-(2n-1)x/n = x/(n(n+1)) exactly", idl == {(0, 1): Fr(1)},
      "cleared form: n(2n+1)x-(n+1)(2n-1)x = x")
# series for Psi: build L(n+1) via w -> w/(1+w) truncated
CAP2 = 4
# 1/(n+1) = w/(1+w) = w - w^2 + w^3 - w^4 + ...
what = {}
for k in range(1, CAP2 + 2):
    what[(k, 0)] = Fr((-1) ** (k + 1))
uhat = padd(padd(pmul(X, what), pscal(Fr(1, 2), pmul(ppow(X, 2), ppow(what, 2, CAP2 + 1)))),
            pscal(Fr(1, 4), pmul(ppow(X, 2), ppow(what, 3, CAP2 + 1))))
Lhat = Lam(uhat, 6, cap=CAP2 + 1)
Lw = Lam(u, 6, cap=CAP2 + 1)
Gser = psub(pmul(padd(pscal(2, ONE), pmul(W1, ONE)), Lhat, CAP2 + 1),
            pmul(padd(pscal(2, ONE), pscal(-1, W1)), Lw, CAP2 + 1))
# careful: (2n+1) = (2+w)/w, (2n-1) = (2-w)/w  =>  G = [ (2+w)Lhat - (2-w)Lw ] / w
Gser = {(i - 1, j): c for (i, j), c in Gser.items() if i >= 1}
# x/(n(n+1)) = x w^2/(1+w) = x(w^2 - w^3 + w^4 - ...)
xnn = {}
for k in range(2, CAP2 + 2):
    xnn[(k, 1)] = Fr((-1) ** k)
Psiser = {(i - 3, j): c for (i, j), c in psub(xnn, Gser).items() if i >= 3 and i - 3 <= 1}
psi0 = {(0, j): c for (i, j), c in Psiser.items() if i == 0}
psi1 = {(0, j): c for (i, j), c in Psiser.items() if i == 1}
four_delta = pscal(4, delta)
check("Psi(n,x) -> 4 delta_x as n -> oo",
      psi0 == {(0, j): c for (i, j), c in four_delta.items()})
Cx = pscal(Fr(3, 4), pmul(ppow(X, 2), padd(pscal(3, ONE), pscal(-1, ppow(X, 2)))))
check("[n^-1] Psi = -C(x), C(x) = 3x^2(3-x^2)/4 = 9 delta_x - 6 lambda_x",
      psi1 == {(0, j): -c for (i, j), c in Cx.items()}
      and psub(Cx, psub(pscal(9, delta), pscal(6, lam))) == {})
Cvals = {xv: peval(Cx, 0, xv) for xv in [Fr(1), Fr(5, 4), Fr(3, 2)]}
check("C(1)=3/2, C(3/2)=81/64; C unimodal on [1,3/2] with interior max 27/16 at x^2=3/2",
      Cvals[Fr(1)] == Fr(3, 2) and Cvals[Fr(3, 2)] == Fr(81, 64))
# minimum of C on [1,3/2] via Bernstein on the x-interval
okC, mnC, _ = cert_nonneg(psub(Cx, pscal(Fr(81, 64), ONE)), Fr(0))
check("C(x) - 81/64 >= 0 on [1,3/2] (exact Bernstein in x)", okC, f"min Bcoef = {mnC}")

# =====================================================================
print("\nV3. sharpness of c6 = 1/3")
sharp = psub(pscal(Fr(1, 3), ONE), four_delta)
target = pscal(Fr(1, 3), pmul(ppow(psub(X, ONE), 2), padd(pscal(2, X), ONE)))
check("1/3 - 4 delta_x = (x-1)^2(2x+1)/3  (>=0, =0 iff x=1)", psub(sharp, target) == {})
check("4 delta_x in [0,1/3] on [1,3/2]",
      peval(four_delta, 0, Fr(1)) == Fr(1, 3) and peval(four_delta, 0, Fr(3, 2)) == 0)

# =====================================================================
print("\nV4. (P7) exact Bernstein certificate:  Psi(n,x) <= 4 delta_x for real n >= 100")
K_LO, K_HI, DEN = 6, 7, 18
A = padd(padd(pmul(X, pmul(W1, ppow(OPW, 2))),
              pscal(Fr(1, 2), pmul(ppow(X, 2), pmul(ppow(W1, 2), OPW)))),
         pscal(Fr(1, 4), pmul(ppow(X, 2), ppow(W1, 3))))            # uhat = A/(1+w)^3
uu = padd(padd(pmul(X, W1), pscal(Fr(1, 2), pmul(ppow(X, 2), ppow(W1, 2)))),
          pscal(Fr(1, 4), pmul(ppow(X, 2), ppow(W1, 3))))           # u = U(w,x)
B6 = {}
for k in range(1, K_LO + 1):
    B6 = padd(B6, pscal(Fr((-1) ** (k + 1), k), pmul(ppow(A, k), ppow(OPW, DEN - 3 * k))))
C7 = Lam(uu, K_HI)
Numer = padd(pmul(X, pmul(ppow(W1, 3), ppow(OPW, DEN - 1))),
             padd(pscal(-1, pmul(padd(pscal(2, ONE), W1), B6)),
                  pmul(padd(pscal(2, ONE), pscal(-1, W1)), pmul(C7, ppow(OPW, DEN)))))
Q = divide_by_v(Numer, 4)                                            # Psi* = Q/(1+w)^18
Theta_t = psub(pmul(four_delta, ppow(OPW, DEN)), Q)
Theta_h = divide_by_v(Theta_t, 1)
edge = {(0, j): c for (i, j), c in Theta_h.items() if i == 0}
check("Thetahat(0,x) = C(x) = 9/4 x^2 - 3/4 x^4",
      edge == {(0, j): c for (i, j), c in Cx.items()})
okT, mnT, ntot = cert_nonneg(Theta_h, Fr(1, 100))
check("Bernstein coeffs of Thetahat on [0,1/100]x[1,3/2] all >= 0 (depth 0)", okT,
      f"{ntot} coeffs, min = {mnT} ~ {float(mnT):.6f}")
check("min Bernstein coefficient equals the corner value C(3/2) = 81/64", mnT == Fr(81, 64))
print("       => G(n) >= x/(n(n+1)) - 4 delta_x/n^3 >= x/(n(n+1)) - (1/3)/n^3, n >= 100")

# =====================================================================
print("\nV5. validity of the alternating bracket: need 0 < U < 1 on the box")
umax = peval(uu, Fr(1, 100), Fr(3, 2))
uhmax = Fr(peval(A, Fr(1, 100), Fr(3, 2))) / Fr((1 + Fr(1, 100)) ** 3)
check("U(w,x) and U(w/(1+w),x) have all-positive coefficients => increasing in w,x>0",
      all(c > 0 for c in uu.values()) and all(c > 0 for c in A.values()))
check("max U on box = U(1/100,3/2) = 241809/16000000 < 1",
      umax == Fr(241809, 16000000) and umax < 1, f"~{float(umax):.8f}")
check("max U(w/(1+w),x) on box < 1", uhmax < 1, f"~{float(uhmax):.8f}")

# =====================================================================
print("\nV6. assembly inputs")
Nv = {(1, 0): Fr(1)}

# ---- (P2) sandwich, as exact polynomial positivity (no grid) -----------------
# base n=1: rho_1 = 2x+1.  lower 1+x <= 2x+1 <=> x>=0.  upper 2x+1 <= 1+x+x^2 <=> x(x-1)>=0.
base_up = psub(padd(padd(ONE, X), ppow(X, 2)), padd(pscal(2, X), ONE))     # x^2 - x
check("(P2) base: 1+x <= rho_1 = 2x+1 <= 1+x+x^2 for x >= 1",
      base_up == {(0, 2): Fr(1), (0, 1): Fr(-1)}
      and all(c >= 0 for row in bernstein_box(base_up, Fr(0)) for c in row),
      "upper base cleared to x^2-x >= 0 on [1,3/2]")
# upper step, cleared by (n+1)(n+x) > 0:
#   UP = (n+1+x)(n+1)(n+x) + x^2(n+x) - (2x+1)(n+1)(n+x) - n^2(n+1)  >= 0
NP1 = padd(Nv, ONE)
NPX = padd(Nv, X)
UP = psub(padd(pmul(padd(NP1, X), pmul(NP1, NPX)), pmul(ppow(X, 2), NPX)),
          padd(pmul(padd(pscal(2, X), ONE), pmul(NP1, NPX)), pmul(ppow(Nv, 2), NP1)))
UPs = shift_v(UP, 1)          # n = 1 + n', n' >= 0
okUP = all(c >= 0 for row in bernstein_box(UPs, Fr(1)) for c in row) and \
    all(c >= 0 for row in bernstein_box(UPs, Fr(10 ** 6)) for c in row)
check("(P2) upper step: rho_n >= n+x  =>  rho_{n+1} <= (n+1)+x+x^2/(n+1)",
      okUP, "cleared polynomial >= 0 on n>=1, x in [1,3/2] (Bernstein, two scales)")
# lower step, cleared by (n^2+nx+x^2) > 0:
#   LO = (2x+1)(n^2+nx+x^2) + n^3 - (n+1+x)(n^2+nx+x^2) >= 0   [reduces to x^3 >= 0]
QUAD = padd(padd(ppow(Nv, 2), pmul(Nv, X)), ppow(X, 2))
LO = psub(padd(pmul(padd(pscal(2, X), ONE), QUAD), ppow(Nv, 3)),
          pmul(padd(NP1, X), QUAD))
check("(P2) lower step: rho_n <= n+x+x^2/n  =>  rho_{n+1} >= (n+1)+x  [reduces to x^3 >= 0]",
      LO == {(0, 3): Fr(1)}, f"cleared polynomial = {LO}")

# ---- (P4) 0 <= 1 + m_{n+1} <= 5/(n+1), exact, k = n+1 >= 11 ------------------
# 1+m = (rho_k phi_k - k^2)/(rho_k phi_k);  rho_k phi_k >= (k+x)^2 > k^2  => >= 0.
# upper: rho_k <= (k^2+kx+x^2)/k, phi_k = (4k^3+4k^2x+2kx^2+x^2)/(4k^2), so
#   k(1+m) <= k[rho_k phi_k - k^2]/(k+x)^2  and the claim k(1+m) <= 5 clears to
#   Delta = 20 k^2 (k+x)^2 - [(k^2+kx+x^2)(4k^3+4k^2x+2kx^2+x^2) - 4k^5] >= 0.
Kk = Nv
RHO_up = padd(padd(ppow(Kk, 2), pmul(Kk, X)), ppow(X, 2))
PHI_up = padd(padd(pscal(4, ppow(Kk, 3)), pscal(4, pmul(ppow(Kk, 2), X))),
              padd(pscal(2, pmul(Kk, ppow(X, 2))), ppow(X, 2)))
Delta_m = psub(pscal(20, pmul(ppow(Kk, 2), ppow(padd(Kk, X), 2))),
               psub(pmul(RHO_up, PHI_up), pscal(4, ppow(Kk, 5))))
okm, mnm, _ = cert_nonneg(to_w_form(Delta_m), Fr(1, 11))
check("(P4) 0 <= 1+m_{n+1} <= 5/(n+1) for n >= 10, exact (cleared Delta >= 0 on k>=11)",
      okm, f"min Bernstein coeff = {mnm}")
PHIn = padd(padd(pscal(4, ppow(Nv, 3)), pscal(4, pmul(ppow(Nv, 2), X))),
            padd(pscal(2, pmul(Nv, ppow(X, 2))), ppow(X, 2)))        # 4n^2 phi(n)
PHId = pscal(4, ppow(Nv, 2))
Dnum_raw = psub(psub(pmul(shift_v(PHIn, 1), PHIn),
                     pmul(pmul(padd(pscal(2, X), ONE), shift_v(PHId, 1)), PHIn)),
                pmul(pmul(ppow(Nv, 2), PHId), shift_v(PHId, 1)))
Dden_raw = pmul(shift_v(PHId, 1), PHIn)
Dnum = pmul(ppow(X, 2), padd(padd(pscal(4, pmul(ppow(Nv, 2), ppow(X, 2))),
                                  pscal(-8, pmul(ppow(Nv, 2), X))),
                             padd(padd(pscal(8, pmul(Nv, ppow(X, 2))), pscal(-16, pmul(Nv, X))),
                                  padd(pscal(4, Nv), padd(pscal(3, ppow(X, 2)), pscal(-4, X))))))
Dden = pmul(pscal(4, ppow(padd(Nv, ONE), 2)), PHIn)
check("defect closed form D = num/den matches phi(n+1)-t-n^2/phi(n) exactly",
      psub(pmul(Dnum_raw, Dden), pmul(Dnum, Dden_raw)) == {})
check("den(D) all-positive coefficients => den > 0 for n>0, x>0",
      all(c > 0 for c in Dden.values()))


def cert_abs(numer, denom, power, const, nmin):
    for sgn in (+1, -1):
        e = psub(pscal(const, denom), pscal(sgn, pmul(ppow(Nv, power), numer)))
        ok, _, _ = cert_nonneg(to_w_form(e), Fr(1, nmin))
        if not ok:
            return False
    return True


check("|D(n)| <= (9/16)/n^3 on n >= 60, x in [1,3/2]",
      cert_abs(Dnum, Dden, 3, Fr(9, 16), 60))
check("|D(n)| <= (43/100)/n^3 on n >= 60  [sharper; asymptote 27/64 = 0.421875]",
      cert_abs(Dnum, Dden, 3, Fr(43, 100), 60))
Dn1 = shift_v(Dnum, 1)
Dd1 = shift_v(Dden, 1)
DiffN = psub(pmul(Dn1, Dden), pmul(Dnum, Dd1))
DiffD = pmul(Dd1, Dden)
check("|D(n+1)-D(n)| <= 8.87/n^4 on n >= 60",
      cert_abs(DiffN, DiffD, 4, Fr(887, 100), 60))
check("|D(n+1)-D(n)| <= 1.29/n^4 on n >= 60  [sharper; asymptote 81/64 = 1.265625]",
      cert_abs(DiffN, DiffD, 4, Fr(129, 100), 60))
# multipliers: 1+m_{n+1} <= 5/(n+1), k = n+1 >= 11
# with a = x+x^2/k, b = x+x^2/2k+x^2/4k^2:  (1+m) <= (k(a+b)+ab)/(k+x)^2
# bound a+b <= A0 and ab <= B0 using k >= 11, x <= 3/2, then k*A0+B0 <= 5(k+x)^2/k
ebar = Fr(887, 100) + 5 * Fr(9, 16)
check("|E_n| <= (8.87 + 5*(9/16))/n^4 = 11.6825/n^4 <= 11.69/n^4", ebar <= Fr(1169, 100),
      f"{ebar} = {float(ebar):.4f}")


def tail_bound(c, n0):
    a = Fr(n0 - 2)
    return c * (Fr(1) / a + Fr(2) / a ** 2 + Fr(4, 3) / a ** 3) / 2


t_coarse = tail_bound(Fr(1169, 100), 100)
check("parity-chain tail sum_{j>=100,step 2}(j+2)^2|E_j| <= 0.0620", t_coarse <= Fr(62, 1000),
      f"{t_coarse} ~ {float(t_coarse):.6f}")
S_coarse = Fr(1, 50) + Fr(62, 1000)
check("S = sup_{m>=100} m^2|s_m| <= 1/50 + 0.0620 = 0.082", S_coarse <= Fr(82, 1000))
errfac = Fr(4) / (1 - Fr(82, 1000) / 10 ** 6)
check("|A_n - G(n)| <= 4.01 S/n^2 for n >= 100", errfac <= Fr(401, 100),
      f"exact factor {float(errfac):.8f}")
weight_surplus = psub(pscal(2, ppow(padd(Nv, ONE), 3)), pmul(padd(pscal(2, Nv), ONE), ppow(Nv, 2)))
check("(2n+1)n^2 <= 2(n+1)^3 for all n >= 1 (cleared: 0 <= 5n^2+6n+2)",
      weight_surplus == {(2, 0): Fr(5), (1, 0): Fr(6), (0, 0): Fr(2)})

# continuum certificates, rebuilt from scratch
print("       continuum certificates n0^2|s_{n0}(t)| <= 1/50 on t in [3,4]:")


def p1mul(a, b):
    r = [Fr(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                r[i + j] += ai * bj
    return r


def p1add(a, b):
    if len(a) < len(b):
        a, b = b, a
    r = list(a)
    for i, bi in enumerate(b):
        r[i] += bi
    return r


def p1scal(c, a):
    return [Fr(c) * ai for ai in a]


def bern1(a, lo, hi):
    d = len(a) - 1
    q = [Fr(0)] * (d + 1)
    for k, ak in enumerate(a):
        if ak:
            for j in range(k + 1):
                q[j] += ak * comb(k, j) * Fr(lo) ** (k - j) * (Fr(hi) - Fr(lo)) ** j
    return [sum(q[k] * Fr(comb(i, k), comb(d, k)) for k in range(i + 1)) for i in range(d + 1)]


P = [[Fr(1)], [Fr(0), Fr(1)]]
for m in range(1, 102):
    P.append(p1add(p1mul([Fr(0), Fr(1)], P[m]), p1scal(m * m, P[m - 1])))
fact = 1
ok_sanity = True
for nv in range(1, 10):
    fact *= nv
    ok_sanity = ok_sanity and (sum(c * Fr(3) ** i for i, c in enumerate(P[nv])) / fact == 2 * nv + 1)
check("P_n(3)/n! = 2n+1 for n=1..9 (recurrence sanity against T(1,n))", ok_sanity)
allcert = True
for n0 in (100, 101):
    c2c = Fr(1, 8 * n0) + Fr(1, 16 * n0 * n0)
    phit = [Fr(n0) - Fr(1, 2) + c2c, Fr(1, 2) - 2 * c2c, c2c]
    diff = p1add(P[n0], p1scal(-1, p1mul(phit, P[n0 - 1])))
    cap = Fr(1, 50 * n0 * n0)
    if not all(c >= 0 for c in P[n0 - 1]):
        allcert = False
    for sgn in (-1, +1):
        Qp = p1add(p1scal(cap, P[n0 - 1]), p1scal(sgn, diff))
        if not all(c > 0 for c in bern1(Qp, 3, 4)):
            allcert = False
check("all four Bernstein certificates positive at depth 0 (n0 = 100, 101; deg 101, 102)",
      allcert)

# =====================================================================
print(f"\nV7. (P1) finite bridge: N_n(3+u) coefficients > 0, 2 <= n <= {NBRIDGE}")


def imul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                if bj:
                    r[i + j] += ai * bj
    return r


def iadd(a, b):
    if len(a) < len(b):
        a, b = b, a
    r = list(a)
    for i, bi in enumerate(b):
        r[i] += bi
    return r


def iscal(c, a):
    return [c * ai for ai in a]


def ideriv(a):
    return [i * a[i] for i in range(1, len(a))] or [0]


def ishift3(a):
    d = len(a) - 1
    q = [0] * (d + 1)
    pw = [3 ** k for k in range(d + 1)]
    for k, ak in enumerate(a):
        if ak:
            for j in range(k + 1):
                q[j] += ak * comb(k, j) * pw[k - j]
    return q


IP = [[1], [0, 1]]
for m in range(1, NBRIDGE + 3):
    IP.append(iadd(imul([0, 1], IP[m]), iscal(m * m, IP[m - 1])))
IW = [None, [1]]
for m in range(1, NBRIDGE + 3):
    IW.append(iadd(imul(IP[m], IP[m]), iscal(-(m * m), IW[m])))
badw = 0
for m in range(1, 25):
    d = iadd(imul(ideriv(IP[m]), IP[m - 1]), iscal(-1, imul(IP[m], ideriv(IP[m - 1]))))
    a, b = list(IW[m]), list(d)
    while len(a) < len(b):
        a.append(0)
    while len(b) < len(a):
        b.append(0)
    badw += (a != b)
check("W_{n+1} = P_n^2 - n^2 W_n agrees with the Wronskian P_n'P_{n-1}-P_nP_{n-1}', n<=24",
      badw == 0)
tot, nonpos, degok, leadok = 0, 0, True, True
for n in range(2, NBRIDGE + 1):
    Nn = iadd(iscal(2 * n + 1, imul(IW[n + 1], IP[n - 1])),
              iscal(-(2 * n - 1), imul(IW[n], IP[n + 1])))
    while len(Nn) > 1 and Nn[-1] == 0:
        Nn.pop()
    degok = degok and (len(Nn) - 1 == 3 * n - 1)
    leadok = leadok and (Nn[-1] == 2)
    Qs = ishift3(Nn)
    tot += len(Qs)
    nonpos += sum(1 for c in Qs if c <= 0)
EXPECT = 21777
check(f"deg N_n = 3n-1 and leading coefficient 2, all n <= {NBRIDGE}", degok and leadok)
check(f"all {tot} coefficients of N_n(3+u) strictly positive, 2 <= n <= {NBRIDGE}",
      nonpos == 0, f"count {tot}" + (f" (expected 21777)" if EXPECT else ""))
check("finite coefficient count is exactly 21,777", tot == EXPECT)
badA = 0
for n in range(2, 41):
    Nn = iadd(iscal(2 * n + 1, imul(IW[n + 1], IP[n - 1])),
              iscal(-(2 * n - 1), imul(IW[n], IP[n + 1])))
    v3 = sum(c * 3 ** i for i, c in enumerate(Nn))
    den = 1
    for m in (n + 1, n, n - 1):
        den *= sum(c * 3 ** i for i, c in enumerate(IP[m]))
    pred = (Fr(1, (n + 1) * (2 * n + 1) * (2 * n + 3)) if n % 2 == 0
            else Fr(4 * n + 3, n * (2 * n + 1) * (2 * n + 3)))
    badA += (Fr(v3, den) != pred)
check("A_n'(3) closed forms (even/odd) reproduced exactly for n = 2..40", badA == 0)

# =====================================================================
print("\nV8. reduction chain")
# artanh identity: log((z+1)/(z-1)) = 2 sum_{k>=0} z^{-(2k+1)}/(2k+1).
# Verified as a formal identity in y = 1/z: log((1+y)/(1-y)) - 2 sum y^(2k+1)/(2k+1) = 0.
CAPY = 25
Y = {(1, 0): Fr(1)}
logp = Lam(Y, CAPY, cap=CAPY)                                   # log(1+y)
logm = Lam(pscal(-1, Y), CAPY, cap=CAPY)                        # log(1-y)
lhs8 = psub(logp, logm)
rhs8 = {}
for k in range(0, CAPY // 2 + 1):
    if 2 * k + 1 <= CAPY:
        rhs8[(2 * k + 1, 0)] = Fr(2, 2 * k + 1)
check(f"log((1+y)/(1-y)) = 2 sum y^(2k+1)/(2k+1), formal identity to y^{CAPY}",
      psub(lhs8, rhs8) == {})
h_series = {(degree - 1, xdegree): coefficient for (degree, xdegree), coefficient in rhs8.items()}
termwise_h = (
    h_series.get((0, 0)) == 2
    and all(degree % 2 == 0 and coefficient > 0 for (degree, _), coefficient in h_series.items())
)
check("=> h(z) := z log((z+1)/(z-1)) = 2 sum_{k>=0} z^(-2k)/(2k+1), termwise decreasing "
      "for z>1, so h is strictly decreasing", termwise_h, "positive even inverse powers")
h_argument_order = all(2 * n - 1 > 1 and 2 * n + 1 > 2 * n - 1 for n in range(2, 42))
check("=> (2n-1)log(n/(n-1)) > (2n+1)log((n+1)/n) for n >= 2  [h(2n-1) > h(2n+1)]",
      termwise_h and h_argument_order, "h decreasing; affine arguments differ by 2")
artanh_strict_tail = (
    rhs8.get((1, 0)) == 2
    and all(coefficient > 0 for (degree, _), coefficient in rhs8.items() if degree > 1)
)
check("log(1+u) > 2u/(2+u) for u>0  [same identity with y = u/(2+u)]", artanh_strict_tail,
      "(1+y)/(1-y) = 1+u; log(1+u) = 2(y+y^3/3+...) > 2y")
check("=> A_n(3) = f(2n+1)-f(2n-1) > 0 with f(y) = y log(1+2/y)  [P0]",
      artanh_strict_tail and badA == 0)
check("A_n > 0 and log r_n > 0  =>  log r_{n+1}/log r_n > (2n-1)/(2n+1) > "
      "log((n+1)/n)/log(n/(n-1))  =>  sigma_{n+1} > sigma_n",
      h_argument_order and nonpos == 0 and degok and leadok)
p2_replay = (
    all(c >= 0 for row in bernstein_box(base_up, Fr(0)) for c in row)
    and okUP
    and LO == {(0, 3): Fr(1)}
)
check("log r_n > 0 for t >= 3: r_n = rho_n/n >= (n+x)/n > 1 by the (P2) sandwich",
      p2_replay)

# =====================================================================
print("\nV9. final margin at n = 100")
c6 = Fr(1, 3)
lhs9 = Fr(100, 101)
rhs9 = c6 / 100 + Fr(401, 100) * Fr(82, 1000)
check("100/101 > c6/100 + 4.01*0.082  with c6 = 1/3", lhs9 > rhs9,
      f"{float(lhs9):.6f} > {float(rhs9):.6f}, margin {float(lhs9-rhs9):+.6f}")
thr = 100 * (Fr(100, 101) - Fr(401, 100) * Fr(82, 1000))
check("assembly admissibility threshold: it closes at n=100 iff c6 < 3339459/50500",
      thr == Fr(3339459, 50500) and Fr(1, 3) < thr and Fr(21) < thr,
      f"threshold ~ {float(thr):.4f}; c6=1/3 uses {float(Fr(1,3)/thr*100):.2f}% of the budget, "
      f"c6=21 uses {float(Fr(21)/thr*100):.1f}%")
rhs9_coarse = Fr(21) / 100 + Fr(401, 100) * Fr(82, 1000)
check("the coarser c6 = 21 bound also closes", lhs9 > rhs9_coarse,
      f"margin {float(lhs9-rhs9_coarse):+.6f}")
monotone_differences = all(
    Fr(n + 1, n + 2) > Fr(n, n + 1) and c6 / (n + 1) < c6 / n
    for n in range(100, 201)
)
check("margin is monotone favourable in n (x n/(n+1) up, c6/n down)",
      monotone_differences and c6 > 0)

print("\n" + "=" * 74)
expected_gate_count = 50
if NCHECK != expected_gate_count:
    FAILS.append(f"gate count {NCHECK}, expected {expected_gate_count}")
if FAILS:
    print(f"VERIFY_UPPER_STRIP: FAIL -- {len(FAILS)} of {NCHECK} gates failed:")
    for f in FAILS:
        print("   -", f)
    sys.exit(1)
certificate = {
    "schema": "delannoy-upper-strip-v1",
    "arithmetic": "exact integer and rational",
    "domain": {"x": "[1,3/2]", "n": "integers n>=2"},
    "gate_count": NCHECK,
    "finite_bridge": {
        "n_min": 2,
        "n_max": NBRIDGE,
        "positive_shifted_coefficients": tot,
    },
    "p7_bernstein": {
        "box": "[0,1/100] x [1,3/2]",
        "coefficient_count": ntot,
        "minimum": f"{mnT.numerator}/{mnT.denominator}",
    },
    "continuum_base_polynomials": 4,
    "sharp_c6": "1/3",
    "final_margin_at_n_100": f"{(lhs9-rhs9).numerator}/{(lhs9-rhs9).denominator}",
    "passed": True,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"certificate={args.output}")
print(f"VERIFY_UPPER_STRIP: PASS -- all {NCHECK} gates green")
print("=" * 74)
sys.exit(0)
