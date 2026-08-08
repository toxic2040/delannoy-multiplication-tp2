# Open structural strengthening

The sharp TP2 threshold is proved: the multiplication table is strictly TP2
exactly for `x >= 1/2`.  The questions below are independent strengthenings of
that theorem.  Their outcome does not affect the published threshold result.

## Coefficient formulation

Let `Q_n(u)` be the odd-cycle probability generating polynomial used in the
paper, and set

```text
L = {(p+1)(q+1), pq, p(q+3), (p+1)(q+2)},
R = {p(q+1), (p+1)q, (p+1)(q+3), p(q+2)}.
```

For `p >= 1`, `q >= 2`, and `p` and `q` of the same parity, write

```text
prod_{n in L} Q_n(1+y) - prod_{n in R} Q_n(1+y)
    = sum_{k>=1} delta_k y^k.
```

The current all-index boundary is:

- `delta_1 > 0` is proved universally;
- `delta_2 > 0` is proved universally on this domain, as the same-parity
  case of the sharp signed law `sgn delta_2 = (-1)^(p+q)` for all
  `p >= 1`, `q >= 2`;
- the sign `delta_k > 0` is open for every fixed `k >= 3`; and
- the stronger raw stochastic dominance statement is open.  Equivalently, it
  is not known whether every coefficient of

  ```text
  (prod_{n in L} Q_n(u) - prod_{n in R} Q_n(u)) / (u-1)
  ```

  is positive throughout the same-parity domain.

The unique odd row on each side has been eliminated exactly.  With
`E_m(y)=Q_(2m)(1+y)`, `O_m(y)=Q_(2m+1)(1+y)`,
`b_j=C(2j,j)/4^j`, and `c_m=1/((2m+1)b_m)`,

```text
O_m = c_m sum_{j=0}^m b_j E_j.
```

This reduces every same-parity cell to an all-even comparison plus an explicit
adverse central-binomial tail.  Controlling that compensation uniformly is
still open.

## Finite evidence and its limit

A separate 90-digit diagonal screen over

```text
2 <= p=q <= 400,    2 <= k <= 20
```

found all `7,578` supported coefficients positive.  For the symmetric relative
margin

```text
m_(p,k) = delta_k / (left_k + right_k),
```

the minimum was

```text
3.868276480615235e-14 at (p,q,k)=(400,400,15).
```

A 120-digit rerun on `390 <= p=q <= 400` reproduced the value.  This is a
diagonal screen, not a full `p,q <= 400` square, and it is not part of the
authoritative v2.0 release runner.

The small margin is not approaching a visible floor.  At fixed `k=15`, a
log-log fit over `100 <= p <= 400` has slope `-5.15749`; the local fit over
`390 <= p <= 400` has slope `-4.97589`.  The observed regime is therefore
consistent with power-law decay.  The clean finite screen is useful regression
evidence, but it does not make a failure at much larger scale implausible.

Closure of this strengthening requires an all-index argument or an exact
counterexample.  More finite screening alone will not settle it.
