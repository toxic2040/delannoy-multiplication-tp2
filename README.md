# Delannoy multiplication-table TP2

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21778524.svg)](https://doi.org/10.5281/zenodo.21778524)

Deterministic exact-arithmetic replay accompanying the sharp half-line theorem:

    T(x,n) = sum_j 2^j C(x,j) C(n,j),

and the multiplication table M_x(i,j) = T(x,ij) is strictly totally positive
of order two exactly when x >= 1/2.

Equivalently, for x >= 1/2,

    T(x,a) T(x,b) > T(x,c) T(x,d)

whenever ab = cd and a > c >= d > b >= 1. For x < 1/2 the property fails.

This is multiplicative sampling of one real-continued Delannoy sequence. It is
not the classical total-positivity statement for the additive Delannoy array.

## Verification

The authoritative runner is fail-closed for the release manifest and for every
implemented gate. It uses paths relative to the checkout:

    python3 -I -B verify_all.py

Its components reconstruct:

- the defining recurrence, the first-cell threshold witnesses, margin identities, and
  grounded-path determinant formulas;
- the compact and unbounded parts of the universal dimension step;
- the displayed two-cut expansion and the exact arithmetic assembled from its
  named analytic remainder inputs;
- the two analytic lower-strip gates and all 260 canonical finite-residue
  certificates, up to p-q symmetry; and
- the upper-strip finite coefficients, Bernstein certificates, and exact
  two-step tail-assembly identities.

All implemented sign decisions use exact integer, rational, or symbolic
polynomial arithmetic. Finite computations prove only their enumerated ranges.
Embedded mutation controls must be rejected for the suite to pass, but they
cover named failure modes rather than every possible source mutation.

The repository is best described as a `REPRODUCIBLE_MANUSCRIPT_REPLAY`. It is
not a proof-assistant development or a self-contained full-theorem
certificate. In particular, the Darboux necessity argument for `0 < x < 1/2`
and the analytic Binet, beta-integral, and related remainder lemmas are proofs
in the paper; the scripts reconstruct their algebraic consequences and finite
certificates. The detailed branch-by-branch boundary is in
[VERIFICATION_SCOPE.md](VERIFICATION_SCOPE.md). Independent coefficientwise
strengthenings that remain unresolved are stated precisely in
[OPEN.md](OPEN.md); none is needed for the sharp TP2 theorem.

The reference environment is CPython 3.14.5 with SymPy 1.14.0, pinned in
`requirements.txt`. The scripts also
run on supported GIL builds because this release does not use a CPU-bound
parallel sweep.

The runner has no quick mode. It validates the sorted release
manifest, gives every component a 900-second timeout, rebuilds each certificate
in a temporary directory, and requires byte-for-byte agreement with the banked
record. Missing files, dependencies, exceptions, skipped outputs, hash changes,
or certificate failures produce a nonzero exit.

## Scope

This repository contains replay code and deterministic certificates.
The paper's PDF and source are archived with this suite in the versioned Zenodo
record [doi:10.5281/zenodo.21778524](https://doi.org/10.5281/zenodo.21778524).

## Citation

J. Councilman, *Strict total positivity of a real Delannoy multiplication
table: the sharp half-line threshold*, version 2.0, 2026.
[doi:10.5281/zenodo.21778524](https://doi.org/10.5281/zenodo.21778524)

Machine-readable citation metadata is in `CITATION.cff`.

## License

Verification code is MIT licensed. Documentation and certificate records are
CC BY 4.0.
