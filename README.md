# Delannoy multiplication-table TP2

Exact verification code for the sharp half-line theorem:

    T(x,n) = sum_j 2^j C(x,j) C(n,j),

and the multiplication table M_x(i,j) = T(x,ij) is strictly totally positive
of order two exactly when x >= 1/2.

Equivalently, for x >= 1/2,

    T(x,a) T(x,b) > T(x,c) T(x,d)

whenever ab = cd and a > c >= d > b >= 1. For x < 1/2 the property fails.

This is multiplicative sampling of one real-continued Delannoy sequence. It is
not the classical total-positivity statement for the additive Delannoy array.

## Verification

The suite is fail-closed and uses paths relative to the checkout:

    python3.14 -I -B verify_all.py

Its components reconstruct:

- the defining recurrence, threshold witnesses, margin identities, and
  grounded-path determinant formulas;
- the compact and unbounded parts of the universal dimension step;
- the uniform two-cut expansion and its exact lower-strip remainder bounds;
- the two analytic lower-strip gates and all 260 canonical finite-residue
  certificates, up to p-q symmetry; and
- the upper-strip finite coefficients and Bernstein certificates.

All sign decisions use exact integer, rational, or symbolic polynomial
arithmetic. Finite computations prove only their enumerated ranges; the
all-parameter branches are separately reconstructed. Embedded mutation
controls must be rejected for the suite to pass.

The reference environment is CPython 3.14.5 with SymPy 1.14.0, pinned in
`requirements.txt`. The scripts also
run on supported GIL builds because this release does not use a CPU-bound
parallel sweep.

The authoritative runner has no quick mode. It validates the sorted release
manifest, gives every component a 900-second timeout, rebuilds each certificate
in a temporary directory, and requires byte-for-byte agreement with the banked
record. Missing files, dependencies, exceptions, skipped outputs, hash changes,
or certificate failures produce a nonzero exit.

## Scope

This repository contains verification code and deterministic certificates.
The paper's PDF and source are archived in the associated Zenodo record. The
record DOI and citation metadata are inserted only after the release files have
passed a clean-checkout replay.

## License

Verification code is MIT licensed. Documentation and certificate records are
CC BY 4.0.
