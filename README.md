# A real Delannoy multiplication table

Continue the Delannoy number `D(m,n)` to real first argument by

    T(x,n) = sum_j 2^j C(x,j) C(n,j)

and ask whether the multiplication table `M_x(i,j) = T(x,ij)` is strictly
totally positive of order two, that is, whether

    T(x,a) T(x,b) > T(x,c) T(x,d)   whenever   ab = cd,  a > c >= d > b >= 1.

This is not the additive Delannoy array, and the known total positivity of that
array does not decide it either way.

The threshold `x >= 1/2` is necessary: the property fails for every negative
`x`, degenerates at `x = 0`, and fails for every noninteger `x` in `(0,1/2)`. A
universal dimension step carries every strict comparison at `x` to `x + 1` for
all `x >= 1/2`, so the endpoint and integer bases settle every positive
half-integer, and the unresolved real parameter set is exactly

    (1/2, 1) union (1, 3/2).

A distinguished sub-family is proved on the whole half-line. The closure lemma
that would settle the rest is proved at `x = 1/2, 1, 3/2` and is open on the
two interiors.

Along the way the paper settles the first three rungs of the threshold-tail
tower. For every `p >= 1`, `q >= 2` the first coefficient of the odd-cycle
closure polynomial has sign `(-1)^(p+q)` — positive in same parity, negative in
mixed. In every **same-parity** cell the second tail `U_2` is strictly
positive, with an all-index proof rather than a bounded census, and the second
coefficient obeys

    delta_2 < (log((p+1)(q+3)) + 5/4) delta_1.

A retained-remainder argument at `x = 1/2` supplies the matching quantitative
endpoint gap; combined with an exact 2,685-cell residual it makes the third
tail `U_3` strictly positive in every same-parity cell, with no conditional.
The first unresolved rung is `U_4 = U_3 - 27 delta_3 > 0`.

This repository carries the verification suite and the proof certificates. The
paper itself is on Zenodo, at the DOI below.

## Verify it

The suite needs Python 3 and SymPy, and nothing else.

    pip install sympy
    python3 verify_manuscript_anchors.py

That recomputes the paper's displayed numeric constants and finite censuses in
exact arithmetic — 551 exact checks in the current run, each printed beside its
recomputed value. It exits nonzero on any mismatch. Section 13 of the paper
enumerates what the suite gates and what it does not.

The four remaining scripts carry the proof certificates.

    python3 verify_dimension_step_compact.py     # both compact center strips
    python3 verify_dimension_step_unbounded.py   # the unbounded strip
    python3 verify_closure_anchors.py            # the three closure anchors
    python3 verify_threshold_tail_three.py       # the third threshold tail

The two dimension-step scripts rewrite the canonical JSON certificates under
`results/`; a clean run leaves them byte-identical, so `git status` staying
empty is itself part of the check. `verify_closure_anchors.py` writes no file;
its evidence is the printed gate count (`all_parameter_gates=9/9`) and the
coefficient digests. `verify_threshold_tail_three.py` gates the quantitative
endpoint gap and the third tail — 48,339 exact checks, including the 1,770-cell
endpoint replay and the 2,685-cell residual — and is stdlib only, no SymPy.

`anchors.txt` is the frozen list of manuscript anchors that
`verify_manuscript_anchors.py` reproduces.

On a recent desktop the anchor script takes about eighty seconds, the third-tail
script about four minutes, and the other three about ten seconds each. The
reference interpreter is CPython 3.14 free-threading; the suite is
single-threaded and also runs unchanged on CPython 3.13.

## Paper

    DOI: 10.5281/zenodo.<CONCEPT>
    https://doi.org/10.5281/zenodo.<CONCEPT>

That is the concept DOI and resolves to the latest version. The Zenodo record
holds the manuscript source and the rendered PDF.

## Citation

    J. Councilman, "Strict total positivity of a real Delannoy multiplication
    table: a dimension ladder and the remaining base strip", 2026.
    DOI: 10.5281/zenodo.<CONCEPT>

## License

The five verification scripts are MIT; see `LICENSE`. The certificate files
under `results/` and the anchor list `anchors.txt` are CC-BY-4.0; see
`LICENSE-CC-BY-4.0`. The manuscript is CC-BY-4.0 and is archived at the DOI
above.
