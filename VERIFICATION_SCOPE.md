# Verification scope

The paper proves the sharp half-line theorem. This repository is its
deterministic exact-arithmetic replay, classified as
`REPRODUCIBLE_MANUSCRIPT_REPLAY`.

That label is deliberate. A clean run shows that the recorded source tree,
implemented identities, exact finite certificates, and named mutation controls
agree. It does not mean that every analytic sentence in the paper has been
translated into executable proof code.

## Branch coverage

| Theorem branch | Executable coverage | Boundary |
|---|---|---|
| Necessity for `x < 0` and degeneracy at `x = 0` | The first adjacent minor is reconstructed and factored; the quartic root check and zero specialization are exact. | Covered by `verify_structural.py`. |
| Necessity for `0 < x < 1/2` | The recurrence and four-corner identities used by the argument are checked. | The two-singularity Darboux expansion, its remainder, and the eventual-sign deduction are paper proofs, not an executable all-parameter certificate. |
| Universal dimension step | The compact and unbounded symbolic certificates reconstruct the stated rational identities, polynomial signs, resultants, and endpoint cases. | Exact SymPy algebra in this repository. A separate Lean 4 development proves `thm:dimension-step` and `cor:half-integers` with zero `sorry`; it is not included in or consumed by this replay. |
| Lower strip `[1/2,1]` | The replay derives the displayed coefficients, binds all three positive remainder envelopes into the large-cell budget, checks the analytic column/interior inequalities, and reconstructs all 260 canonical residue cells coefficientwise. | Binet, beta-integral, log-Gamma, digamma, and related analytic lemmas are named manuscript inputs. The runner checks their arithmetic use, not the lemmas from first principles. |
| Upper strip `[1,3/2]` | The replay checks all 21,777 finite shifted coefficients, all 540 Bernstein coefficients, the recurrence defect bounds, the two-step parity recurrence, weighted contraction algebra, tail integral, base polynomials, and final margin. | Elementary monotonicity and logarithmic inequalities remain ordinary paper reasoning around the exact gates. |
| Adjacent-cell lift to every ordered equal-product quadruple | Product preservation and adjacent-cell identities are reconstructed. | The rectangle-sum reduction is a paper proof. |

## Fail-closed boundary

`verify_all.py` fails on a missing or extra release file, manifest drift,
dependency mismatch, component exception or timeout, certificate mismatch,
failed gate, or surviving named mutation. The compact Bernstein conversion
also rejects a caller-supplied degree below the polynomial's actual degree;
higher terms cannot be silently discarded.

Mutation controls are regression tests for their stated targets. They do not
establish that arbitrary source edits must be detected independently of the
manifest, code review, and rebuilt certificate comparison.

The versioned paper and source/replay archive remain at
[doi:10.5281/zenodo.21778524](https://doi.org/10.5281/zenodo.21778524).
