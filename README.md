# A note and restoration on the permutation, polynomial, and saddle-point moments in "Sieve weights and their smoothings"

Ruqing Chen (2026). doi:[10.5281/zenodo.24297078](https://doi.org/10.5281/zenodo.24297078)

A. Granville, D. Koukoulopoulos and J. Maynard, *Sieve weights and their smoothings*, Ann. Sci. Éc. Norm. Supér.
(4) 54 (2021), no. 5, 1089–1177; arXiv:1606.06781. This note checks Sections 2.5, 3 and 4 (the permutation and
F_q[t] analogues). Page numbers refer to the arXiv version (77 pages).

* **Proposition 3.1.** The divisor sum (3.1) omits the multiplicity `prod_j binom(c_j, b_j)`. As printed the
  proposition fails (for m = 1 the right side is `1 - 1/e` for every k, while `c(1,k)` is the Bell number
  `B_2k`), and the recurrence on p. 21 holds only for the unweighted sum. With the weights restored,
  `c(m,k) = E[M(X;m)^2k]` follows from the generating function
  `prod_{|S| odd}(1 - x_S) / prod_{|S| even}(1 - x_S)`, which also gives the Pascal shift
  `M(c + e_i; m) = M(c; m) - M(c; m - i)` and a two-line end to the proof of Theorem 1.4.
* **p. 20.** `binom(q,c1)/(q-1)^c1 = (1/c1!)(1 + O(c1/q))` for `c1 <= q` is false; the expansion is
  `1 - c1(c1-3)/(2q) + ...`, valid for `c1 <= sqrt q`. A quadratic Poisson moment (`<= 16^k c(m,k)`) repairs
  the error term; Theorem 1.4 holds as stated.
* **Section 2.5.** Three slips (the factor `r`, non-constant zeta factors, a symmetry cancellation that fails)
  give the region exponent `2^{2k-1} - 6k - 1`; corrected, `2^{2k-1} - 4k - 1`. The limit
  `lim_q Poly_q(n,m,2;k)` is a 2k-fold mixed difference of the lattice-point count:
  `L_2(m,2) = 15m^2 - 51m + 65` (m >= 3), and for k = 3 the exponent is estimated at 20–22 from exact values
  for m <= 12. The conclusion of Section 2.5 is unaffected.
* Remark 3.1 (`3c(m,2) = 64m^3 - 135m^2 + 182m - 66`) is verified for 1 <= m <= 30.

The exponent for k = 3 is a numerical extrapolation, not a proof.

| path | content |
|---|---|
| `gkm_sieve_weights_audit.tex`, `.pdf` | the note (4 pages) |
| `code/verify_gkm_2021.py` | exact coefficient extraction (6-dim for k = 3, modulo two 61-bit primes), Poisson moments, ratio checks, region integrals, and the figure (about 40 s) |
| `data/verify_gkm_2021.txt` | output of the script |
| `data/gkm_exact_values.json` | exact `c(m,k)` (k <= 3) and `L_2(m,k)` (k = 2, 3) |
| `figures/fig_gkm_2021_audit.pdf`, `.png` | Figure 1 |

Requirements: Python 3 with numpy, scipy, sympy, matplotlib. Run `python code/verify_gkm_2021.py`.
