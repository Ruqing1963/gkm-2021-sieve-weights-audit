"""Verification script for the audit note on
A. Granville, D. Koukoulopoulos and J. Maynard, "Sieve weights and their smoothings",
Compositio Math. 157 (2021), 597-648 (DOI 10.1112/S0010437X2000771X; arXiv:1606.07737).
Page numbers refer to the arXiv version (77 pages).

Run:  python verify_gkm_2021.py
Writes ../data/verify_gkm_2021.txt and ../figures/fig_gkm_2021_audit.{pdf,png}

Key generating function.  For X_j ~ Poisson(1/j) independent and the true divisor sum
    M(c; l) = sum_{sum_j j b_j = l} (-1)^{b_1+...+b_m} prod_j binom(c_j, b_j)  = [x^l] prod_j (1 - x^j)^{c_j},
    E[ prod_{i=1}^{2k} x_i^{.}-coefficients ] :  sum_{l_1..l_2k} E[prod_i M(X; l_i)] x^l
       = prod_j exp((1/j)(prod_i (1 - x_i^j) - 1)) = prod_{S odd} (1 - x_S) / prod_{S even} (1 - x_S),
with S over nonempty subsets of [2k] and x_S = prod_{i in S} x_i.  Its coefficient at v is (-1)^{|v|} N(v), where
N(v) counts tuples (r_S) with r_S in {0,1} for |S| odd, r_S >= 0 for |S| even and sum_{S ni i} r_S = v_i.
Hence c(m,k) = N(m,...,m) = E[M(X;m)^{2k}]  (Proposition 3.1 with the binomial weights), and
    L_h(m,k) := E[(sum_{m-h<l<=m} M(X;l))^{2k}] = sum_{v in (m-h,m]^{2k}} (-1)^{|v|} N(v),
the q -> infinity limit of Poly_q(n,m,h;k).
"""
import itertools
import math
import os
import sys
from functools import lru_cache

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_TXT = os.path.join(ROOT, "data", "verify_gkm_2021.txt")
OUT_PDF = os.path.join(ROOT, "figures", "fig_gkm_2021_audit.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig_gkm_2021_audit.png")
_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def check(name, ok):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        check.failures += 1


check.failures = 0


# ---------------------------------------------------------------------------
# divisor sums
# ---------------------------------------------------------------------------
def b_vectors(m, l):
    """all b = (b_1..b_m) >= 0 with sum j b_j = l."""
    out = []

    def rec(j, rem, cur):
        if j > m:
            if rem == 0:
                out.append(tuple(cur))
            return
        for b in range(rem // j + 1):
            rec(j + 1, rem - j * b, cur + [b])
    rec(1, l, [])
    return out


@lru_cache(maxsize=None)
def bvec_cached(m, l):
    return tuple(b_vectors(m, l)) if l >= 0 else tuple()


def M_true(c, l):
    m = len(c)
    if l < 0:
        return 0
    s = 0
    for b in bvec_cached(m, l):
        t = 1
        for cj, bj in zip(c, b):
            if bj > cj:
                t = 0
                break
            t *= math.comb(cj, bj)
        if t:
            s += (-1) ** sum(b) * t
    return s


def M_printed(c, l):
    m = len(c)
    if l < 0:
        return 0
    return sum((-1) ** sum(b) for b in bvec_cached(m, l) if all(bj <= cj for bj, cj in zip(b, c)))


def poisson_expect(m, f, cap=None):
    """E[f(X)] for X_j ~ Poisson(1/j), j<=m, truncated where the tail is < 1e-18."""
    caps = [max(10, int(26 / j) + 8) for j in range(1, m + 1)] if cap is None else cap
    pmfs = []
    for j in range(1, m + 1):
        lam = 1.0 / j
        pmfs.append([math.exp(-lam) * lam**c / math.factorial(c) for c in range(caps[j - 1] + 1)])
    tot = 0.0
    for c in itertools.product(*[range(cp + 1) for cp in caps]):
        w = 1.0
        for j, cj in enumerate(c):
            w *= pmfs[j][cj]
        if w < 1e-30:
            continue
        tot += w * f(c)
    return tot


# ---------------------------------------------------------------------------
# exact coefficients of G = prod_{S odd}(1-x_S) / prod_{S even}(1-x_S) on [0,Mx]^{2k}
# ---------------------------------------------------------------------------
def G_grid(k, Mx, p=None):
    d = 2 * k
    shape = (Mx + 1,) * d
    dtype = np.int64 if p is not None else object
    arr = np.zeros(shape, dtype=dtype)
    arr[(0,) * d] = 1
    subsets = [S for r in range(1, d + 1) for S in itertools.combinations(range(d), r)]

    def shifted_add(arr, S, step, sign):
        if step > Mx:
            return arr
        dst = tuple(slice(step, None) if i in S else slice(None) for i in range(d))
        src = tuple(slice(0, Mx + 1 - step) if i in S else slice(None) for i in range(d))
        val = arr[dst] + sign * arr[src]
        arr[dst] = val % p if p is not None else val
        return arr

    for S in subsets:
        if len(S) % 2 == 1:
            arr = shifted_add(arr, S, 1, -1)            # multiply by (1 - x_S)
        else:
            step = 1                                     # multiply by 1/(1-y) = (1+y)(1+y^2)(1+y^4)...
            while step <= Mx:
                arr = shifted_add(arr, S, step, +1)
                step *= 2
    return arr


def crt(residues, primes):
    x, P = 0, 1
    for r, p in zip(residues, primes):
        t = ((r - x) * pow(P, -1, p)) % p
        x += P * t
        P *= p
    return x if x <= P // 2 else x - P


def c_and_L2(k, Mx, primes=None):
    if primes is None:
        g = G_grid(k, Mx)
        get = lambda idx: int(g[idx])
    else:
        gs = [G_grid(k, Mx, p) for p in primes]
        get = lambda idx: crt([int(g[idx]) for g in gs], primes)
    cs, L2 = {}, {}
    for m in range(0, Mx + 1):
        cs[m] = get((m,) * (2 * k))
        if m >= 1:
            s = 0
            for eps in itertools.product((0, 1), repeat=2 * k):
                s += get(tuple(m - e for e in eps))
            L2[m] = s
    return cs, L2


# ===========================================================================
say("=" * 76)
say("1. Proposition 3.1 with the printed (3.1) versus the true divisor sum")
say("=" * 76)
bell = [sp.bell(n) for n in range(8)]
say(f"   Bell numbers B_2, B_4, B_6 = {bell[2]}, {bell[4]}, {bell[6]}")
check("m=1: printed M((c1);1) = -1_{c1>=1}", all(M_printed((c,), 1) == (-1 if c >= 1 else 0) for c in range(8)))
check("m=1: true M((c1);1) = -c1", all(M_true((c,), 1) == -c for c in range(8)))
check("m=2: printed M = 1_{c1>=2} - 1_{c2>=1}",
      all(M_printed((a, b), 2) == (1 if a >= 2 else 0) - (1 if b >= 1 else 0) for a in range(6) for b in range(6)))
check("m=2: true M = binom(c1,2) - c2",
      all(M_true((a, b), 2) == math.comb(a, 2) - b for a in range(8) for b in range(8)))
e = math.e
p1 = 1 - 1 / e
p2 = e**-0.5 + 2 / e - 4 * e**-1.5
say(f"   printed E[M^2k], m=1: 1-1/e = {p1:.8f} for every k;  m=2: e^-1/2 + 2e^-1 - 4e^-3/2 = {p2:.8f}")


# exact c(m,k) from the generating function
cs1, L21 = c_and_L2(1, 12)
cs2, L22 = c_and_L2(2, 30)
say("   computing k=3 grid (6-dim, two 61-bit primes) ...")
PR = [2305843009213693951, int(sp.prevprime(2**61 - 1))]
cs3, L23 = c_and_L2(3, 12, PR)
cs3b, _ = c_and_L2(3, 6, [int(sp.prevprime(2**60)), int(sp.prevprime(2**59))])
check("k=3: CRT with an independent prime pair gives the same c(m,3), m<=6",
      all(cs3[m] == cs3b[m] for m in range(7)))
C = {1: cs1, 2: cs2, 3: cs3}
check("c(1,k) = Bell(2k) for k=1,2,3", C[1][1] == 2 and C[2][1] == 15 and C[3][1] == 203)
check("c(m,1) = 2 for m=1..12", all(cs1[m] == 2 for m in range(1, 13)))
rem31 = all(cs2[m] * 3 == 64 * m**3 - 135 * m**2 + 182 * m - 66 for m in range(1, 31))
check("Remark 3.1: c(m,2) = (64m^3-135m^2+182m-66)/3 for m = 1..30", rem31)
say("   m : c(m,1)  c(m,2)  c(m,3)")
for m in range(1, 7):
    say(f"   {m} : {cs1[m]:>5} {cs2[m]:>7} {cs3[m]:>14}")

say("   E[M_true(X;m)^2k] by truncated Poisson sums vs c(m,k) (generating function):")
say(f"   {'m':>2} {'k':>2} {'c(m,k)':>14} {'E[M_true^2k]':>16} {'E[M_printed^2k]':>16}")
rows1 = []
for m in range(1, 5):
    for k in (1, 2, 3):
        Et = poisson_expect(m, lambda c, m=m, k=k: float(M_true(c, m)) ** (2 * k))
        Ep = poisson_expect(m, lambda c, m=m, k=k: float(M_printed(c, m)) ** (2 * k))
        rows1.append((m, k, C[k][m], Et, Ep))
        say(f"   {m:>2} {k:>2} {C[k][m]:>14} {Et:>16.6f} {Ep:>16.8f}")
check("E[M_true^2k] = c(m,k) (rel. err < 1e-9) for m<=4, k<=3",
      all(abs(r[3] / r[2] - 1) < 1e-9 for r in rows1))
pm = {1: 1, 2: 2, 3: 3, 4: 5}
check("|M_printed| <= p(m), so E[M_printed^2k] <= p(m)^2k, and E[M_printed^2k] != c(m,k) for m<=4, k<=3",
      all(r[4] <= pm[r[0]] ** (2 * r[1]) and abs(r[4] - r[2]) > 0.5 for r in rows1))
check("printed m=1 value equals 1-1/e and m=2 value equals e^-1/2+2e^-1-4e^-3/2",
      abs(rows1[0][4] - p1) < 1e-12 and abs(rows1[3][4] - p2) < 1e-12)

say("-" * 76)
say("   Recurrence on p. 21 versus the Pascal shift identity")
ok_pascal, ok_printed_rec_on_printed, bad_printed_rec_on_true = True, True, 0
rng = np.random.default_rng(7)
for _ in range(400):
    m = int(rng.integers(1, 7))
    c = tuple(int(x) for x in rng.integers(0, 5, size=m))
    i = int(rng.integers(1, m + 1))
    ce = list(c)
    ce[i - 1] += 1
    ce = tuple(ce)
    ok_pascal &= M_true(ce, m) == M_true(c, m) - M_true(c, m - i)
    ci = list(c)
    ci[i - 1] = 0
    ci = tuple(ci)
    rhs_p = M_printed(c, m) + (-1) ** (c[i - 1] + 1) * M_printed(ci, m - i * (c[i - 1] + 1))
    ok_printed_rec_on_printed &= M_printed(ce, m) == rhs_p
    rhs_t = M_true(c, m) + (-1) ** (c[i - 1] + 1) * M_true(ci, m - i * (c[i - 1] + 1))
    bad_printed_rec_on_true += M_true(ce, m) != rhs_t
check("Pascal shift M(e_i+c; m) = M(c; m) - M(c; m-i) for the true M (400 random cases)", ok_pascal)
check("p.21 recurrence holds for the printed (unweighted) M (400 cases)", ok_printed_rec_on_printed)
say(f"   p.21 recurrence fails for the true M in {bad_printed_rec_on_true}/400 cases;"
    f" e.g. M((2);1) = {M_true((2,), 1)}, recurrence gives {M_true((1,), 1) + M_true((0,), -1)}")
check("p.21 recurrence fails for the true M in some case", bad_printed_rec_on_true > 0)
# T_i bound
okT = True
for m in range(1, 5):
    for k in (1, 2):
        for i in range(1, m + 1):
            Ti = poisson_expect(m, lambda c, m=m, k=k, i=i: c[i - 1] * float(M_true(c, m)) ** (2 * k))
            okT &= Ti <= 2 ** (2 * k) / i * C[k][m] * (1 + 1e-9)
check("T_i = E[X_i M(X;m)^2k] <= (4^k/i) c(m,k) for m<=4, k<=2, all i", okT)

# ===========================================================================
say("=" * 76)
say("2. The binomial ratio on p. 20 (j = 1)")
say("=" * 76)


def logratio(q, c):
    """log of binom(q,c) c! / (q-1)^c = sum_{i<c} log(1-i/q) - c log(1-1/q), accurately."""
    if c >= q:
        return math.lgamma(q + 1) - c * math.log(q - 1) if c == q else -math.inf
    return float(np.sum(np.log1p(-np.arange(c) / q))) - c * math.log1p(-1 / q)


def ratio(q, c):
    return math.exp(logratio(q, c))


for q in (10**4, 10**6):
    say(f"   q = {q}:")
    for c in (10, int(q**0.5), 3 * int(q**0.5), q // 10, q):
        R = ratio(q, c)
        say(f"     c1 = {c:>8}: ratio = {R:.6e}   1-c(c-3)/(2q) = {1 - c*(c-3)/(2*q):.6f}   "
            f"|ratio-1|*q/c = {abs(R-1)*q/c:.3e}")
q = 10**6
cs_small = np.arange(2, 200)
errs = [abs(ratio(q, int(c)) - (1 - c * (c - 3) / (2 * q))) * q**2 / c**4 for c in cs_small]
check("ratio = 1 - c(c-3)/(2q) + O(c^4/q^2) (constant < 1 for c<=200, q=10^6)", max(errs) < 1)
check("|ratio - 1| q/c is unbounded (not O(c/q)): > 300 at c = 3 sqrt q, q=10^6",
      abs(ratio(q, 3000) - 1) * q / 3000 > 300)
check("at c1 = q the ratio is ~ e^{1-q} sqrt(2 pi q)",
      abs(logratio(q, q) - (1 - q + 0.5 * math.log(2 * math.pi * q))) < 1e-3)
okQ = True
for m in range(1, 5):
    for k in (1, 2):
        Q2 = poisson_expect(m, lambda c, m=m, k=k: c[0] * (c[0] - 1) * float(M_true(c, m)) ** (2 * k))
        Sh = poisson_expect(m, lambda c, m=m, k=k: float(M_true((c[0] + 2,) + tuple(c[1:]), m)) ** (2 * k))
        Sp = poisson_expect(m, lambda c, m=m, k=k: float(M_true(c, m) - 2 * M_true(c, m - 1) + M_true(c, m - 2)) ** (2 * k))
        okQ &= abs(Q2 - Sh) < 1e-8 * max(1, Sh) and abs(Sh - Sp) < 1e-8 * max(1, Sh) and Sh <= 4 ** (2 * k) * C[k][m]
check("E[X1(X1-1) M^2k] = E[M(X+2e1)^2k] = E[(M_m - 2M_{m-1} + M_{m-2})^2k] <= 16^k c(m,k) (m<=4,k<=2)", okQ)

# ===========================================================================
say("=" * 76)
say("3. Section 2.5: the h = 2 moment L_2(m,k) = lim_q Poly_q(n,m,2;k)")
say("=" * 76)
check("L_2 identity: E[(M(X;m)+M(X;m-1))^2k] = sum_eps G(m-eps), m<=4, k=1,2",
      all(abs(poisson_expect(m, lambda c, m=m, k=k: float(M_true(c, m) + M_true(c, m - 1)) ** (2 * k))
              / {1: L21, 2: L22}[k][m] - 1) < 1e-9 for m in range(1, 5) for k in (1, 2)))
say(f"   {'m':>3} {'c(m,3)':>26} {'L_2(m,3)':>22} {'L_2/c':>12}")
for m in range(1, 13):
    say(f"   {m:>3} {cs3[m]:>26} {L23[m]:>22} {L23[m]/cs3[m]:>12.4e}")
ms = np.arange(6, 13)
sl_L = np.diff(np.log([float(L23[m]) for m in ms])) / np.diff(np.log(ms))
sl_c = np.diff(np.log([float(cs3[m]) for m in ms])) / np.diff(np.log(ms))
say("   local log-log slopes (m = 6..12):")
say("     c(m,3):   " + " ".join(f"{s:6.2f}" for s in sl_c) + "    (degree 2^5-6-1 = 25)")
say("     L_2(m,3): " + " ".join(f"{s:6.2f}" for s in sl_L) + "    (corrected 2^5-12-1 = 19; paper 2^5-18-1 = 13)")
# polynomial degree in m: finite differences of L_2(m,3) of order 20 vanish? not enough points; use the ratio
ratio_slope = np.diff(np.log([float(L23[m]) / float(cs3[m]) for m in ms])) / np.diff(np.log(ms))
say("     L_2/c:    " + " ".join(f"{s:6.2f}" for s in ratio_slope) + "    (corrected -2k = -6; paper -4k = -12)")
check("all L_2(m,3) > 0 for 1<=m<=12", all(L23[m] > 0 for m in range(1, 13)))
check("local exponent of L_2(m,3) exceeds 13 for m >= 7 and increases with m",
      all(s > 13 for s in sl_L[1:]) and all(np.diff(sl_L) > 0))
say("   k = 2: L_2(m,2) = " + ", ".join(str(L22[m]) for m in range(1, 13)))
check("k = 2: L_2(m,2) = 15 m^2 - 51 m + 65 exactly for 3 <= m <= 30 (order m^2, not m^-1 or m^-5)",
      all(L22[m] == 15 * m * m - 51 * m + 65 for m in range(3, 31)))

say("-" * 76)
say("   The theta ~ 1/2 region with r = 1 - 1/m, rescaled theta_j = 1/2 + w_j/m, |w_j| <= 1:")
say("   even I: Z(r^|I| e(theta_I)) -> m/(|I| - 2 pi i w_I);  odd I: 1/Z -> 2;")
say("   1 + r e(theta_j) = (1 - 2 pi i w_j)/m + O(m^-2);  e(-m theta_j) = e(-m/2) e(-w_j).")
say("   Region contribution ~ const * m^{(2^{2k-1}-1) - 2k - 2k} * J_k,")
say("   J_k = int_{[-1,1]^{2k}} prod_{|I| even} (|I| - 2 pi i w_I)^{-1} prod_j e(-w_j)(1 - 2 pi i w_j) dw.")
from scipy.stats import qmc


def J_integral(k, log2n=18, seeds=(1, 2, 3, 4, 5, 6, 7, 8), with_r=True):
    d = 2 * k
    evens = [S for r in range(2, d + 1, 2) for S in itertools.combinations(range(d), r)]
    A = np.zeros((len(evens), d))
    for a, S in enumerate(evens):
        A[a, list(S)] = 1
    sizes = A.sum(1)
    vals = []
    for sd in seeds:
        pts = qmc.Sobol(d, scramble=True, seed=sd).random_base2(log2n) * 2 - 1
        acc = 0j
        for ch in range(0, len(pts), 1 << 15):
            w = pts[ch:ch + (1 << 15)]
            wI = w @ A.T
            f = np.prod(1.0 / (sizes - 2j * np.pi * wI), axis=1)
            lin = (1 - 2j * np.pi * w) if with_r else (-2j * np.pi * w)
            f = f * np.prod(np.exp(-2j * np.pi * w) * lin, axis=1)
            acc += f.sum()
        vals.append(acc / len(pts) * 2.0**d)
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


J2, e2 = J_integral(2)
J2r1, e2r1 = J_integral(2, with_r=False)
say(f"   k=2: J_2 = {J2.real:+.4e} (+- {e2:.1e});  with the paper's factor 1+e(theta) (r=1): {J2r1.real:+.4e} (+- {e2r1:.1e})")
check("k=2: first-order terms do not cancel: J_2 != 0 even with r = 1 (symmetry argument fails)",
      abs(J2r1) > 100 * e2r1 and abs(J2) > 100 * e2)
say("   The box |w_j| <= 1 is arbitrary; the w-integral is not absolutely convergent over R^{2k},")
say("   so a 'region contribution' is cutoff dependent and does not by itself give the order of Poly_q(n,m,2;k).")
J3, e3 = J_integral(3)
say(f"   k=3: J_3 = {J3.real:+.3e} {J3.imag:+.3e}i (+- {e3:.1e}): not resolved from 0 at this precision.")

say("-" * 76)
say("   Exponent of the exact L_2(m,3): fit local slope s(m) = D + a/m + b/m^2 (+ c/m^3), m >= lo")


def extrap(f):
    ms_ = np.array([float(m) for m in sorted(f) if m - 1 in f and m + 1 in f and m - 1 >= 1])
    s_ = np.array([math.log(float(f[int(m) + 1]) / float(f[int(m) - 1])) / math.log((m + 1) / (m - 1)) for m in ms_])
    est = []
    for nterm in (2, 3, 4):
        for lo in (4, 5, 6):
            sel = ms_ >= lo
            X = np.vstack([ms_[sel] ** (-p) for p in range(nterm)]).T
            est.append(np.linalg.lstsq(X, s_[sel], rcond=None)[0][0])
    return min(est), max(est)


ex = {"c(m,2) [true 3]": extrap({m: cs2[m] for m in range(1, 13)}),
      "L_2(m,2) [true 2]": extrap({m: L22[m] for m in range(1, 13)}),
      "c(m,3) [true 25]": extrap({m: cs3[m] for m in range(1, 13)}),
      "L_2(m,3)": extrap({m: L23[m] for m in range(1, 13)})}
for kk, (lo_, hi_) in ex.items():
    say(f"     {kk:<20}: estimated exponent in [{lo_:.2f}, {hi_:.2f}]")
check("method recovers the known exponents 3 and 2 (k=2) within 0.3 and brackets 25 for c(m,3) within [24,29]",
      abs(ex["c(m,2) [true 3]"][0] - 3) < 0.3 and abs(ex["c(m,2) [true 3]"][1] - 3) < 0.3
      and abs(ex["L_2(m,2) [true 2]"][0] - 2) < 0.3 and abs(ex["L_2(m,2) [true 2]"][1] - 2) < 0.3
      and 24 <= ex["c(m,3) [true 25]"][0] and ex["c(m,3) [true 25]"][1] <= 29)
check("L_2(m,3): estimated exponent lies in [19, 23], far above the 13 of Section 2.5",
      19 <= ex["L_2(m,3)"][0] and ex["L_2(m,3)"][1] <= 23)
say(f"TOTAL FAILURES: {check.failures}")

import json
with open(os.path.join(ROOT, "data", "gkm_exact_values.json"), "w", encoding="utf-8") as fh:
    json.dump({"c1": {m: str(v) for m, v in cs1.items()}, "c2": {m: str(v) for m, v in cs2.items()},
               "c3": {m: str(v) for m, v in cs3.items()}, "L2_2": {m: str(v) for m, v in L22.items()},
               "L2_3": {m: str(v) for m, v in L23.items()},
               "printed": [[r[0], r[1], r[4]] for r in rows1]}, fh, indent=1)
with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
if "--nofig" in sys.argv:
    sys.exit(1 if check.failures else 0)

# ===========================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.7})
colk = {1: "#1f4e79", 2: "#2e8b57", 3: "#c0392b"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.55))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.74, wspace=0.3)

handles = []
mm = list(range(1, 5))
for k in (1, 2, 3):
    h, = ax1.plot(mm, [C[k][m] for m in mm], "o-", color=colk[k], ms=3.5, lw=1.3,
                  label=rf"$c(m,{k})=\mathbb{{E}}[M(\mathbf{{X}};m)^{{{2*k}}}]$")
    handles.append(h)
for k in (1, 2, 3):
    h, = ax1.plot(mm, [r[4] for r in rows1 if r[1] == k], "s--", color=colk[k], ms=3, lw=1.0, alpha=0.85,
                  label=rf"printed (3.1): $\mathbb{{E}}[M^{{{2*k}}}]$")
    handles.append(h)
ax1.set_yscale("log")
ax1.set_xticks(mm)
ax1.set_xlabel(r"$m$")
ax1.set_ylabel("moment")
ax1.set_ylim(0.15, 3e8)
ax1.text(0.04, 0.95, "(a)", transform=ax1.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

def local_slopes(f, mrange):
    return [math.log(float(f[m + 1]) / float(f[m - 1])) / math.log((m + 1) / (m - 1)) for m in mrange]


mS = list(range(3, 12))
h1, = ax2.plot(mS, local_slopes(L23, mS), "o-", color=colk[3], ms=3.5, lw=1.3,
               label=r"local exponent of $L_2(m,3)$ (exact)")
h1b, = ax2.plot(mS, local_slopes(cs3, mS), "s-", color=colk[1], ms=3, lw=1.1,
                label=r"local exponent of $c(m,3)$ (exact)")
h2, = ax2.axhline(19, color="k", lw=1.0, ls="--", label=r"$2^{2k-1}-4k-1=19$ (corrected)"),
h3, = ax2.axhline(13, color="#7f7f7f", lw=1.0, ls=":", label=r"$2^{2k-1}-6k-1=13$ (Section 2.5)"),
ax2.axhline(25, color=colk[1], lw=0.8, ls="-.")
ax2.text(11.2, 25.4, "25", fontsize=7, color=colk[1], ha="right", va="bottom")
ax2.set_xlim(2.5, 11.5)
ax2.set_ylim(0, 27.5)
ax2.set_xlabel(r"$m$")
ax2.set_ylabel(r"$d\log f/d\log m$")
ax2.text(0.04, 0.04, "(b)", transform=ax2.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
ins = ax2.inset_axes([0.52, 0.1, 0.44, 0.3])
qq = 10**4
cc = np.unique(np.logspace(0, 4, 300).astype(int))
ins.plot(cc / qq**0.5, [ratio(qq, int(c)) for c in cc], color=colk[1], lw=1.1)
ins.plot(cc / qq**0.5, np.exp(-(cc / qq**0.5) ** 2 / 2), color="#e08e0b", lw=0.9, ls="--")
ins.set_xscale("log")
ins.set_yscale("log")
ins.set_ylim(1e-12, 3)
ins.set_xlim(1e-2, 100)
ins.tick_params(labelsize=6, length=2, pad=1)
ins.set_xlabel(r"$c_1/\sqrt{q}$", fontsize=6.5, labelpad=0)
ins.set_facecolor("white")
hb, = ax2.plot([], [], color=colk[1], lw=1.1, label=r"inset: $\binom{q}{c_1}c_1!/(q-1)^{c_1}$, $q=10^4$")
hc, = ax2.plot([], [], color="#e08e0b", lw=0.9, ls="--", label=r"inset: $e^{-c_1^2/(2q)}$")
fig.legend(handles=handles + [h1, h1b, h2, h3, hb, hc], loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=2.2, borderaxespad=0.2, fontsize=7)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
sys.exit(1 if check.failures else 0)
