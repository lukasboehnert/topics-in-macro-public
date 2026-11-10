# %% [markdown]
# # In-class exercise: collateral, sudden stops and a tax on debt (student version)
#
# **How this works.** Five short parts on the lecture's model. Part 1 is the perfect-foresight model from the
# first half of the lecture, by hand and in three lines of code. Parts 2-5 use the stochastic model, packed into
# one function `model(...)` that solves it, simulates it and returns the lecture's moments. Gaps in the student
# version are marked `# TODO` and contain `...`; fill them in, run the cell, and the `check(...)` line under it
# prints ✓ when your number matches.
#
# | Part | Question |
# | --- | --- |
# | 1 | Perfect foresight: how large must a shock be to trigger a crisis? |
# | 2 | Write the asset-pricing equation and reproduce the lecture's moments |
# | 3 | Tighter or looser credit: vary $\kappa$ |
# | 4 | Does the tax still help if the collateral is valued at a fixed price? |
# | 5 | Your turn: one experiment of your own |
#
# **Run** cell by cell in VS Code, or `uv run exercise_SS_student.py` (about two minutes). The solution,
# `exercise_SS.py`, is posted after class.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

try:
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures_exercise"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f"


def check(name, value, target, tol=0.05):
    """Print whether your result matches the solution."""
    try:
        value = round(float(value), 3)
        ok = abs(value - target) < tol
    except (TypeError, ValueError):
        ok = False
    print(f"{'✓' if ok else '✗'} {name}: yours = {value}, should be about {target}")


# %% [markdown]
# ## Part 1. Perfect foresight
# Log utility, $\beta(1+r)=1$, $r=4\%$, dividend $\alpha A$ with $\alpha=0.3$, $A=1$, so $q^*=\alpha A/r=7.5$;
# $\kappa=0.10$; the economy starts in the unconstrained steady state with debt $d_0$. Output falls to $A^L$ in
# period 1 and returns to $A$ for good.
#
# (a) *On paper.* Without the constraint, how much does the economy borrow in period 1? Show that the constraint
# binds if and only if $A - A^L > (1+r)(\kappa q^* - d_0)$.
#
# (b) If it binds only in period 1: $d_2=\kappa q_1$, $q_1=q^*c_1/c_2$, $c_1=A^L+d_2-(1+r)d_0$, $c_2=A-rd_2$.
# With $x=c_1/c_2$, show that $r\kappa q^*x^2+(\kappa q^*-A)x+A^L-(1+r)d_0=0$, and code the root $x<1$.
#
# (c) Compute the threshold and the fall in consumption and in $q$ for a 20% output loss, for $d_0=0.6$ and
# for $d_0=0.5$. What does the initial debt level do to the economy's fragility?

# %%
r, alpha, kappa_pf, A_ss = 0.04, 0.3, 0.10, 1.0
q_star = alpha * A_ss / r


def crisis(AL, d0):
    """Return (threshold %, % fall in c, % fall in q) for an output drop to AL."""
    thresh = 100 * (1 + r) * (kappa_pf * q_star - d0)
    c_star = A_ss - r * d0
    if d0 + (A_ss - AL) / (1 + r) <= kappa_pf * q_star:                  # slack: smooth with borrowing
        c1 = c_star - r / (1 + r) * (A_ss - AL)
        return thresh, 100 * (1 - c1 / c_star), 0.0
    a, b, c = ..., ..., ...   # TODO the three coefficients
    x = ...                       # TODO the root below one
    c2 = A_ss - r * kappa_pf * q_star * x
    return thresh, 100 * (1 - x * c2 / c_star), 100 * (1 - x)


for d0 in (0.6, 0.5):
    t, dc, dq = crisis(0.8, d0)
    print(f"d0 = {d0}: threshold {t:.1f}% of output; a 20% loss lowers c by {dc:.1f}% and q by {dq:.1f}%")
check("threshold, d0 = 0.6 (%)", crisis(0.8, 0.6)[0], 15.6, tol=0.1)
check("fall in c, 20% loss, d0 = 0.6 (%)", crisis(0.8, 0.6)[1], 22.31, tol=0.1)
check("fall in c, 20% loss, d0 = 0.5 (%)", crisis(0.8, 0.5)[1], 0.79, tol=0.1)

# %% [markdown]
# ## Part 2. The stochastic model in one function
# $u'(c_t)=\beta(1+r)(1+\tau)E_tu'(c_{t+1})+\mu_t$, $\;d_{t+1}\le\kappa q_t$, $\;c_t=A_t+d_{t+1}-(1+r)d_t$, and
# the asset-pricing equation (write it in the `# TODO` line):
# $$q_t\,u'(c_t)=\beta E_t\left[u'(c_{t+1})(\alpha A_{t+1}+q_{t+1})+\kappa\mu_{t+1}q_{t+1}\right]$$
# `model()` returns the lecture's moments. Option `fixed_q`: value the collateral at a fixed price $\bar q$
# instead of the market price (Part 4).

# %%
def rouwenhorst(n, rho, sd):
    p = (1 + rho) / 2
    P = np.array([[p, 1 - p], [1 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:-1, :-1] += p * P
        Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1] /= 2
        P = Z
    s = sd / np.sqrt(1 - rho**2) * np.sqrt(n - 1)
    return np.linspace(-s, s, n), P


def bisect(f, lo, hi, n=50):
    for _ in range(n):
        mid = (lo + hi) / 2
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return (lo + hi) / 2


def model(kappa=0.14, sigma=2.0, beta=0.91, sd_e=0.05, tau=0.0, fixed_q=None, T=60_000, maxit=600):
    R, rho, NA, ND = 1 + r, 0.85, 9, 400
    logA, P = rouwenhorst(NA, rho, sd_e)
    A = np.exp(logA)
    dg = np.linspace(-0.3, 0.55, ND)
    DD, AA = np.meshgrid(dg, A, indexing="ij")
    up = lambda c: c ** -sigma                                                       # noqa: E731
    u = lambda c: np.log(c) if sigma == 1 else c ** (1 - sigma) / (1 - sigma)       # noqa: E731
    interp = lambda x, F: np.stack([np.interp(x[:, j], dg, F[:, j]) for j in range(NA)], 1)  # noqa: E731

    def solve(tau):
        C, Q, M = np.maximum(AA - r * DD, 0.05), np.full((ND, NA), beta * alpha / (1 - beta)), np.zeros((ND, NA))
        lo, hi = np.maximum(dg[0], R * DD - AA + 1e-6), np.full((ND, NA), dg[-1])
        D_old = None
        for it in range(maxit):
            prem = kappa * M * Q if fixed_q is None else 0.0        # collateral premium only if q is the collateral
            X = (up(C) * (alpha * A + Q) + prem) @ P.T
            EU = up(C) @ P.T
            cons = lambda d1: np.maximum(AA + d1 - R * DD, 1e-10)                    # noqa: E731
            price = (lambda d1: beta * interp(d1, X) / up(cons(d1))) if fixed_q is None else \
                (lambda d1: np.full_like(d1, fixed_q))                               # noqa: E731
            d_u = bisect(lambda d1: up(cons(d1)) - beta * R * (1 + tau) * interp(d1, EU), lo, hi)
            slack = lambda d1: kappa * price(d1) - d1                                # noqa: E731
            bind = slack(d_u) < 0
            cg = AA[:, None, :] + dg[None, :, None] - R * DD[:, None, :]
            qg = beta * X[None] / up(np.maximum(cg, 1e-10)) if fixed_q is None else fixed_q
            okg = (cg > 0) & (kappa * qg - dg[None, :, None] >= 0) & (dg[None, :, None] <= d_u[:, None, :])
            lo_b = dg[(ND - 1) - np.argmax(okg[:, ::-1, :], axis=1)]
            D = np.where(bind, bisect(slack, np.minimum(lo_b, d_u), d_u), d_u)
            c = cons(D)
            Qn = ...                                         # TODO the asset price
            Mn = np.where(bind, np.maximum(up(c) - beta * R * interp(D, EU), 0), 0)
            err = (np.abs(c - C) + np.abs(Qn - Q)).max()
            if it == maxit - 1:
                return dict(C=(c + C) / 2, Q=(Qn + Q) / 2, M=(Mn + M) / 2, D=(D + D_old) / 2)
            C, Q, M, D_old = c, Qn, Mn, D
            if err < 1e-7:
                break
        return dict(C=C, Q=Q, M=M, D=D)

    def value(sol):
        k = np.clip(np.searchsorted(dg, sol["D"]) - 1, 0, ND - 2)
        w = (sol["D"] - dg[k]) / (dg[k + 1] - dg[k])
        cols = np.arange(NA)[None, :]
        V = u(sol["C"]) / (1 - beta)
        for _ in range(3000):
            EV = V @ P.T
            Vn = u(sol["C"]) + beta * ((1 - w) * EV[k, cols] + w * EV[k + 1, cols])
            if np.abs(Vn - V).max() < 1e-9:
                break
            V = Vn
        return Vn

    rng = np.random.default_rng(2026)
    cumP, draws = P.cumsum(1), rng.random(T)
    st = np.empty(T, int)
    st[0] = NA // 2
    for t in range(1, T):
        st[t] = min(np.searchsorted(cumP[st[t - 1]], draws[t]), NA - 1)

    def simulate(sol):
        d = np.empty(T + 1)
        d[0] = 0.2
        for t in range(T):
            d[t + 1] = np.interp(d[t], dg, sol["D"][:, st[t]])
        bind = np.empty(T, bool)
        for k in range(NA):
            m = st == k
            bind[m] = np.interp(d[:-1][m], dg, sol["M"][:, k]) > 1e-8
        ca = -(d[1:] - d[:-1]) / A[st]
        ss = bind & (ca > ca.mean() + ca.std())
        return d[:-1], bind, ss

    out = {}
    sol0 = solve(0.0)
    d, bind, ss = simulate(sol0)
    out.update({"debt/GDP": 100 * np.mean(d / A[st]), "binds": 100 * bind.mean(), "sudden stops": 100 * ss.mean()})
    if tau > 0:
        sol1 = solve(tau)
        _, bind1, ss1 = simulate(sol1)
        V0, V1 = value(sol0), value(sol1)
        v0 = np.array([np.interp(x, dg, V0[:, k]) for x, k in zip(d[::25], st[::25])])
        v1 = np.array([np.interp(x, dg, V1[:, k]) for x, k in zip(d[::25], st[::25])])
        gain = np.exp((1 - beta) * (v1 - v0)) - 1 if sigma == 1 else (v1 / v0) ** (1 / (1 - sigma)) - 1
        out.update({"sudden stops, taxed": 100 * ss1.mean(), "welfare gain of tax": 100 * gain.mean()})
    return pd.Series(out)


base = model(tau=0.025)
print(base.round(3).to_string())
check("constraint binds, % of years", base["binds"], 9.45, tol=0.3)
check("sudden stops, % of years", base["sudden stops"], 7.72, tol=0.3)
check("welfare gain of a 2.5% tax, %", base["welfare gain of tax"], 0.052, tol=0.01)

# %% [markdown]
# ## Part 3. Tighter or looser credit
# Solve the model for $\kappa\in\{0.12, 0.14, 0.16\}$ (no tax). How do average debt and the frequency of
# sudden stops change? Debt rises with $\kappa$; why do crises become *rarer*?

# %%
res_k = ...          # TODO one model() per kappa
print(res_k.round(2).to_string())
check("sudden stops, kappa = 0.12", res_k.loc[0.12, "sudden stops"], 8.72, tol=0.3)
check("debt / GDP, kappa = 0.16", res_k.loc[0.16, "debt/GDP"], 43.0, tol=0.3)

# %% [markdown]
# ## Part 4. No price in the collateral
# Replace $\kappa q_t$ by $\kappa\bar q$ with $\bar q = 3.25$, the lecture's average price (`fixed_q=3.25`), and
# evaluate the same 2.5% tax. How often does the constraint bind now, and are there still sudden stops? Does the
# tax still raise welfare? Explain with the pecuniary externality.

# %%
r4 = ...                                                  # TODO
print(r4.round(3).to_string())
check("constraint binds, fixed price (% of years)", r4["binds"], 95.26, tol=0.5)
check("welfare gain of a 2.5% tax, fixed price (%)", r4["welfare gain of tax"], -0.016, tol=0.005)

# %% [markdown]
# ## Part 5. Your turn
# Pick one: (a) more volatile output, `sd_e=0.06`; (b) more risk-averse households, `sigma=3`; (c) more
# impatient households, `beta=0.90`. Predict first, then run `model(..., tau=0.025)`. Does a 2.5% tax help more
# or less than in the baseline? Two sentences on why.

# %%
r5 = ...                                                                             # TODO
print(r5.round(3).to_string())
