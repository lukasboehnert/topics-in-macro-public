# %% [markdown]
# # Week 3 lab (fast version): financial crises, collateral and macroprudential policy
#
# Part A of the Week 3 lab (part B, the planner and the tax on debt, is `overborrowing_*.py`, Bianchi 2011):
# a small open economy borrows from abroad against the market value of an asset, $d_{t+1} \le \kappa q_t$.
# This is the collateral constraint of Mendoza (2010) and Bianchi and Mendoza (2018), stripped down to an
# endowment economy so that the whole model fits in one script.
#
# Same cells, calibration and numbers as `week3_simple.py`, written compactly and vectorised. This script writes
# the part A figures on the slides to `figures/`. About a minute.
#
# Cells: F0 facts (needs `data/JSTdatasetR6.xlsx`, see `data/fetch_jst.py`), M0 perfect foresight,
# M1 the stochastic model, M2 accuracy (a side check), M3 simulation, M4 a sudden stop.

# %%
from pathlib import Path
import time

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
DATA, FIGS = HERE / "data", HERE / "figures"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


T0 = time.time()

# %% [markdown]
# ## F0. Facts: the anatomy of credit booms (Mendoza and Terrones 2012, on JST data)
# A credit boom: the cyclical part of log real credit per capita (HP filter, $\lambda=100$) exceeds 1.75 of
# its country-specific standard deviation. The peak is the year of the largest deviation within a boom.
# Event window: average cyclical components of credit, GDP, consumption, investment and house prices, and the
# current account, from 4 years before to 4 years after the peak. Plus: how often a boom ends in a crisis.

# %%
def hp_cycle(y, lam=100.0):
    """Cyclical component of the Hodrick-Prescott filter (dense solve, fine for annual series)."""
    n = len(y)
    D = np.diff(np.eye(n), 2, axis=0)
    return y - np.linalg.solve(np.eye(n) + lam * D.T @ D, y)


jst_file = DATA / "JSTdatasetR6.xlsx"
if jst_file.exists():
    jst = pd.read_excel(jst_file, sheet_name=0).sort_values(["iso", "year"])
    jst = jst[(jst.year >= 1950)].copy()                                   # post-war: no wartime gaps
    jst["l_credit"] = np.log(jst.tloans / jst.cpi / jst["pop"])
    jst["l_gdp"] = np.log(jst.rgdpmad)
    jst["l_cons"] = np.log(jst.rconpc)
    jst["l_inv"] = np.log(jst.iy * jst.rgdpmad)
    jst["l_house"] = np.log(jst.hpnom / jst.cpi)
    jst["ca_gdp"] = 100 * jst.ca / jst.gdp
    VARS = {"l_credit": "real credit", "l_gdp": "real GDP", "l_cons": "real consumption",
            "l_inv": "real investment", "l_house": "real house prices", "ca_gdp": "current account / GDP"}
    for v in VARS:
        jst[v + "_c"] = np.nan
        for iso, g in jst.groupby("iso"):
            ok = g[v].notna()
            if ok.sum() > 20:
                cyc = hp_cycle(g.loc[ok, v].values)
                jst.loc[g.index[ok.values], v + "_c"] = cyc * (1 if v == "ca_gdp" else 100)
    W = 4
    peaks = []
    for iso, g in jst.groupby("iso"):
        c = g.l_credit_c.values
        boom = c > 1.75 * np.nanstd(c)
        t = 0
        while t < len(c):                                                  # one peak per run of boom years
            if boom[t]:
                s = t
                while t < len(c) and boom[t]:
                    t += 1
                peaks.append((iso, g.year.values[s + np.nanargmax(c[s:t])]))
            t += 1
    ev = {v: [] for v in VARS}
    crisis_after = []
    for iso, yr in peaks:
        g = jst[jst.iso == iso].set_index("year")
        for v in VARS:
            ev[v].append([g[v + "_c"].get(yr + h, np.nan) for h in range(-W, W + 1)])
        crisis_after.append(g.crisisJST.reindex(range(yr - 1, yr + 3)).max() == 1)
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
    for ax, v in zip(axs.flat, VARS):
        m = np.nanmean(np.array(ev[v]), 0)
        ax.plot(range(-W, W + 1), m, "o-", ms=3, color=RED if v == "l_credit" else BLUE)
        ax.axhline(0, color=GREY, lw=0.6)
        ax.axvline(0, color=GREY, lw=0.6, ls=":")
        ax.set(title=VARS[v], xlabel="years from credit-boom peak")
    axs[0, 0].set_ylabel("% deviation from trend")
    axs[1, 2].set_ylabel("pp of GDP, from trend")
    show(fig, "f1_booms")
    print(f"{len(peaks)} credit booms in {jst.iso.nunique()} countries since 1950; "
          f"{100 * np.mean(crisis_after):.0f}% end in a banking crisis within [t-1, t+2]")
else:
    print("No JST data: run `uv run data/fetch_jst.py` first (cell F0 skipped).")

# %% [markdown]
# ## M0. Perfect foresight: small shocks vs large shocks 
# Log utility, $\beta(1+r)=1$, asset in fixed supply $k=1$ paying $\alpha A_t$, collateral $d_{t+1}\le\kappa q_t$.
# The economy starts in the unconstrained steady state with debt $d_0$ and $q^*=\alpha A/r$. In period 1
# output falls to $A^L$, then returns to $A$ for good.
#
# - **Unconstrained:** borrow to smooth: consumption falls only by the annuity value of the loss.
# - **Binding** (if $d_0 + (A-A^L)/(1+r) > \kappa q^*$): $d_2=\kappa q_1$, $q_1 = q^* c_1/c_2$ (asset pricing
#   with $\mu_2=0$), $c_1 = A^L + d_2 - (1+r)d_0$, $c_2 = A - r d_2$. With $x\equiv c_1/c_2$:
#   $\; r\kappa q^* x^2 + (\kappa q^* - A)x + A^L-(1+r)d_0 = 0$, and $x<1$ is the root we want.

# %%
r_pf, alpha_pf, kappa_pf, A_pf, d0 = 0.04, 0.3, 0.10, 1.0, 0.60
q_star = alpha_pf * A_pf / r_pf
c_star = A_pf - r_pf * d0


def perfect_foresight(AL):
    d2 = d0 + (A_pf - AL) / (1 + r_pf)                         # unconstrained borrowing
    if d2 <= kappa_pf * q_star:
        c1 = A_pf - r_pf * d0 - r_pf / (1 + r_pf) * (A_pf - AL)
        return dict(c1=c1, q1=q_star, d2=d2, bind=False)
    a, b, c = r_pf * kappa_pf * q_star, kappa_pf * q_star - A_pf, AL - (1 + r_pf) * d0
    x = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)            # the root below one
    c2 = A_pf - r_pf * kappa_pf * q_star * x
    return dict(c1=x * c2, q1=q_star * x, d2=kappa_pf * q_star * x, bind=True)


shocks = np.linspace(0, 0.30, 121)
pf = pd.DataFrame([perfect_foresight(A_pf * (1 - s)) for s in shocks], index=100 * shocks)
thresh = 100 * (1 + r_pf) * (kappa_pf * q_star - d0)
d2_free = d0 + shocks * A_pf / (1 + r_pf)                     # debt chosen without the constraint
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(pf.index, pf.d2, color=RED, lw=2, label="with collateral constraint")
ax[0].plot(pf.index, d2_free, color=BLUE, ls="--", label="without constraint")
ax[0].axhline(kappa_pf * q_star, color=GREY, lw=0.6)
ax[0].set(title="Borrowing: debt $d_2$ chosen in period 1", xlabel="output loss in period 1, %", ylabel="level")
ax[1].plot(pf.index, 100 * (pf.q1 / q_star - 1), color=RED, lw=2)
ax[1].axhline(0, color=BLUE, ls="--")
ax[1].set(title="Price of the asset $q_1$", xlabel="output loss in period 1, %", ylabel="% from steady state")
ax[2].plot(pf.index, 100 * (pf.c1 / c_star - 1), color=RED, lw=2)
ax[2].plot(pf.index, -100 * r_pf / (1 + r_pf) * shocks * A_pf / c_star, color=BLUE, ls="--")
ax[2].set(title="Consumption $c_1$", xlabel="output loss in period 1, %", ylabel="% from steady state")
for a in ax:
    a.axvline(thresh, color=GREY, ls=":", lw=1)
ax[0].legend(frameon=False, fontsize=9)
show(fig, "m0_threshold")
print(f"q* = {q_star:.2f}, debt limit kappa q* = {kappa_pf * q_star:.2f}, d0 = {d0}: "
      f"the constraint binds once output falls by more than {thresh:.1f}%")
for s in (0.10, 0.20):
    p = perfect_foresight(A_pf * (1 - s))
    print(f"  {100 * s:.0f}% output loss: c falls {100 * (1 - p['c1'] / c_star):.1f}%, q falls "
          f"{100 * (1 - p['q1'] / q_star):.1f}%")

# phase diagram: q_{t+1} = q_t / beta_tilde - r q*
fig, ax = plt.subplots(figsize=(5.2, 4))
qq = np.linspace(0, 1.25 * q_star, 50)
beta_pf = 1 / (1 + r_pf)
ax.plot(qq, qq, color=GREY, lw=0.8, label="45°")
ax.plot(qq, qq / beta_pf - r_pf * q_star, color=BLUE, label=r"$q_{t+1}=q_t/\beta - rq^*$ (slack)")
ax.plot(qq, qq / (0.8 * beta_pf) - r_pf * q_star, color=RED, label=r"$q_{t+1}=q_t/\tilde\beta_t - rq^*$ (binding)")
ax.axvline(q_star, color=GREY, ls=":", lw=0.8)
ax.axhline(q_star, color=GREY, ls=":", lw=0.8)
ax.set(xlim=(0, 1.25 * q_star), ylim=(-r_pf * q_star - 0.5, 1.4 * q_star), xlabel="$q_t$", ylabel="$q_{t+1}$",
       title="Phase diagram for the asset price")
ax.annotate("$q^*$", (q_star, -r_pf * q_star - 0.4), ha="center")
ax.legend(frameon=False, fontsize=8, loc="upper left")
show(fig, "m0_phase")

# %% [markdown]
# ## M1. The stochastic model and its competitive equilibrium
# $\max E_0\sum_t\beta^t c_t^{1-\sigma}/(1-\sigma)$ s.t. $c_t = A_t + d_{t+1} - (1+r)d_t$ (the asset is in fixed
# supply, so its trades net out) and $d_{t+1}\le\kappa q_t$. Equilibrium conditions:
# $$u'(c_t) = \beta(1+r)E_t u'(c_{t+1}) + \mu_t, \qquad
#   q_t\,u'(c_t) = \beta E_t\left[u'(c_{t+1})(\alpha A_{t+1}+q_{t+1}) + \kappa\mu_{t+1}q_{t+1}\right]$$
# The last term is the **collateral premium**: the asset is worth more because it relaxes tomorrow's constraint.
#
# Calibration (annual): $\sigma=2$, $\beta=0.91$, $r=4\%$, $\alpha=0.3$, $\kappa=0.14$; $\log A$ AR(1) with
# $\rho=0.85$, $\sigma_\varepsilon=5\%$ (Rouwenhorst, 9 states).
#
# **Time iteration:** guess $C, Q, M$ as functions of $(d, A)$. For each state, (i) solve the Euler equation
# for $d'$ by bisection; (ii) if that violates $d'\le\kappa q$, solve $d' = \kappa q(d')$ instead, where
# $q(d')$ comes from the asset-pricing equation; (iii) update $C, Q, M$; repeat.
# With a tax $\tau$ on debt (rebated lump sum), the Euler equation becomes
# $u'(c_t) = \beta(1+r)(1+\tau_t)E_t u'(c_{t+1}) + \mu_t$; the in-class exercise uses that.

# %%
sigma, beta, r, alpha, kappa = 2.0, 0.91, 0.04, 0.3, 0.14
R = 1 + r
rho, sd_e, NA, ND = 0.85, 0.05, 9, 400


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


logA, P = rouwenhorst(NA, rho, sd_e)
A = np.exp(logA)
dg = np.linspace(-0.3, 0.55, ND)                      # debt grid (d > 0 is debt)
DD, AA = np.meshgrid(dg, A, indexing="ij")           # state (d, A), shape (ND, NA)


def up(c):
    return c ** -sigma


def u(c):
    return np.log(c) if sigma == 1 else c ** (1 - sigma) / (1 - sigma)


def interp(x, F):                                     # F(d', A) on the grid, evaluated at x (ND, NA)
    return np.stack([np.interp(x[:, j], dg, F[:, j]) for j in range(NA)], 1)


def bisect(f, lo, hi, n=50):                          # f decreasing, f(lo) > 0 > f(hi), vectorised
    for _ in range(n):
        mid = (lo + hi) / 2
        pos = f(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return (lo + hi) / 2


def solve(tau=0.0, tol=1e-7, maxit=600, start=None):
    """tau: scalar or (ND, NA) array, the tax on debt in each state."""
    C, Q, M = np.maximum(AA - r * DD, 0.05), np.full((ND, NA), beta * alpha / (1 - beta)), np.zeros((ND, NA))
    if start is not None:                                        # warm start from another solution
        C, Q, M = start["C"], start["Q"], start["M"]
    D_old = None
    lo = np.maximum(dg[0], R * DD - AA + 1e-6)
    hi = np.full((ND, NA), dg[-1])
    for it in range(maxit):
        X = (up(C) * (alpha * A + Q) + kappa * M * Q) @ P.T     # E[...] in the asset-pricing equation, by (d', A)
        EU = up(C) @ P.T                                         # E u'(c'), by (d', A)

        def cons(d1):
            return np.maximum(AA + d1 - R * DD, 1e-10)

        d_u = bisect(lambda d1: up(cons(d1)) - beta * R * (1 + tau) * interp(d1, EU), lo, hi)
        slack = lambda d1: kappa * beta * interp(d1, X) / up(cons(d1)) - d1     # noqa: E731
        bind = slack(d_u) < 0
        # binding: the largest root of slack below d_u. q(d') need not be monotone, so first find the last grid
        # point below d_u where the constraint still holds, then bisect between it and d_u.
        cg = AA[:, None, :] + dg[None, :, None] - R * DD[:, None, :]             # (d, d' on grid, A)
        sg = kappa * beta * X[None] / up(np.maximum(cg, 1e-10)) - dg[None, :, None]
        okg = (cg > 0) & (sg >= 0) & (dg[None, :, None] <= d_u[:, None, :])
        lo_b = dg[(ND - 1) - np.argmax(okg[:, ::-1, :], axis=1)]
        D = np.where(bind, bisect(slack, np.minimum(lo_b, d_u), d_u), d_u)
        c = cons(D)
        Qn = beta * interp(D, X) / up(c)
        Mn = np.where(bind, np.maximum(up(c) - beta * R * interp(D, EU), 0), 0)
        gap = np.abs(c - C) + np.abs(Qn - Q)
        err = gap.max()
        if it == maxit - 1:                                       # stuck in a two-cycle: take the midpoint
            print(f"  tau={np.max(tau):.3f}: no convergence after {maxit} iterations (largest change {err:.1e}, "
                  f"99th pct {np.percentile(gap, 99):.1e}); averaging the last two iterates")
            return dict(C=(c + C) / 2, Q=(Qn + Q) / 2, M=(Mn + M) / 2, D=(D + D_old) / 2, it=it, tau=tau, err=err)
        C, Q, M, D_old = c, Qn, Mn, D
        if err < tol:
            break
    return dict(C=C, Q=Q, M=M, D=D, it=it, tau=tau, err=err)


t = time.time()
ce = solve()
print(f"Competitive equilibrium: {ce['it']} iterations, {time.time() - t:.1f} s")

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for j, lab, col in [(1, "low A", RED), (NA // 2, "mean A", GREY), (NA - 2, "high A", BLUE)]:
    ax[0].plot(dg, ce["D"][:, j], color=col, label=lab)
    ax[0].plot(dg, kappa * ce["Q"][:, j], color=col, ls=":", lw=1)
    ax[1].plot(dg, ce["Q"][:, j], color=col)
    ax[2].plot(dg, ce["C"][:, j], color=col)
ax[0].plot(dg, dg, color=GREY, lw=0.6)
ax[0].set(title=r"Debt policy $d'$ (dotted: limit $\kappa q$)", xlabel="debt today $d$")
ax[1].set(title="Asset price $q$", xlabel="debt today $d$")
ax[2].set(title="Consumption $c$", xlabel="debt today $d$")
for a in ax:
    a.set_xlim(0, 0.5)
ax[1].set_ylim(1.5, 4.5)
ax[0].legend(frameon=False, fontsize=8)
show(fig, "m1_policy")

# %% [markdown]
# ## M2. Accuracy: Euler equation errors
# On a fine grid, in states where the constraint is slack: $\log_{10}|1 - \tilde c/c|$, where $\tilde c$
# solves $u'(\tilde c) = \beta(1+r)E u'(c')$ using the solution itself.

# %%
def euler_errors(sol, nfine=2000):
    df = np.linspace(0.0, 0.45, nfine)
    errs = []
    for j in range(NA):
        d1 = np.interp(df, dg, sol["D"][:, j])
        c = A[j] + d1 - R * df
        Eu = sum(P[j, k] * up(A[k] + np.interp(d1, dg, sol["D"][:, k]) - R * d1) for k in range(NA))
        slack = kappa * np.interp(df, dg, sol["Q"][:, j]) - d1 > 1e-6
        errs.append(np.log10(np.abs(1 - (beta * R * Eu) ** (-1 / sigma) / c))[slack])
    return np.concatenate(errs)


ee = euler_errors(ce)
print(f"Euler errors (log10): median {np.median(ee):.1f}, 99th percentile {np.percentile(ee, 99):.1f}")

# %% [markdown]
# ## M3. Simulation and the definition of a sudden stop
# Simulate 100,000 years. A **sudden stop**: the constraint binds and the current account reversal is more than
# one standard deviation above its mean (Bianchi 2011; Mendoza 2010 uses a similar rule).

# %%
rng = np.random.default_rng(2026)
TS = 101_000
cumP = P.cumsum(1)
draws = rng.random(TS)
state = np.empty(TS, int)
state[0] = NA // 2
for t in range(1, TS):
    state[t] = min(np.searchsorted(cumP[state[t - 1]], draws[t]), NA - 1)


def simulate(sol, burn=1000):
    d = np.empty(TS + 1)
    d[0] = 0.2
    for t in range(TS):
        d[t + 1] = np.interp(d[t], dg, sol["D"][:, state[t]])
    j, dd = state, d[:-1]

    def get(F):                                       # F(d, A) at the simulated states, one A at a time
        out = np.empty(TS)
        for k in range(NA):
            m = j == k
            out[m] = np.interp(dd[m], dg, F[:, k])
        return out

    s = pd.DataFrame({"A": A[j], "d": dd, "d1": d[1:], "q": get(sol["Q"]), "mu": get(sol["M"])})
    s["c"] = s.A + s.d1 - R * s.d
    s["nx"] = 100 * (s.A - s.c) / s.A                     # trade balance / GDP, %
    s["ca"] = -100 * (s.d1 - s.d) / s.A                   # current account / GDP, %
    s["debt"] = 100 * s.d / s.A
    s["bind"] = s.mu > 1e-8
    s["ss"] = s.bind & (s.ca > s.ca.mean() + s.ca.std())
    return s.iloc[burn:].reset_index(drop=True)


def moments(s):
    return pd.Series({"debt / GDP, %": s.debt.mean(), "constraint binds, % years": 100 * s.bind.mean(),
                      "sudden stops, % years": 100 * s.ss.mean(),
                      "s.d. log c / s.d. log A": np.std(np.log(s.c)) / np.std(np.log(s.A)),
                      "mean asset price q": s.q.mean()})


sim = simulate(ce)
print(moments(sim).round(2).to_string())

# %% [markdown]
# ## M4. Anatomy of a sudden stop, and amplification
# Event window around sudden stops (averages across events), and Mendoza's (2010, Table 4) question: how much
# worse is a bad shock when the constraint binds? Compare years with a large fall in $A$ (more than one s.d.
# of the innovation) when the constraint binds (SS) and when it does not (NSS).

# %%
def window(s, var, W=3):
    idx = s.index[s.ss & (s.index > W) & (s.index < len(s) - W)]
    return np.array([s[var].values[i - W:i + W + 1] for i in idx]).mean(0)


def event_panel(sims, labels, colors, name, W=3):
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.0))
    panels = [("A", "output $A$", lambda x: 100 * (x / x.mean() - 1)), ("c", "consumption", None),
              ("q", "asset price $q$", None), ("nx", "trade balance / GDP", None), ("debt", "debt / GDP", None)]
    for s, lab, col in zip(sims, labels, colors):
        for a, (v, ttl, f) in zip(ax, panels):
            y = window(s, v, W)
            if v in ("c", "q"):
                y = 100 * (y / s[v].mean() - 1)
            elif f is not None:
                y = 100 * (y / s[v].mean() - 1)
            a.plot(range(-W, W + 1), y, "o-", ms=3, color=col, label=lab)
            a.set(title=ttl, xlabel="years")
    ax[0].set_ylabel("% from mean / % of GDP")
    ax[0].legend(frameon=False, fontsize=8)
    show(fig, name)


event_panel([sim], ["laissez-faire"], [RED], "m4_event")

big_fall = np.r_[False, np.diff(np.log(sim.A.values)) < -sd_e]
amp = pd.DataFrame({k: {"consumption, %": 100 * np.mean(np.diff(np.log(sim.c.values))[m[1:]]),
                         "asset price, %": 100 * np.mean(np.diff(np.log(sim.q.values))[m[1:]]),
                         "trade balance / GDP, pp": np.mean(np.diff(sim.nx.values)[m[1:]]),
                         "output A, %": 100 * np.mean(np.diff(np.log(sim.A.values))[m[1:]])}
                    for k, m in [("SS", big_fall & sim.bind.values), ("NSS", big_fall & ~sim.bind.values)]})
print("Change on impact after a large fall in A (Mendoza 2010, Table 4 style):")
print(amp.round(2).to_string())

# %%
print(f"Total time {time.time() - T0:.1f} s")
