# %% [markdown]
# # Week 2 lab: shift-share IV and event studies, the fast version
#
# **Question.** How much manufacturing employment did US local labour markets lose because of import
# competition from China, 1990-2007?
#
# **Data** (Autor, Dorn and Hanson 2013, as distributed by Goldsmith-Pinkham, Sorkin and Swift 2020
# and Adão, Kolesár and Morales 2019): 722 commuting zones (CZs) $\ell$ x 2 periods $t$ (1990-2000, 2000-2007),
# 397 manufacturing industries $n$ per period.
#
# - $y_{\ell t}$: change in the manufacturing employment share of the working-age population, pp per decade
# - $x_{\ell t}$: change in US imports from China per worker in the CZ, \$1000 per worker per decade
# - $s_{\ell n t}$: lagged (1980 / 1990) share of industry $n$ in CZ employment; they sum to $S_{\ell t} < 1$
# - $g_{n t}$: growth of imports from China in eight other high-income countries, \$1000 per US worker
# - $z_{\ell t} = \sum_n s_{\ell n t}\, g_{n t}$: the shift-share instrument
#
# Same cells, same specifications and same numbers as `week2_simple.py`, and the figures are the ones in the
# deck (saved to `figures/`):
#
# | Part | Cells |
# | --- | --- |
# | A. Shift-share IV | S0 data, S1 OLS, S2 the ADH instrument, S3 incomplete shares, S4 standard errors and the placebo, S5 the shock-level view, S6 Rotemberg weights, S7 balance |
# | B. Event studies | E0 simulated staggered panel, E1 static TWFE and its weights, E2 event study: TWFE, imputation, LP-DiD |
#
# **How the code is written.** Everything is numpy on two arrays: the CZ x shock share matrix `Sh`
# (1444 x 794) and the shock vector `g`. Then `z = Sh @ g`, the shock-level aggregation is one matrix product,
# and the placebo (S4) draws 1000 shock vectors at once.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run week2_fast.py` (a few seconds).


# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

try:                                       # VS Code cells / Jupyter: figures appear under each cell
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:                          # plain script: only save the figures
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# ---------------------------------------------------------------------------
# Toolkit: weighted 2SLS with robust / clustered SEs, and residualising
# ---------------------------------------------------------------------------
def tsls(y, x, z, W, w, cluster=None):
    """Weighted 2SLS of y on x (instrument z) with controls W (no constant added).
    z = x gives OLS. Returns coefficient on x, its SE (HC0 or clustered, no small-sample
    correction, as linearmodels' default) and the first-stage F (same SE type)."""
    r = np.sqrt(w)
    X, Z = np.column_stack([x, W]) * r[:, None], np.column_stack([z, W]) * r[:, None]
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    b = np.linalg.lstsq(Xh, y * r, rcond=None)[0]
    e = y * r - X @ b
    A = np.linalg.inv(Xh.T @ Xh)
    se = np.sqrt(sandwich(A, Xh * e[:, None], cluster)[0, 0])
    # first stage: x on z and W
    g = np.linalg.lstsq(Z, x * r, rcond=None)[0]
    Az = np.linalg.inv(Z.T @ Z)
    F = g[0] ** 2 / sandwich(Az, Z * (x * r - Z @ g)[:, None], cluster)[0, 0]
    return b[0], se, F


def sandwich(A, S, cluster):
    if cluster is not None:
        S = pd.DataFrame(S).groupby(cluster).sum().to_numpy()
    return A @ (S.T @ S) @ A


def resid(v, W, w):
    """Residual of v (vector or matrix) from a WLS regression on W, weights w."""
    r = np.sqrt(w)[:, None]
    V = v.reshape(len(w), -1)
    B = np.linalg.lstsq(W * r, V * r, rcond=None)[0]
    out = V - W @ B
    return out[:, 0] if v.ndim == 1 else out


# %% [markdown]
# ## S0. The data (slide "Running example")
# One row per CZ and period in `adh`; the shares are reshaped into the matrix `Sh`, rows = (CZ, period),
# columns = (industry, period). A share is zero outside its own period, so `Sh @ g` uses 1980 shares with the
# 1990s shocks and 1990 shares with the 2000s shocks.

# %%
adh = pd.read_csv(DATA / "ADHdata_AKM.csv")
adh["t2"] = adh.t2.astype(int)
adh["year"] = np.where(adh.t2 == 1, 2000, 1990)          # year the shares refer to (the start of the period)
shares = pd.read_stata(DATA / "Lshares.dta")
shocks = pd.read_stata(DATA / "shocks.dta")
names = pd.read_stata(DATA / "sic_code_desc.dta").set_index("sic").description

shocks = shocks.sort_values(["year", "sic87dd"]).reset_index(drop=True)
Sh = (shares.pivot_table(index=["czone", "year"], columns=["year", "sic87dd"], values="ind_share",
                         fill_value=0.0, aggfunc="sum")
      .reindex(index=pd.MultiIndex.from_frame(adh[["czone", "year"]]),
               columns=pd.MultiIndex.from_frame(shocks[["year", "sic87dd"]]), fill_value=0.0)
      .to_numpy())
g = shocks.g.to_numpy()                                   # g_nt, 794 = 397 industries x 2 periods
per = (shocks.year == 2000).astype(int).to_numpy()        # period of each shock
sic3 = (shocks.sic87dd // 10).astype(int).to_numpy()      # industry clusters for exposure-robust SEs

y, x, w = adh.d_sh_empl_mfg.to_numpy(), adh.shock.to_numpy(), adh.weights.to_numpy()
t2, state = adh.t2.to_numpy(), adh.statefip.to_numpy()
z = Sh @ g
S = Sh.sum(1)                                             # sum of shares = lagged manufacturing share
one = np.ones(len(y))
print(f"{adh.czone.nunique()} CZs x 2 periods, {len(g)} shocks; "
      f"max |z - ADH's IV| = {np.abs(z - adh.IV).max():.1e}")
print(f"sum of shares S: mean {S.mean():.3f}, range {S.min():.3f} to {S.max():.3f}")

# controls, as matrices
Wt = np.column_stack([one, t2])                                       # period effect
WS = np.column_stack([one, t2, S * (1 - t2), S * t2])                 # + S x period (BHJ)
adh_vars = ["l_shind_manuf_cbp", "l_sh_popedu_c", "l_sh_popfborn", "l_sh_empl_f",
            "l_sh_routine33", "l_task_outsource"]
div = pd.get_dummies(adh.division, prefix="div", drop_first=True, dtype=float).to_numpy()
Wadh = np.column_stack([Wt, adh[adh_vars].to_numpy(), div])           # ADH's controls
WadhS = np.column_stack([Wadh, S * (1 - t2), S * t2])                 # ADH + S x period

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
gu = shocks.g_importsUSA.to_numpy()
for p, c, lab in [(0, BLUE, "1990-2000"), (1, RED, "2000-2007")]:
    m = per == p
    ax[0].scatter(np.arcsinh(g[m]), np.arcsinh(gu[m]), s=9, color=c, alpha=0.6, label=lab)
tk = np.array([-100, -10, -1, 0, 1, 10, 100, 500])
for a_ in (ax[0].set_xticks, ax[0].set_yticks):
    a_(np.arcsinh(tk), tk)
ax[0].set_xlabel("imports from China, 8 other countries ($g_n$)")
ax[0].set_ylabel("imports from China, US")
ax[0].set_title("397 industries: Chinese import growth (\\$1000 per US worker)", fontsize=11)
ax[0].legend(frameon=False, fontsize=9)
# right: binned scatter of y on x within each period (20 population-weighted bins of exposure)
for p, c, lab in [(0, BLUE, "1990-2000"), (1, RED, "2000-2007")]:
    m = t2 == p
    q = np.quantile(x[m], np.linspace(0, 1, 21))
    bb = np.clip(np.searchsorted(q, x[m], side="right") - 1, 0, 19)
    mx = [np.average(x[m][bb == k], weights=w[m][bb == k]) for k in range(20)]
    my = [np.average(y[m][bb == k], weights=w[m][bb == k]) for k in range(20)]
    sl = np.polyfit(x[m], y[m], 1, w=np.sqrt(w[m]))
    ax[1].scatter(mx, my, color=c, s=22, label=lab, zorder=3)
    xx = np.array([min(mx), max(mx)])
    ax[1].plot(xx, np.polyval(sl, xx), color=c, lw=1.2)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xlabel("$x$: exposure, \\$1000 per worker")
ax[1].set_ylabel("$y$: $\\Delta$ manuf. emp. / pop., pp")
ax[1].set_title("722 commuting zones: exposure and job loss (binned)", fontsize=11)
ax[1].legend(frameon=False, fontsize=9, loc="lower left")
show(fig, "data")
for p in (0, 1):
    m = per == p
    print(f"S0 industries {1990 + 10 * p}s: corr(US, other countries) = {np.corrcoef(g[m], gu[m])[0, 1]:.2f}, "
          f"in asinh {np.corrcoef(np.arcsinh(g[m]), np.arcsinh(gu[m]))[0, 1]:.2f}")


# %% [markdown]
# # Part A. Shift-share IV
# $$y_{\ell t} = \beta\, x_{\ell t} + \gamma' w_{\ell t} + \varepsilon_{\ell t}, \qquad
#   x_{\ell t} \text{ instrumented by } z_{\ell t} = \sum_n s_{\ell n t}\, g_{n t}$$
# All regressions are weighted by CZ population and cluster by state, as in ADH, unless stated otherwise.

# %%
def binscatter(ax, v, u, W, w, nb=20, color=BLUE, label=None):
    """Binned scatter of u on v after residualising both on W (means added back)."""
    vr = resid(v, W, w) + np.average(v, weights=w)
    ur = resid(u, W, w) + np.average(u, weights=w)
    q = np.quantile(vr, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, vr, side="right") - 1, 0, nb - 1)
    mv = np.array([np.average(vr[b == k], weights=w[b == k]) for k in range(nb)])
    mu = np.array([np.average(ur[b == k], weights=w[b == k]) for k in range(nb)])
    ax.scatter(mv, mu, color=color, s=22, label=label, zorder=3)
    sl = np.sum(w * (vr - vr.mean()) * ur) / np.sum(w * (vr - vr.mean()) ** 2)
    xx = np.array([mv.min(), mv.max()])
    ax.plot(xx, mu.mean() + sl * (xx - mv.mean()), color=color, lw=1.2)
    return sl


# %% [markdown]
# ## S1. OLS
# $$y_{\ell t} = \beta\, x_{\ell t} + \delta_t + \varepsilon_{\ell t}$$

# %%
b, se, _ = tsls(y, x, x, Wt, w, state)
print(f"S1 OLS: {b:.3f} ({se:.3f})")
b_ctrl, se_ctrl, _ = tsls(y, x, x, Wadh, w, state)
print(f"S1 OLS with ADH controls: {b_ctrl:.3f} ({se_ctrl:.3f})")
fig, ax = plt.subplots(figsize=(4.8, 3.3))
binscatter(ax, x, y, Wt, w, color=GREY)
ax.set_xlabel("$x$: Chinese import exposure per worker\n(\\$1000 per decade, net of period means; 20 bins)")
ax.set_ylabel("$y$: change in manuf. employment /\nworking-age population (pp/decade)")
ax.set_title(f"OLS: slope {b:.2f}", fontsize=11)
show(fig, "s1_ols")


# %% [markdown]
# ## S2. The ADH instrument
# First stage $x$ on $z$, reduced form $y$ on $z$, IV = reduced form / first stage.

# %%
res = {}
res["t2"] = tsls(y, x, z, Wt, w, state)
b_fs, se_fs, _ = tsls(x, z, z, Wt, w, state)
b_rf, se_rf, _ = tsls(y, z, z, Wt, w, state)
print(f"S2 first stage {b_fs:.3f} ({se_fs:.3f}), reduced form {b_rf:.3f} ({se_rf:.3f}), "
      f"ratio {b_rf / b_fs:.3f}")
print(f"S2 IV: {res['t2'][0]:.3f} ({res['t2'][1]:.3f}), F = {res['t2'][2]:.0f}")
res["adh"] = tsls(y, x, z, Wadh, w, state)
print(f"S2 IV with ADH controls: {res['adh'][0]:.3f} ({res['adh'][1]:.3f}), F = {res['adh'][2]:.0f}")
print(f"   unweighted: {tsls(y, x, z, Wadh, one, state)[0]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
binscatter(ax[0], z, x, Wt, w, color=BLUE)
ax[0].set_title(f"First stage: slope {b_fs:.2f}", fontsize=11)
ax[0].set_xlabel("$z$: instrument, exposure via imports into\n8 other countries (\\$1000 per worker; 20 bins)"); ax[0].set_ylabel("$x$: Chinese import exposure per worker\n(\\$1000 per decade)")
binscatter(ax[1], z, y, Wt, w, color=RED)
ax[1].set_title(f"Reduced form: slope {b_rf:.2f}", fontsize=11)
ax[1].set_xlabel("$z$: instrument, exposure via imports into\n8 other countries (\\$1000 per worker; 20 bins)"); ax[1].set_ylabel("$y$: change in manuf. employment /\nworking-age population (pp/decade)")
show(fig, "s2_ssiv")


# %% [markdown]
# ## S3. Incomplete shares
# The shares only cover manufacturing, so $S_{\ell t} = \sum_n s_{\ell n t} < 1$. If shocks were drawn at random
# with mean $\mu_t$, then $E[z_{\ell t}] = \mu_t S_{\ell t}$: high-manufacturing CZs get a high instrument even with
# random shocks. The fix: control for $S_{\ell t}$ by period.

# %%
res["S"] = tsls(y, x, z, WS, w, state)
res["adhS"] = tsls(y, x, z, WadhS, w, state)
mu = np.array([g[per == 0].mean(), g[per == 1].mean()])  # mean shock in each period
print(f"S3 mu_t = {mu.round(2)}")
for k, lab in [("t2", "period only"), ("S", "+ S x period"), ("adh", "ADH controls"),
               ("adhS", "ADH controls + S x period")]:
    print(f"S3 {lab:28s} {res[k][0]:7.3f} ({res[k][1]:.3f})  F = {res[k][2]:.0f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
for p, c, lab in [(0, BLUE, "1990-2000"), (1, RED, "2000-2007")]:
    m = t2 == p
    ax[0].scatter(S[m], z[m], s=5, color=c, alpha=0.4)
    ss = np.linspace(0, S.max(), 2)
    ax[0].plot(ss, mu[p] * ss, color=c, lw=1.5, label=f"{lab}: $E[z] = {mu[p]:.1f}\\,S$")
ax[0].set_ylim(0, 25)
ax[0].set_xlabel("$S$: sum of shares (lagged manufacturing share)")
ax[0].set_ylabel("$z$")
ax[0].set_title("Random shocks still predict $z$ through $S$", fontsize=11)
ax[0].legend(frameon=False, fontsize=9, loc="upper left")
order = [("t2", "period only"), ("S", "+ $S$ x period"), ("adh", "ADH controls"), ("adhS", "ADH + $S$ x period")]
for j, (k, lab) in enumerate(order):
    c = GREY if k in ("t2", "adh") else BLUE
    ax[1].errorbar(res[k][0], -j, xerr=1.96 * res[k][1], fmt="o", color=c, capsize=3)
    ax[1].text(res[k][0] + 1.96 * res[k][1] + 0.04, -j, f"{res[k][0]:.2f}", va="center", fontsize=9, color=c)
ax[1].set_yticks(-np.arange(len(order)), [lab for _, lab in order])
ax[1].axvline(0, color="grey", lw=0.6)
ax[1].set_xlim(-1.0, 0.2)
ax[1].set_xlabel("$\\hat\\beta$ with 95% CI (state clusters)")
ax[1].set_title("Effect of \\$1000 exposure on manuf. emp. share", fontsize=11)
show(fig, "s3_shares")


# %% [markdown]
# ## S4. Standard errors
# CZs with similar shares get similar $z$ and similar unobserved industry shocks: their errors are correlated
# however far apart they are. Exposure-robust SEs (Adão, Kolesár and Morales 2019) come from the equivalent
# shock-level IV (Borusyak, Hull and Jaravel 2022): residualise $y$ and $x$ on the controls, average them across
# CZs with weights $w_\ell s_{\ell n}$, and run an IV across shocks, weighted by $s_n = \sum_\ell w_\ell s_{\ell n}$,
# clustered by SIC3 industry group.

# %%
def shock_level(u, W, ww=w):
    """Shock-level aggregate of u residualised on W: ubar_n = sum_l ww_l s_ln u_l / s_n."""
    ur = resid(u, W, ww)
    sn = Sh.T @ ww
    return (Sh.T @ (ww[:, None] * ur.reshape(len(ww), -1))) / sn[:, None], sn


def ss_iv(u, v, W, gg=g):
    """Shock-level IV of ubar on vbar, instrument g, period effects, weights s_n, SIC3 clusters.
    u: CZ outcome, v: CZ treatment (v = z for a reduced form). Returns coef, SE, F."""
    ub, sn = shock_level(u, W)
    vb, _ = shock_level(v, W)
    Wp = np.column_stack([1 - per, per])
    return tsls(ub[:, 0], vb[:, 0], gg, Wp, sn, sic3)


for k, W_ in [("S", WS), ("adhS", WadhS)]:
    b_, se_r, _ = tsls(y, x, z, W_, w)
    _, se_c, _ = tsls(y, x, z, W_, w, state)
    bs, se_e, F_e = ss_iv(y, x, W_)
    print(f"S4 {k:5s} beta {b_:.3f}: robust {se_r:.3f}, state {se_c:.3f}, exposure-robust {se_e:.3f} "
          f"(shock-level beta {bs:.3f}, F {F_e:.0f})")
ses = {"robust": tsls(y, x, z, WadhS, w)[1], "state clusters": tsls(y, x, z, WadhS, w, state)[1],
       "exposure-robust": ss_iv(y, x, WadhS)[1]}
b_pref = res["adhS"][0]

# Placebo (Adão, Kolesár and Morales 2019): draw random shocks g* ~ N(0, 1), build z* = Sh g*, regress the
# actual y on z* with the preferred controls. The true effect is zero, so a 5% test should reject 5% of the time.
R = 1000
Gs = rng.standard_normal((len(g), R))
Zs = Sh @ Gs
yr = resid(y, WadhS, w)
Zr = resid(Zs, WadhS, w)
den = (w[:, None] * Zr ** 2).sum(0)
bet = (w[:, None] * Zr * yr[:, None]).sum(0) / den
E = yr[:, None] - Zr * bet
se_rob = np.sqrt(((w[:, None] * Zr * E) ** 2).sum(0)) / den
se_st = np.sqrt((pd.DataFrame(w[:, None] * Zr * E).groupby(state).sum() ** 2).sum(axis=0).to_numpy()) / den
# exposure-robust: the same regression at the shock level (shocks g*, clusters SIC3)
sn = Sh.T @ w
yb = Sh.T @ (w * yr) / sn
zb = Sh.T @ (w[:, None] * Zr) / sn[:, None]                    # z-bar for each draw
Gt = Gs - np.array([np.average(Gs[per == p], weights=sn[per == p], axis=0) for p in (0, 1)])[per]
num = (sn[:, None] * Gt * yb[:, None]).sum(0)
dd = (sn[:, None] * Gt * zb).sum(0)                           # = den (same coefficient)
rb = yb[:, None] - zb * bet                                    # shock-level residuals (period effects are
rb = rb - np.array([np.average(rb[per == p], weights=sn[per == p], axis=0) for p in (0, 1)])[per]  # absorbed)
se_exp = np.sqrt((pd.DataFrame(sn[:, None] * Gt * rb).groupby(sic3).sum() ** 2).sum(axis=0).to_numpy()) / np.abs(dd)
print(f"S4 placebo check: shock-level coefficient = CZ coefficient, max diff {np.abs(num / dd - bet).max():.1e}")
rej = {"robust": np.mean(np.abs(bet / se_rob) > 1.96), "state clusters": np.mean(np.abs(bet / se_st) > 1.96),
       "exposure-robust": np.mean(np.abs(bet / se_exp) > 1.96)}
print("S4 placebo rejection rates at 5%:", {k: round(v, 3) for k, v in rej.items()})

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
cols = [GREY, BLUE, GREEN]
for j, (k, s_) in enumerate(ses.items()):
    ax[0].errorbar(b_pref, -j, xerr=1.96 * s_, fmt="o", color=cols[j], capsize=4)
    ax[0].text(b_pref + 1.96 * s_ + 0.02, -j, f"SE {s_:.3f}", va="center", fontsize=9, color=cols[j])
ax[0].set_yticks(-np.arange(3), list(ses))
ax[0].axvline(0, color="grey", lw=0.6)
ax[0].set_xlim(-0.65, 0.25)
ax[0].set_title(f"ADH + $S$ x period: $\\hat\\beta$ = {b_pref:.2f}, 95% CI", fontsize=11)
ax[1].barh(-np.arange(3), [rej[k] for k in ses], color=cols)
for j, k in enumerate(ses):
    ax[1].text(rej[k] + 0.01, -j, f"{100 * rej[k]:.0f}%", va="center", fontsize=9)
ax[1].axvline(0.05, color=RED, ls="--", lw=1)
ax[1].set_yticks(-np.arange(3), list(ses))
ax[1].set_xlim(0, max(rej.values()) + 0.12)
ax[1].set_title("Placebo: random shocks, rejection rate of a 5% test", fontsize=11)
show(fig, "s4_se")


# %% [markdown]
# ## S5. The shock-level view
# The CZ-level IV is a shock-level IV in disguise: 794 observations (industry x period), weighted by $s_n$.
# How many shocks really identify $\beta$? The effective number is $1 / \sum_n \hat s_n^2$ with
# $\hat s_n = s_n / \sum_m s_m$.

# %%
ybar, sn = shock_level(y, WadhS)
xbar, _ = shock_level(x, WadhS)
sh_ = sn / sn.sum()
eff_n = 1 / np.sum(sh_ ** 2)
c3 = pd.Series(sh_).groupby(sic3).sum()
eff_c = 1 / np.sum(c3 ** 2)
top = np.argsort(-sh_ * np.abs(g - mu[per]))[:5]
print(f"S5 effective number of shocks {eff_n:.0f} (of {len(g)}); of SIC3 clusters {eff_c:.0f} (of {len(c3)})")
print("S5 largest shocks:", np.sort(g)[-3:].round(0), " largest weight", sh_.max().round(3))
print("S5 top industries by weight x |shock|:")
for k in top:
    print(f"    {shocks.year[k]:.0f} {names.get(int(shocks.sic87dd[k]), '?')[:40]:40s} "
          f"g = {g[k]:6.1f}, weight {sh_[k]:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for p, c, lab in [(0, BLUE, "1990-2000"), (1, RED, "2000-2007")]:
    m = per == p
    ax[0].scatter(np.arcsinh(g[m]), ybar[m, 0], s=3000 * sh_[m], color=c, alpha=0.45, label=lab,
                  edgecolor="none")
ax[0].set_xticks(np.arcsinh(tk), tk)
ax[0].set_ylim(-2, 2)
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlabel("$g_n$: shock (\\$1000 per US worker)")
ax[0].set_ylabel("$\\bar y_n$: residual outcome")
ax[0].set_title("794 shocks (size = weight $s_n$)", fontsize=11)
ax[0].legend(frameon=False, fontsize=9)
cs = np.cumsum(np.sort(sh_)[::-1])
ax[1].plot(np.arange(1, len(cs) + 1), cs, color=BLUE)
ax[1].axvline(eff_n, color=RED, ls="--", lw=1)
ax[1].text(eff_n + 15, 0.3, f"effective number\n= {eff_n:.0f}", color=RED, fontsize=9)
ax[1].set_xlabel("shocks, ranked by weight")
ax[1].set_ylabel("cumulative weight")
ax[1].set_title("How concentrated are the weights?", fontsize=11)
show(fig, "s5_shocks")


# %% [markdown]
# ## S6. Rotemberg weights (the exogenous-shares view)
# Goldsmith-Pinkham, Sorkin and Swift (2020): $\hat\beta = \sum_k \hat\alpha_k \hat\beta_k$, where $\hat\beta_k$ is the
# just-identified IV using share $k$ alone as the instrument and
# $\hat\alpha_k = \tilde g_k\, s_k' M x / \sum_j \tilde g_j\, s_j' M x$ (with $M$ residualising on the controls, $\tilde g$
# the shocks demeaned within period: with $S$ x period controlled, the weights are only unique for demeaned shocks).

# %%
Shr = resid(Sh, WadhS, w)                                  # M s_k for all k at once
xr, yr = resid(x, WadhS, w), resid(y, WadhS, w)
sx = Shr.T @ (w * xr)                                     # s_k' W M x
sy = Shr.T @ (w * yr)
gt = g - mu[per]
alpha = gt * sx / np.sum(gt * sx)
beta_k = np.divide(sy, sx, out=np.full_like(sy, np.nan), where=np.abs(sx) > 1e-12)
print(f"S6 sum alpha_k beta_k = {np.nansum(alpha * beta_k):.3f} (beta-hat {b_pref:.3f}); "
      f"negative weights sum to {alpha[alpha < 0].sum():.3f}")
topk = np.argsort(-np.abs(alpha))[:10]
print(f"S6 top 10 shares carry {alpha[topk].sum():.2f} of the weight; top 1: {alpha[topk[0]]:.2f}")
def short(k, n=34):
    nm = names.get(int(shocks.sic87dd[k]), "?")
    nm = nm if len(nm) <= n else nm[:n].rsplit(" ", 1)[0].rstrip(",") + "..."
    return f"{nm} ({'1990s' if per[k] == 0 else '2000s'})"


lab6 = [short(k) for k in topk]
for k, l in zip(topk, lab6):
    print(f"    {l:36s} alpha {alpha[k]:6.3f}  beta_k {beta_k[k]:7.2f}  g {g[k]:6.1f}")
fig, ax = plt.subplots(figsize=(9, 3.9))
ax.barh(-np.arange(10), alpha[topk], color=[BLUE if a_ > 0 else RED for a_ in alpha[topk]])
for j, k in enumerate(topk):
    ax.text(alpha[k] + 0.004, -j, f"$\\hat\\beta_k$ = {beta_k[k]:.2f}", va="center", fontsize=9)
ax.set_yticks(-np.arange(10), lab6, fontsize=9)
ax.set_xlim(min(0, alpha[topk].min() - 0.01), alpha[topk].max() + 0.07)
ax.set_xlabel("Rotemberg weight $\\hat\\alpha_k$")
ax.set_title(f"Ten largest of 794 weights: together {alpha[topk].sum():.2f}", fontsize=11)
show(fig, "s6_rotemberg")


# %% [markdown]
# ## S7. Balance
# Predetermined CZ characteristics should not be predicted by the instrument, given the controls the design
# needs (period and $S$ x period). Regress each one on $z$, standardised, and compare state-clustered with
# exposure-robust SEs.

# %%
bal_vars = {"l_sh_popedu_c": "college share", "l_sh_popfborn": "foreign-born share",
            "l_sh_empl_f": "female emp. share", "l_sh_routine33": "routine occ. share",
            "l_task_outsource": "offshorability", "l_shind_manuf_cbp": "manuf. share (CBP)"}
zs = z / np.sqrt(np.cov(resid(z, WS, w), aweights=w))      # z in residual s.d. units
bal = []
for v, lab in bal_vars.items():
    c = adh[v].to_numpy()
    c = c / np.sqrt(np.cov(c, aweights=w))
    b_, se_s, _ = tsls(c, zs, zs, WS, w, state)
    _, se_e, _ = ss_iv(c, zs, WS)
    bal.append((lab, b_, se_s, se_e))
    print(f"S7 {lab:20s} {b_:6.3f}  state {se_s:.3f}  exposure-robust {se_e:.3f}")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for j, (lab, b_, s1, s2) in enumerate(bal):
    ax.errorbar(b_, -j + 0.12, xerr=1.96 * s1, fmt="o", color=BLUE, capsize=3,
                label="state clusters" if j == 0 else None)
    ax.errorbar(b_, -j - 0.12, xerr=1.96 * s2, fmt="o", color=GREEN, capsize=3,
                label="exposure-robust" if j == 0 else None)
ax.set_yticks(-np.arange(len(bal)), [b_[0] for b_ in bal])
ax.axvline(0, color="grey", lw=0.6)
ax.set_xlabel("s.d. change per 1 s.d. of $z$ (controls: period, $S$ x period), 95% CI")
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Is the instrument balanced?", fontsize=11)
show(fig, "s7_balance")


# %% [markdown]
# # Part B. Event studies with staggered adoption
# A simulated panel where we know the truth: $N = 600$ units, $T = 20$ years, three cohorts adopt in years 6,
# 10 and 14, a quarter never adopts. The effect grows with time since adoption and is larger for early
# adopters:
# $$Y_{it} = \alpha_i + \lambda_t + \tau_{it} D_{it} + \varepsilon_{it}, \qquad
#   \tau_{it} = c_{E_i}\,(1 + 0.3\,(t - E_i)).$$

# %%
N, T = 600, 20
cohorts = np.array([6, 10, 14, 99])                   # 99 = never
scale = {6: 1.5, 10: 1.0, 14: 0.5, 99: 0.0}
Ei = cohorts[np.arange(N) % 4]
tt = np.arange(1, T + 1)
D = (tt[None, :] >= Ei[:, None]).astype(float)
K = tt[None, :] - Ei[:, None]                          # years since adoption
tau = np.vectorize(scale.get)(Ei)[:, None] * (1 + 0.3 * K) * D
rng = np.random.default_rng(2)                       # own seed: same draws as week2_simple.py
alpha_i = rng.normal(0, 1, N)
lam = 0.1 * tt + np.sin(tt / 3)
Y = alpha_i[:, None] + lam[None, :] + tau + rng.normal(0, 0.5, (N, T))
att = tau[D == 1].mean()

fig, ax = plt.subplots(figsize=(7, 3.4))
for c, col, lab in [(6, BLUE, "adopt in 6"), (10, GREEN, "adopt in 10"), (14, ORANGE, "adopt in 14"),
                    (99, GREY, "never")]:
    ax.plot(tt, Y[Ei == c].mean(0), color=col, label=lab)
    if c < 99:
        ax.axvline(c, color=col, ls=":", lw=1)
ax.set_xlabel("year"); ax.set_ylabel("mean outcome")
ax.legend(frameon=False, fontsize=9, ncol=2)
ax.set_title("Simulated staggered adoption", fontsize=11)
show(fig, "e0_sim")


# %% [markdown]
# ## E1. Static TWFE and its implicit weights
# $$Y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$
# By FWL, $\hat\beta = \sum_{it} \omega_{it} Y_{it}$ with $\omega_{it} = \tilde D_{it} / \sum \tilde D_{it}^2$, $\tilde D$ the
# two-way demeaned treatment. Plugging in the true model: $\hat\beta \approx \sum_{D_{it}=1} \omega_{it} \tau_{it}$.
# Long-run effects of early adopters get negative weight: they serve as controls for late adopters.

# %%
def twoway_demean(A):
    """Residual from regressing A (N x T, balanced) on unit and time dummies."""
    return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()


Dt = twoway_demean(D)
omega = Dt / np.sum(Dt * D)                            # weights on treated cells, sum to 1 over D = 1
beta_twfe = np.sum(Dt * Y) / np.sum(Dt * D)
print(f"E1 static TWFE {beta_twfe:.3f} vs true ATT {att:.3f}; "
      f"weights on tau: sum {omega[D == 1].sum():.2f}, negative {omega[(D == 1) & (omega < 0)].sum():.3f}")
print(f"   share of treated cells with negative weight: {np.mean(omega[D == 1] < 0):.2f}")
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5))
for c, col in [(6, BLUE), (10, GREEN), (14, ORANGE)]:
    m = Ei == c
    kk = np.arange(0, T - c + 1)
    ax[0].plot(kk, omega[m][:, c - 1:].sum(0) * 1, "o-", ms=3, color=col, label=f"adopt in {c}")
ax[0].axhline(0, color="grey", lw=0.6)
ax[0].set_xlabel("years since adoption"); ax[0].set_ylabel("TWFE weight (cohort total)")
ax[0].set_title("Implicit weights on $\\tau_{it}$", fontsize=11)
ax[0].legend(frameon=False, fontsize=9)
ax[1].bar([0, 1], [att, beta_twfe], color=[GREEN, GREY], width=0.5)
for j, v in enumerate([att, beta_twfe]):
    ax[1].text(j, v + 0.05, f"{v:.2f}", ha="center")
ax[1].set_xticks([0, 1], ["true ATT", "static TWFE"])
ax[1].set_title("Average effect on the treated", fontsize=11)
show(fig, "e1_twfe")


# %% [markdown]
# ## E2. Event studies: TWFE, imputation and LP-DiD
# - **TWFE event study**, the common way: leads $-5..-2$ and lags $0..8$, both ends binned, $k=-1$ omitted.
# - **Imputation** (Borusyak, Jaravel and Spiess 2024): fit $\alpha_i + \lambda_t$ on untreated cells only,
#   impute $\hat Y_{it}(0)$, average $Y_{it} - \hat Y_{it}(0)$ by years since adoption.
# - **LP-DiD** (Dube, Girardi, Jordà and Taylor 2025): for each $h$, a local projection
#   $Y_{i,t+h} - Y_{i,t-1} = \beta_h \Delta D_{it} + \delta^h_t + e_{it}$ on newly treated units and clean controls
#   (not yet treated at $t+h$). Week 1's LP, run as a difference-in-differences.

# %%
kk = np.arange(-4, 5)                                  # a typical short event window
truth = np.array([tau[K == k].mean() if k >= 0 else 0.0 for k in kk])

# TWFE event study with binned ends, as is common practice: dummies for k <= -4, -3, -2, 0..3, >= 4 (relative to k = -1)
Kc = np.where(Ei[:, None] < 99, np.clip(K, -4, 4), -1)  # never-treated sit in the reference group
cols_k = [k for k in kk if k != -1]
Xd = np.stack([(Kc == k).astype(float) for k in cols_k], axis=-1)       # N x T x 8
Xdt = np.stack([twoway_demean(Xd[..., j]) for j in range(Xd.shape[-1])], axis=-1).reshape(N * T, -1)
b_es = np.linalg.lstsq(Xdt, twoway_demean(Y).ravel(), rcond=None)[0]
es_twfe = np.insert(b_es, cols_k.index(0), 0.0)                         # put back k = -1 as 0

# Imputation: unit and time effects from untreated cells only (alternating projections)
U = D == 0
a_, l_ = np.zeros(N), np.zeros(T)
for _ in range(200):
    a_ = np.nanmean(np.where(U, Y - l_[None, :], np.nan), 1)
    l_ = np.nanmean(np.where(U, Y - a_[:, None], np.nan), 0)
Y0 = a_[:, None] + l_[None, :]
es_imp = np.array([(Y - Y0)[(K == k) & (D == 1)].mean() if k >= 0 else np.nan for k in kk])
# pre-trend test the BJS way: only untreated cells; add lead dummies to the untreated-only regression
# (here we just report the average untreated residual by k < 0, which should be ~0)
pre_imp = np.array([(Y - Y0)[(K == k) & (Ei[:, None] < 99)].mean() if k < 0 else np.nan for k in kk])
es_imp = np.where(kk < 0, pre_imp, es_imp)


def lpdid(h):
    """LP-DiD at horizon h (h < 0: placebo using Y_{t+h} - Y_{t-1})."""
    rows = []
    for t in range(1, T):                              # t index (0-based), need t-1 >= 0
        if t + h >= T or t + h < 0:
            continue
        newly = (D[:, t] == 1) & (D[:, t - 1] == 0)
        clean = D[:, max(t, t + h)] == 0
        keep = newly | clean
        rows.append(pd.DataFrame({"dy": Y[keep, t + h] - Y[keep, t - 1], "d": newly[keep].astype(float),
                                  "t": t}))
    d = pd.concat(rows)
    d["dy"] -= d.groupby("t").dy.transform("mean")      # time effects by FWL
    d["d"] -= d.groupby("t").d.transform("mean")
    return np.sum(d.d * d.dy) / np.sum(d.d ** 2)


es_lp = np.array([lpdid(k) if k != -1 else 0.0 for k in kk])
print("E2 years since adoption:", kk)
print("   truth      ", truth.round(2))
print("   TWFE ES    ", es_twfe.round(2))
print("   imputation ", es_imp.round(2))
print("   LP-DiD     ", es_lp.round(2))

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(kk, truth, color="black", lw=2.5, alpha=0.25, label="truth")
ax.plot(kk, es_twfe, "o-", color=GREY, ms=4, label="TWFE event study (binned)")
ax.plot(kk, es_imp, "s-", color=BLUE, ms=4, label="imputation (BJS)")
ax.plot(kk, es_lp, "^--", color=RED, ms=4, label="LP-DiD")
ax.axhline(0, color="grey", lw=0.6); ax.axvline(-0.5, color="grey", lw=0.6, ls=":")
ax.set_xlabel("years since adoption"); ax.set_ylabel("effect")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Same data, three event-study estimators", fontsize=11)
show(fig, "e2_es")

# %%
