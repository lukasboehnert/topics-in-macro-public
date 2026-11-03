# %% [markdown]
# # Week 2 lab: shift-share IV and event studies, step by step
#
# **Question.** How much manufacturing employment did US local labour markets lose because of import
# competition from China, 1990-2007?
#
# **Data** (Autor, Dorn and Hanson 2013, ADH; files from the Goldsmith-Pinkham, Sorkin and Swift 2020
# replication package): 722 commuting zones (CZs) $\ell$ x 2 periods $t$ (1990-2000, 2000-2007),
# 397 manufacturing industries $n$.
#
# - $y_{\ell t}$ `d_sh_empl_mfg`: change in manufacturing employment / working-age population, pp per decade
# - $x_{\ell t}$ `shock`: change in US imports from China per worker in the CZ, \$1000 per worker per decade
# - $s_{\ell n t}$ `ind_share`: share of industry $n$ in CZ employment at the start of the period (1980 shares for
#   the 1990s, 1990 shares for the 2000s, as ADH); summed over manufacturing they give $S_{\ell t} < 1$
# - $g_{nt}$ `g`: growth of imports from China in eight other high-income countries, \$1000 per US worker
# - $z_{\ell t} = \sum_n s_{\ell n t}\, g_{nt}$: the shift-share ("Bartik") instrument
#
# The cells follow the slides, and each one draws the figure on its slide:
#
# | Part | Cells |
# | --- | --- |
# | A. Shift-share IV | S0 data, S1 OLS, S2 the ADH instrument, S3 incomplete shares, S4 standard errors and the placebo, S5 the shock-level view, S6 Rotemberg weights, S7 balance |
# | B. Event studies | E0 simulated staggered panel, E1 static TWFE and its weights, E2 event study: TWFE, imputation, LP-DiD |
#
# **How the code is written.** Every regression is a formula on a DataFrame, estimated with linearmodels
# (`IV2SLS.from_formula("y ~ 1 + t2 + [shock ~ z]", data, weights=...)`, which also does OLS when there is no
# `[... ~ ...]` part) or statsmodels. Building the instrument and the shock-level data are pandas merges and
# group-bys. `week2_fast.py` gives the same numbers with numpy (the placebo uses more draws there).
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run week2_simple.py`.


# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

try:                                       # VS Code cells / Jupyter: figures appear under each cell
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:                          # plain script: only save the figures
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures_simple"
FIGS.mkdir(exist_ok=True)
rng = np.random.default_rng(1)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# %% [markdown]
# ## S0. The data (slide "Running example")
# Three files: the CZ panel, the lagged industry shares, and the industry shocks. The instrument is a merge and
# a group-by: multiply each CZ's share in industry $n$ by the shock to $n$, then add up over industries.

# %%
df = pd.read_csv(DATA / "ADHdata_AKM.csv")
df["t2"] = df.t2.astype(int)
df["year"] = np.where(df.t2 == 1, 2000, 1990)               # the year the shares refer to
shares = pd.read_stata(DATA / "Lshares.dta")                # czone, year, sic87dd, ind_share
shocks = pd.read_stata(DATA / "shocks.dta")                 # year, sic87dd, g (and US imports)
names = pd.read_stata(DATA / "sic_code_desc.dta").set_index("sic").description

m = shares.merge(shocks[["year", "sic87dd", "g"]], on=["year", "sic87dd"])
m["sg"] = m.ind_share * m.g
inst = m.groupby(["czone", "year"]).agg(z=("sg", "sum"), S=("ind_share", "sum")).reset_index()
df = df.merge(inst, on=["czone", "year"])
print(df[["czone", "year", "d_sh_empl_mfg", "shock", "z", "IV", "S"]].head(3).round(3))
print(f"max |z - ADH's IV| = {(df.z - df.IV).abs().max():.1e}")
print(f"sum of shares S: mean {df.S.mean():.3f}, range {df.S.min():.3f} to {df.S.max():.3f}")

df["S_90"] = df.S * (1 - df.t2)                            # S x period: one coefficient per period
df["S_00"] = df.S * df.t2
ADH = ("l_shind_manuf_cbp + l_sh_popedu_c + l_sh_popfborn + l_sh_empl_f + l_sh_routine33"
       " + l_task_outsource + C(division)")
CONTROLS = {"t2": "t2", "S": "t2 + S_90 + S_00", "adh": f"t2 + {ADH}", "adhS": f"t2 + S_90 + S_00 + {ADH}"}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].scatter(np.arcsinh(shocks.g), np.arcsinh(shocks.g_importsUSA), s=9, alpha=0.6,
              c=np.where(shocks.year == 1990, BLUE, RED))
ax[0].set_xlabel("imports from China, 8 other countries (asinh)")
ax[0].set_ylabel("imports from China, US (asinh)")
ax[0].set_title("397 industries x 2 periods: the shocks", fontsize=11)
for t, c in [(0, BLUE), (1, RED)]:                          # binned scatter: 20 bins of exposure per period
    d = df[df.t2 == t].assign(bin=lambda d: pd.qcut(d.shock, 20, labels=False))
    d = d.assign(wx=d.shock * d.weights, wy=d.d_sh_empl_mfg * d.weights).groupby("bin")[["wx", "wy", "weights"]].sum()
    ax[1].scatter(d.wx / d.weights, d.wy / d.weights, color=c, s=22)
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_xlabel("$x$: exposure"); ax[1].set_ylabel("$y$, pp")
ax[1].set_title("722 commuting zones: exposure and job loss (binned)", fontsize=11)
print(shocks.groupby("year").apply(lambda d: d.g.corr(d.g_importsUSA)).round(2).rename("corr(US, other countries)"))
show(fig, "data")


# %% [markdown]
# # Part A. Shift-share IV
# $$y_{\ell t} = \beta\, x_{\ell t} + \gamma' w_{\ell t} + \varepsilon_{\ell t}, \qquad
#   x_{\ell t} \text{ instrumented by } z_{\ell t}$$
# One helper runs every CZ-level regression: weighted by CZ population, SEs clustered by state (as ADH).

# %%
def fit(formula, data=df, cov="clustered", w="weights", cl="statefip"):
    kw = {"clusters": data[cl]} if cov == "clustered" else {}
    return IV2SLS.from_formula(formula, data, weights=data[w]).fit(cov_type=cov, **kw)


def iv(controls, z="z", data=df, cov="clustered", w="weights"):
    """2SLS of y on x with the given controls. Returns coef, SE and first-stage F (same SE type)."""
    r = fit(f"d_sh_empl_mfg ~ 1 + {controls} + [shock ~ {z}]", data, cov, w)
    fs = fit(f"shock ~ 1 + {controls} + {z}", data, cov, w)
    return r.params["shock"], r.std_errors["shock"], (fs.params[z] / fs.std_errors[z]) ** 2


def binscatter(ax, v, u, controls, color, nb=20):
    """Binned scatter of u on v, both residualised on the controls (WLS), means added back."""
    rv = smf.wls(f"{v} ~ {controls}", df, weights=df.weights).fit().resid + np.average(df[v], weights=df.weights)
    ru = smf.wls(f"{u} ~ {controls}", df, weights=df.weights).fit().resid + np.average(df[u], weights=df.weights)
    b = pd.qcut(rv, nb, labels=False)
    gm = pd.DataFrame({"v": rv * df.weights, "u": ru * df.weights, "w": df.weights, "b": b}).groupby("b").sum()
    ax.scatter(gm.v / gm.w, gm.u / gm.w, color=color, s=22)


# %% [markdown]
# ## S1. OLS
# $$y_{\ell t} = \beta\, x_{\ell t} + \delta_t + \varepsilon_{\ell t}$$

# %%
ols = fit("d_sh_empl_mfg ~ 1 + t2 + shock")
print(ols.summary.tables[1])
print(f"S1 OLS: {ols.params['shock']:.3f} ({ols.std_errors['shock']:.3f})")
ols_c = fit(f"d_sh_empl_mfg ~ 1 + {CONTROLS['adh']} + shock")
print(f"S1 OLS with ADH controls: {ols_c.params['shock']:.3f} ({ols_c.std_errors['shock']:.3f})")
fig, ax = plt.subplots(figsize=(4.8, 3.3))
binscatter(ax, "shock", "d_sh_empl_mfg", "t2", GREY)
ax.set_title(f"OLS: slope {ols.params['shock']:.2f}", fontsize=11)
ax.set_xlabel("$x$: Chinese import exposure per worker\n(\\$1000 per decade, net of period means; 20 bins)")
ax.set_ylabel("$y$: change in manuf. employment /\nworking-age population (pp/decade)")
show(fig, "s1_ols")


# %% [markdown]
# ## S2. The ADH instrument
# First stage, reduced form, and their ratio: the IV.

# %%
fs = fit("shock ~ 1 + t2 + z")
rf = fit("d_sh_empl_mfg ~ 1 + t2 + z")
print(f"S2 first stage {fs.params['z']:.3f} ({fs.std_errors['z']:.3f}), reduced form {rf.params['z']:.3f} "
      f"({rf.std_errors['z']:.3f}), ratio {rf.params['z'] / fs.params['z']:.3f}")
res = {k: iv(c) for k, c in CONTROLS.items()}
print(f"S2 IV: {res['t2'][0]:.3f} ({res['t2'][1]:.3f}), F = {res['t2'][2]:.0f}")
print(f"S2 IV with ADH controls: {res['adh'][0]:.3f} ({res['adh'][1]:.3f}), F = {res['adh'][2]:.0f}")
df["one"] = 1.0
print(f"   unweighted: {iv(CONTROLS['adh'], w='one')[0]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
binscatter(ax[0], "z", "shock", "t2", BLUE)
ax[0].set_title(f"First stage: slope {fs.params['z']:.2f}", fontsize=11)
ax[0].set_xlabel("$z$: instrument, exposure via imports into\n8 other countries (\\$1000 per worker; 20 bins)"); ax[0].set_ylabel("$x$: Chinese import exposure per worker\n(\\$1000 per decade)")
binscatter(ax[1], "z", "d_sh_empl_mfg", "t2", RED)
ax[1].set_title(f"Reduced form: slope {rf.params['z']:.2f}", fontsize=11)
ax[1].set_xlabel("$z$: instrument, exposure via imports into\n8 other countries (\\$1000 per worker; 20 bins)"); ax[1].set_ylabel("$y$: change in manuf. employment /\nworking-age population (pp/decade)")
show(fig, "s2_ssiv")


# %% [markdown]
# ## S3. Incomplete shares
# With random shocks of mean $\mu_t$, $E[z_{\ell t}] = \mu_t S_{\ell t}$. The fix: control for $S$ by period.

# %%
mu = shocks.groupby("year").g.mean()                       # mean shock in each period
for k, lab in [("t2", "period only"), ("S", "+ S x period"), ("adh", "ADH controls"),
               ("adhS", "ADH controls + S x period")]:
    print(f"S3 {lab:28s} {res[k][0]:7.3f} ({res[k][1]:.3f})  F = {res[k][2]:.0f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for yr, c in [(1990, BLUE), (2000, RED)]:
    d = df[df.year == yr]
    ax[0].scatter(d.S, d.z, s=5, alpha=0.4, color=c)
    ax[0].plot([0, 0.7], [0, 0.7 * mu[yr]], color=c, label=f"$E[z] = {mu[yr]:.1f}\\,S$")
ax[0].set_ylim(0, 25); ax[0].legend(frameon=False)
ax[0].set_xlabel("$S$"); ax[0].set_ylabel("$z$")
order = ["t2", "S", "adh", "adhS"]
ax[1].errorbar([res[k][0] for k in order], -np.arange(4), xerr=[1.96 * res[k][1] for k in order],
               fmt="o", capsize=3, color=BLUE)
ax[1].set_yticks(-np.arange(4), ["period", "+ S x period", "ADH", "ADH + S x period"])
ax[1].axvline(0, color="grey", lw=0.6)
show(fig, "s3_shares")


# %% [markdown]
# ## S4. Standard errors
# **Exposure-robust SEs from the shock-level IV** (Borusyak, Hull and Jaravel 2022), in four lines of pandas:
# 1. residualise $y$ and $x$ on the CZ controls (WLS);
# 2. attach each CZ's residuals to its industry shares;
# 3. average them by industry with weights $w_\ell s_{\ell n}$: $\bar y_n$, $\bar x_n$, and the weight $s_n = \sum_\ell w_\ell s_{\ell n}$;
# 4. run the IV of $\bar y_n$ on $\bar x_n$ with instrument $g_n$ and period effects, weights $s_n$, SEs clustered by SIC3.
#
# The coefficient is identical to the CZ-level one; only the SE changes.

# %%
def shock_level(cols, controls, data=df):
    """Shock-level data set: s_n-weighted averages of the residualised CZ variables in `cols`."""
    r = data[["czone", "year", "weights"]].copy()
    for c in cols:
        r[c] = smf.wls(f"{c} ~ {controls}", data, weights=data.weights).fit().resid
    sl = shares.merge(r, on=["czone", "year"])
    sl["ws"] = sl.weights * sl.ind_share
    for c in cols:
        sl[c] = sl[c] * sl.ws
    out = sl.groupby(["year", "sic87dd"])[cols + ["ws"]].sum()
    out[cols] = out[cols].div(out.ws, axis=0)
    out = out.reset_index().merge(shocks[["year", "sic87dd", "g"]], on=["year", "sic87dd"])
    out["sic3"] = (out.sic87dd // 10).astype(int)
    return out


def ss_iv(ycol, xcol, controls, data=df, gdf=None):
    """Shock-level IV of ybar on xbar (instrument g), period effects, weights s_n, SIC3 clusters.
    gdf: other shocks to use as the instrument (year, sic87dd, g), e.g. the placebo draws."""
    sh = shock_level([ycol, xcol], controls, data)
    if gdf is not None:
        sh = sh.drop(columns="g").merge(gdf, on=["year", "sic87dd"])
    r = IV2SLS.from_formula(f"{ycol} ~ 0 + C(year) + [{xcol} ~ g]", sh, weights=sh.ws).fit(
        cov_type="clustered", clusters=sh.sic3)
    return r.params[xcol], r.std_errors[xcol]


for k in ["S", "adhS"]:
    b_, se_r, _ = iv(CONTROLS[k], cov="robust")
    _, se_c, _ = iv(CONTROLS[k])
    bs, se_e = ss_iv("d_sh_empl_mfg", "shock", CONTROLS[k])
    print(f"S4 {k:5s} beta {b_:.3f}: robust {se_r:.3f}, state {se_c:.3f}, exposure-robust {se_e:.3f} "
          f"(shock-level beta {bs:.3f})")

# %% [markdown]
# **The placebo** (Adão, Kolesár and Morales 2019). Draw random shocks $g^*_n \sim N(0,1)$, build $z^*$, and regress
# the actual $y$ on $z^*$ with the preferred controls. The truth is zero, so a 5% test should reject 5% of the time.
# 200 draws here (each one is three regressions); `week2_fast.py` does 1000.

# %%
R = 200
rej = {"robust": 0, "state clusters": 0, "exposure-robust": 0}
ctrl = CONTROLS["adhS"]
for r in range(R):
    gstar = shocks[["year", "sic87dd"]].assign(g=rng.standard_normal(len(shocks)))
    zz = shares.merge(gstar, on=["year", "sic87dd"]).assign(sg=lambda d: d.ind_share * d.g)
    d = df.drop(columns=["zstar"], errors="ignore").merge(
        zz.groupby(["czone", "year"]).sg.sum().rename("zstar").reset_index(), on=["czone", "year"])
    t_rob = fit(f"d_sh_empl_mfg ~ 1 + {ctrl} + zstar", d, cov="robust").tstats["zstar"]
    t_st = fit(f"d_sh_empl_mfg ~ 1 + {ctrl} + zstar", d).tstats["zstar"]
    b_, se_ = ss_iv("d_sh_empl_mfg", "zstar", ctrl, d, gdf=gstar)     # reduced form = IV with x = z*
    rej["robust"] += abs(t_rob) > 1.96
    rej["state clusters"] += abs(t_st) > 1.96
    rej["exposure-robust"] += abs(b_ / se_) > 1.96
rej = {k: v / R for k, v in rej.items()}
print("S4 placebo rejection rates at 5%:", rej)
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(-np.arange(3), list(rej.values()), color=[GREY, BLUE, GREEN])
ax.set_yticks(-np.arange(3), list(rej))
ax.axvline(0.05, color=RED, ls="--")
ax.set_title("Placebo: rejection rate of a 5% test", fontsize=11)
show(fig, "s4_se")


# %% [markdown]
# ## S5. The shock-level view
# The importance weight of shock $n$ is $s_n = \sum_\ell w_\ell s_{\ell n}$; the effective number of shocks is
# $1 / \sum_n \hat s_n^2$ with $\hat s_n = s_n / \sum_m s_m$.

# %%
sh = shock_level(["d_sh_empl_mfg", "shock"], CONTROLS["adhS"])
sh["sw"] = sh.ws / sh.ws.sum()
print(f"S5 effective number of shocks {1 / (sh.sw ** 2).sum():.0f} (of {len(sh)}); "
      f"of SIC3 clusters {1 / (sh.groupby('sic3').sw.sum() ** 2).sum():.0f} (of {sh.sic3.nunique()})")
sh["name"] = sh.sic87dd.astype(int).map(names)
print(sh.sort_values("sw", ascending=False)[["year", "name", "g", "sw"]].head(5).round(4).to_string())
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter(np.arcsinh(sh.g), sh.d_sh_empl_mfg, s=3000 * sh.sw, alpha=0.45,
           c=np.where(sh.year == 1990, BLUE, RED), edgecolor="none")
ax.set_ylim(-2, 2)
ax.set_xlabel("$g_n$ (asinh)"); ax.set_ylabel("$\\bar y_n$")
show(fig, "s5_shocks")


# %% [markdown]
# ## S6. Rotemberg weights (the exogenous-shares view)
# Each industry share, used alone as the instrument, gives $\hat\beta_k = s_k' M y / s_k' M x$. The shift-share IV is
# $\hat\beta = \sum_k \hat\alpha_k \hat\beta_k$ with $\hat\alpha_k \propto \tilde g_k\, s_k' M x$ ($\tilde g$: shocks demeaned
# within period). One loop over the 794 share instruments: residualise the share on the controls, two weighted
# sums.

# %%
wide = (shares.pivot_table(index=["czone", "year"], columns=["year", "sic87dd"], values="ind_share",
                           fill_value=0.0, aggfunc="sum"))
wide.columns = [f"s_{int(yr)}_{int(n)}" for yr, n in wide.columns]
d6 = df.merge(wide.reset_index(), on=["czone", "year"], how="left").fillna(0.0)
xr = smf.wls(f"shock ~ {CONTROLS['adhS']}", d6, weights=d6.weights).fit().resid
yr_ = smf.wls(f"d_sh_empl_mfg ~ {CONTROLS['adhS']}", d6, weights=d6.weights).fit().resid
Xc = smf.wls(f"shock ~ {CONTROLS['adhS']}", d6, weights=d6.weights).fit().model.exog   # the control matrix
sw = np.sqrt(d6.weights.to_numpy())
rows = []
for k, row in shocks.iterrows():
    col = f"s_{int(row.year)}_{int(row.sic87dd)}"
    s = d6[col].to_numpy()
    s_res = s - Xc @ np.linalg.lstsq(Xc * sw[:, None], s * sw, rcond=None)[0]   # M s_k (WLS residual)
    rows.append({"year": row.year, "sic87dd": row.sic87dd, "g": row.g,
                 "sx": np.sum(d6.weights * s_res * xr), "sy": np.sum(d6.weights * s_res * yr_)})
rot = pd.DataFrame(rows)
rot["g_dm"] = rot.g - rot.year.map(mu)
rot["alpha"] = rot.g_dm * rot.sx / (rot.g_dm * rot.sx).sum()
rot["beta_k"] = rot.sy / rot.sx
rot["name"] = rot.sic87dd.astype(int).map(names)
print(f"S6 sum alpha_k beta_k = {(rot.alpha * rot.beta_k).sum():.3f}; "
      f"negative weights sum to {rot.alpha[rot.alpha < 0].sum():.3f}")
top = rot.reindex(rot.alpha.abs().sort_values(ascending=False).index).head(10)
print(top[["year", "name", "alpha", "beta_k", "g"]].round(3).to_string())
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.barh(-np.arange(10), top.alpha, color=BLUE)
ax.set_yticks(-np.arange(10), [f"{n[:30]} ({int(y)}s)" for n, y in zip(top.name, top.year)], fontsize=8)
show(fig, "s6_rotemberg")


# %% [markdown]
# ## S7. Balance
# Regress each predetermined CZ characteristic (in s.d. units) on $z$ (in residual s.d. units), with the controls
# the design needs: period and $S$ x period. State-clustered vs exposure-robust SEs.

# %%
zres = smf.wls(f"z ~ {CONTROLS['S']}", df, weights=df.weights).fit().resid
df["zs"] = df.z / np.sqrt(np.cov(zres, aweights=df.weights))
bal = {"l_sh_popedu_c": "college share", "l_sh_popfborn": "foreign-born share", "l_sh_empl_f": "female emp. share",
       "l_sh_routine33": "routine occ. share", "l_task_outsource": "offshorability",
       "l_shind_manuf_cbp": "manuf. share (CBP)"}
out = []
for v, lab in bal.items():
    df[v + "_sd"] = df[v] / np.sqrt(np.cov(df[v], aweights=df.weights))
    r = fit(f"{v}_sd ~ 1 + {CONTROLS['S']} + zs")
    b_, se_e = ss_iv(v + "_sd", "zs", CONTROLS["S"])
    out.append((lab, r.params["zs"], r.std_errors["zs"], se_e))
    print(f"S7 {lab:20s} {r.params['zs']:6.3f}  state {r.std_errors['zs']:.3f}  exposure-robust {se_e:.3f}")
fig, ax = plt.subplots(figsize=(7, 3.4))
for j, (lab, b_, s1, s2) in enumerate(out):
    ax.errorbar(b_, -j + 0.1, xerr=1.96 * s1, fmt="o", color=BLUE, capsize=3)
    ax.errorbar(b_, -j - 0.1, xerr=1.96 * s2, fmt="o", color=GREEN, capsize=3)
ax.set_yticks(-np.arange(len(out)), [o[0] for o in out])
ax.axvline(0, color="grey", lw=0.6)
show(fig, "s7_balance")


# %% [markdown]
# # Part B. Event studies with staggered adoption
# A simulated panel where we know the truth: 600 units, 20 years, adoption in year 6, 10 or 14, a quarter never
# adopts. The effect grows with time since adoption and is larger for early adopters:
# $\tau_{it} = c_{E_i}(1 + 0.3\,(t - E_i))$ with $c = 1.5, 1, 0.5$ for the three cohorts.

# %%
N, T = 600, 20
scale = {6: 1.5, 10: 1.0, 14: 0.5, 99: 0.0}                # 99 = never adopts
E = np.array([6, 10, 14, 99])[np.arange(N) % 4]
rng = np.random.default_rng(2)                              # own seed: same draws as week2_fast.py
alpha_i = rng.normal(0, 1, N)
panel = pd.DataFrame({"i": np.repeat(np.arange(N), T), "t": np.tile(np.arange(1, T + 1), N),
                      "E": np.repeat(E, T), "a": np.repeat(alpha_i, T)})
panel["D"] = (panel.t >= panel.E).astype(int)
panel["k"] = panel.t - panel.E                              # years since adoption
panel["tau"] = panel.E.map(scale) * (1 + 0.3 * panel.k) * panel.D
panel["Y"] = panel.a + 0.1 * panel.t + np.sin(panel.t / 3) + panel.tau + rng.normal(0, 0.5, len(panel))
att = panel.loc[panel.D == 1, "tau"].mean()
fig, ax = plt.subplots(figsize=(7, 3.4))
for c, col in [(6, BLUE), (10, GREEN), (14, ORANGE), (99, GREY)]:
    ax.plot(panel[panel.E == c].groupby("t").Y.mean(), color=col, label="never" if c == 99 else f"adopt in {c}")
ax.legend(frameon=False, fontsize=9)
show(fig, "e0_sim")


# %% [markdown]
# ## E1. Static TWFE
# $$Y_{it} = \alpha_i + \lambda_t + \beta D_{it} + \varepsilon_{it}$$
# The implicit weights: $\hat\beta = \sum_{it} \omega_{it} Y_{it}$ with $\omega_{it} \propto \tilde D_{it}$, the residual from
# regressing $D$ on unit and year dummies.

# %%
twfe = smf.ols("Y ~ D + C(i) + C(t)", panel).fit()
panel["Dt"] = smf.ols("D ~ C(i) + C(t)", panel).fit().resid
panel["omega"] = panel.Dt / (panel.Dt * panel.D).sum()
treated = panel[panel.D == 1]
print(f"E1 static TWFE {twfe.params['D']:.3f} vs true ATT {att:.3f}; "
      f"negative weights sum to {treated.omega[treated.omega < 0].sum():.3f}, "
      f"on {np.mean(treated.omega < 0):.2f} of treated cells")
fig, ax = plt.subplots(figsize=(6, 3.4))
for c, col in [(6, BLUE), (10, GREEN), (14, ORANGE)]:
    w_ = treated[treated.E == c].groupby("k").omega.sum()
    ax.plot(w_.index, w_.values, "o-", ms=3, color=col, label=f"adopt in {c}")
ax.axhline(0, color="grey", lw=0.6); ax.legend(frameon=False, fontsize=9)
show(fig, "e1_twfe")


# %% [markdown]
# ## E2. Event studies: TWFE, imputation and LP-DiD
# - **TWFE event study**: dummies for years since adoption, $-4..4$, both ends binned (common practice), $k=-1$ omitted.
# - **Imputation** (Borusyak, Jaravel and Spiess 2024): unit and year effects from untreated cells only, then
#   $\hat\tau_{it} = Y_{it} - \hat\alpha_i - \hat\lambda_t$ averaged by $k$.
# - **LP-DiD** (Dube, Girardi, Jordà and Taylor 2025): for each $h$,
#   $Y_{i,t+h} - Y_{i,t-1} = \beta_h \Delta D_{it} + \delta^h_t + e_{it}$, on newly adopting units and clean controls.

# %%
kk = list(range(-4, 5))                                         # a typical short event window
panel["kb"] = np.where(panel.E < 99, panel.k.clip(-4, 4), -1)   # never-treated in the reference group
for k in kk:
    if k != -1:
        panel[f"ev_{k + 5}"] = (panel.kb == k).astype(int)       # ev_1 = k <= -4, ..., ev_9 = k >= 4
ev = " + ".join(f"ev_{k + 5}" for k in kk if k != -1)
es = smf.ols(f"Y ~ {ev} + C(i) + C(t)", panel).fit()
es_twfe = [es.params[f"ev_{k + 5}"] if k != -1 else 0.0 for k in kk]

untr = panel[panel.D == 0]
imp = smf.ols("Y ~ C(i) + C(t)", untr).fit()
panel["tau_hat"] = panel.Y - imp.predict(panel)
es_imp = [panel[(panel.k == k) & (panel.E < 99)].tau_hat.mean() for k in kk]   # k < 0: pre-period check

wideY = panel.pivot(index="i", columns="t", values="Y")
wideD = panel.pivot(index="i", columns="t", values="D")


def lpdid(h):
    rows = []
    for t in range(2, T + 1):
        if not 1 <= t + h <= T:
            continue
        newly = (wideD[t] == 1) & (wideD[t - 1] == 0)
        clean = wideD[max(t, t + h)] == 0
        keep = newly | clean
        rows.append(pd.DataFrame({"dy": wideY[t + h][keep] - wideY[t - 1][keep],
                                  "dD": newly[keep].astype(int), "t": t}))
    return smf.ols("dy ~ dD + C(t)", pd.concat(rows)).fit().params["dD"]


es_lp = [lpdid(k) if k != -1 else 0.0 for k in kk]
truth = [panel[panel.k == k].tau.mean() if k >= 0 else 0.0 for k in kk]
print(pd.DataFrame({"k": kk, "truth": truth, "TWFE": es_twfe, "imputation": es_imp, "LP-DiD": es_lp})
      .set_index("k").T.round(2).to_string())
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.plot(kk, truth, color="black", lw=2.5, alpha=0.25, label="truth")
ax.plot(kk, es_twfe, "o-", color=GREY, label="TWFE event study")
ax.plot(kk, es_imp, "s-", color=BLUE, label="imputation")
ax.plot(kk, es_lp, "^--", color=RED, label="LP-DiD")
ax.legend(frameon=False, fontsize=9)
show(fig, "e2_es")

# %%
