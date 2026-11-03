# %% [markdown]
# # In-class exercise: the China shock, your turn (solution)
#
# **How this works.** Six short parts. Each one changes one thing in the lecture's specification and asks what
# happens. The code style is that of `week2_simple.py`: pandas merges for the instrument, one formula per
# regression. The student version, `exercise_ADH_student.py`, has gaps marked `# TODO` with `...`, and a
# `check(...)` line under each gap prints ✓ when your result matches.
#
# **Preferred specification** (the end point of the lecture): ADH's controls plus the incomplete-share control
# $S$ x period, weighted by CZ population, SEs clustered by state:
# $$y_{\ell t} = \beta x_{\ell t} + \gamma' w_{\ell t} + \delta_t + \lambda_t S_{\ell t} + \varepsilon_{\ell t},
#   \qquad x_{\ell t} \text{ instrumented by } z_{\ell t} = \sum_n s_{\ell n t} g_{nt}$$
#
# | Part | Question |
# | --- | --- |
# | 1 | Build the instrument yourself |
# | 2 | Where did the jobs go? Total and non-manufacturing employment |
# | 3 | Why imports from *other* countries? Use US imports as the shift |
# | 4 | Exposure-robust standard errors for Part 2 |
# | 5 | Leave one industry out: computers |
# | 6 | Event studies: when is static TWFE fine? |
#
# **Run** cell by cell in VS Code, or `uv run exercise_ADH.py`.


# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def check(name, value, target, tol=0.005):
    """Print whether your result matches the solution."""
    try:
        value = round(float(value), 3)
        ok = abs(value - target) < tol
    except (TypeError, ValueError):
        ok = False
    print(f"{'✓' if ok else '✗'} {name}: yours = {value}, should be about {target}")


df = pd.read_csv(DATA / "ADHdata_AKM.csv")
df["t2"] = df.t2.astype(int)
df["year"] = np.where(df.t2 == 1, 2000, 1990)
shares = pd.read_stata(DATA / "Lshares.dta")
shocks = pd.read_stata(DATA / "shocks.dta")
ADH = ("l_shind_manuf_cbp + l_sh_popedu_c + l_sh_popfborn + l_sh_empl_f + l_sh_routine33"
       " + l_task_outsource + C(division)")
PREF = f"t2 + S_90 + S_00 + {ADH}"


def iv(data, y="d_sh_empl_mfg", z="z", controls=PREF):
    """Weighted 2SLS of y on shock (instrument z), state clusters. Returns the fitted result."""
    f = f"{y} ~ 1 + {controls} + [shock ~ {z}]"
    return IV2SLS.from_formula(f, data, weights=data.weights).fit(cov_type="clustered",
                                                                     clusters=data.statefip)


# %% [markdown]
# ## Part 1. Build the instrument
# Merge the shares with the shocks on (year, industry), multiply, and add up by (CZ, year). Do the same for the
# sum of shares $S$. Then add $S$ x period.

# %%
m = shares.merge(shocks[["year", "sic87dd", "g"]], on=["year", "sic87dd"])
m["sg"] = m.ind_share * m.g
inst = m.groupby(["czone", "year"]).agg(z=("sg", "sum"), S=("ind_share", "sum")).reset_index()
df = df.merge(inst, on=["czone", "year"])
df["S_90"] = df.S * (1 - df.t2)
df["S_00"] = df.S * df.t2
check("mean of z", df.z.mean(), 1.755)
check("max |z - IV|", (df.z - df.IV).abs().max(), 0.0)

res = iv(df)
check("beta, preferred specification", res.params["shock"], -0.270)


# %% [markdown]
# ## Part 2. Where did the jobs go?
# The lecture used manufacturing employment / population. Run the preferred specification for total employment
# (`d_sh_empl`) and non-manufacturing employment (`d_sh_empl_nmfg`). The three outcomes add up
# (mfg + non-mfg = total). Do non-manufacturing jobs absorb the manufacturing loss?

# %%
r_tot = iv(df, y="d_sh_empl")
r_nm = iv(df, y="d_sh_empl_nmfg")
print(f"manufacturing {res.params['shock']:.3f}, non-manufacturing {r_nm.params['shock']:.3f}, "
      f"total {r_tot.params['shock']:.3f}")
check("beta, total employment", r_tot.params["shock"], -0.162)
check("beta, non-manufacturing", r_nm.params["shock"], 0.109)
# Answer: non-manufacturing employment / population rises by 0.11 pp per $1000, so total employment falls by
# 0.16 pp, about 60% of the manufacturing loss, and is no longer significant (SE 0.19). BHJ (2022) make the same
# point: with the S x period control the effect on total employment is much smaller than in ADH.


# %% [markdown]
# ## Part 3. Why imports from other countries?
# Build a second instrument `z_us` with US imports from China (`g_importsUSA`) as the shift, and use it instead
# of `z`. US imports also reflect US demand shocks: industries where US demand fell import less. Which way
# should that bias the estimate?

# %%
m["sg_us"] = m.ind_share * m.merge(shocks[["year", "sic87dd", "g_importsUSA"]],
                                   on=["year", "sic87dd"]).g_importsUSA
df = df.merge(m.groupby(["czone", "year"]).sg_us.sum().rename("z_us").reset_index(), on=["czone", "year"])
r_us = iv(df, z="z_us")
print(f"z (other countries) {res.params['shock']:.3f}, z_us (US imports) {r_us.params['shock']:.3f}")
check("beta with US imports as the shift", r_us.params["shock"], -0.081)
# Answer: much closer to zero (-0.08). A US demand slump in an industry lowers both imports and employment, so
# US imports partly move with US employment *for other reasons*. That pushes the estimate towards zero, the same
# reason OLS is smaller than the IV. Imports into other rich countries capture China's supply side instead.


# %% [markdown]
# ## Part 4. Exposure-robust standard errors
# `ss_iv` (from `week2_simple.py`) runs the equivalent shock-level IV. Use it for total employment (Part 2).
# Is the effect on total employment still significant?

# %%
def ss_iv(ycol, xcol, controls, data):
    r = data[["czone", "year", "weights"]].copy()
    for c in [ycol, xcol]:
        r[c] = smf.wls(f"{c} ~ {controls}", data, weights=data.weights).fit().resid
    sl = shares.merge(r, on=["czone", "year"])
    sl["ws"] = sl.weights * sl.ind_share
    for c in [ycol, xcol]:
        sl[c] = sl[c] * sl.ws
    sh = sl.groupby(["year", "sic87dd"])[[ycol, xcol, "ws"]].sum()
    sh[[ycol, xcol]] = sh[[ycol, xcol]].div(sh.ws, axis=0)
    sh = sh.reset_index().merge(shocks[["year", "sic87dd", "g"]], on=["year", "sic87dd"])
    sh["sic3"] = (sh.sic87dd // 10).astype(int)
    out = IV2SLS.from_formula(f"{ycol} ~ 0 + C(year) + [{xcol} ~ g]", sh, weights=sh.ws).fit(
        cov_type="clustered", clusters=sh.sic3)
    return out.params[xcol], out.std_errors[xcol]


b_e, se_e = ss_iv("d_sh_empl", "shock", PREF, df)
print(f"total employment: {b_e:.3f}, state-clustered SE {r_tot.std_errors['shock']:.3f}, "
      f"exposure-robust SE {se_e:.3f}")
check("exposure-robust SE, total employment", se_e, 0.139)
# Answer: no. The exposure-robust SE (0.14) is smaller than the state-clustered one here, but -0.16 is still
# well within two SEs of zero.


# %% [markdown]
# ## Part 5. Leave one industry out
# Electronic computers (SIC 3571) in the 2000s has the largest Rotemberg weight. Rebuild `z` and `S` without
# industry 3571 (in both periods) and re-estimate. Does one industry drive the result?

# %%
m5 = m[m.sic87dd != 3571]
inst5 = m5.groupby(["czone", "year"]).agg(z5=("sg", "sum"), S5=("ind_share", "sum")).reset_index()
d5 = df.merge(inst5, on=["czone", "year"])
d5["S_90"] = d5.S5 * (1 - d5.t2)
d5["S_00"] = d5.S5 * d5.t2
r5 = iv(d5, z="z5")
print(f"without computers {r5.params['shock']:.3f} ({r5.std_errors['shock']:.3f}) vs all {res.params['shock']:.3f}")
check("beta without SIC 3571", r5.params["shock"], -0.263)
# Answer: no: -0.26 instead of -0.27. The biggest Rotemberg weight does not drive the estimate.


# %% [markdown]
# ## Part 6. Event studies: when is static TWFE fine?
# The lecture's simulation, with the effect $\tau_{it} = c_{E_i}(1 + \text{slope}\cdot(t - E_i))$. Set the
# effect to be the **same for every cohort and constant over time** (all $c = 1$, slope $= 0$) and compare the
# static TWFE estimate with the true ATT. Then try $c = 1$ for all cohorts but slope $= 0.3$: which of the two
# assumptions matters?

# %%
def simulate(scale, slope, seed=2):
    rng = np.random.default_rng(seed)
    N, T = 600, 20
    E = np.array([6, 10, 14, 99])[np.arange(N) % 4]
    panel = pd.DataFrame({"i": np.repeat(np.arange(N), T), "t": np.tile(np.arange(1, T + 1), N),
                          "E": np.repeat(E, T), "a": np.repeat(rng.normal(0, 1, N), T)})
    panel["D"] = (panel.t >= panel.E).astype(int)
    panel["tau"] = panel.E.map(scale) * (1 + slope * (panel.t - panel.E)) * panel.D
    panel["Y"] = panel.a + 0.1 * panel.t + np.sin(panel.t / 3) + panel.tau + rng.normal(0, 0.5, len(panel))
    return panel


def twfe_vs_att(panel):
    b = smf.ols("Y ~ D + C(i) + C(t)", panel).fit().params["D"]
    return b, panel.loc[panel.D == 1, "tau"].mean()


b_hom, att_hom = twfe_vs_att(simulate({6: 1, 10: 1, 14: 1, 99: 0}, slope=0.0))
print(f"homogeneous, constant: TWFE {b_hom:.3f}, ATT {att_hom:.3f}")
check("TWFE with a constant, common effect", b_hom, 0.974)
b_dyn, att_dyn = twfe_vs_att(simulate({6: 1, 10: 1, 14: 1, 99: 0}, slope=0.3))
print(f"common but growing effect: TWFE {b_dyn:.3f}, ATT {att_dyn:.3f}")
check("TWFE with a common, growing effect", b_dyn, 1.639)
# Answer: with a constant, common effect TWFE is fine (0.97 vs 1.00, noise). A common but growing effect is
# enough to break it (1.64 vs 2.65): already-treated units serve as controls while their effect is still growing,
# even though all cohorts are alike.
