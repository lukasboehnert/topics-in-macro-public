# %% [markdown]
# # Week 1 lab: local projections and VARs, step by step
#
# **Question.** What happens to UK output after the 1-year interest rate rises by 1 percentage point?
#
# **Data** (Cesa-Bianchi, Thwaites and Vicondoa 2020), monthly, 1979m7 to 2009m2:
#
# - $y$: output, 100 × log of monthly GDP
# - $u$: unemployment rate, %
# - $i$: 1-year gilt yield, %
# - $z$: Cloyne and Hürtgen (2016) narrative monetary policy shock, pp
#
# The cells follow the slides, and each one draws the figure on its slide:
#
# | Part | Cells |
# | --- | --- |
# | A. Local projections | L1 no controls, L2 lags and date-$t$ controls, L3 LP-IV, L4 lags with an instrument, L5 standard errors, L6 variance decomposition |
# | B. VARs | V0 lag length, identification by regression, V1 recursive VAR, V2 ordering, V3 lag length, proxy SVAR as 2SLS, V4 instrument, V5 variance decomposition |
# | C. LP vs VAR | C1 horizon by horizon, C2 one instrument two estimators, C3 bias vs variance, C4 variance decompositions |
# | Take-home | C5 non-invertibility |
#
# **How the code is written.** Every local projection is a regression formula on one DataFrame of leads and lags, for example
#
# `smf.ols("y_h12 ~ i + y + u + y_l1 + u_l1 + i_l1", data=df).fit(cov_type="HC1")`
#
# so the same code carries over to the exercises (`exercise_PT.py`, `exercise_PT_student.py`). The VARs are written out with numpy and iterated by hand. The figures are the ones on the slides, saved to `figures_simple/`.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with
#
# `uv run week1_simple.py`


# %% [markdown]
# ## Setup
# Packages, horizons ($h = 0, \dots, 48$ months) and small helpers: `lag`, `lead` and `ols` for the VAR part, and the plotting functions that draw the slide figures.

# %%
from pathlib import Path

import matplotlib
import matplotlib.dates
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
DATA = HERE / "data" / "CTV_DATA.xlsx"
FIGS = HERE / "figures_simple"
FIGS.mkdir(exist_ok=True)
H = 48                       # horizons 0..H (months)
hs = np.arange(H + 1)
rng = np.random.default_rng(1)


def lag(x, l):
    """x_{t-l}, with NaN where it does not exist."""
    out = np.full(len(x), np.nan)
    out[l:] = x[:len(x) - l]
    return out


def lead(x, h):
    """x_{t+h}, with NaN where it does not exist."""
    out = np.full(len(x), np.nan)
    out[:len(x) - h] = x[h:]
    return out


def ols(Y, X):
    """OLS of Y on X (drop rows with NaN). Returns coefficients, HC standard
    errors, residuals and the regressors actually used."""
    keep = ~np.isnan(np.column_stack([Y, X])).any(axis=1)
    Y, X = Y[keep], X[keep]
    n, k = X.shape
    XXi = np.linalg.inv(X.T @ X)
    b = XXi @ X.T @ Y
    e = Y - X @ b
    V = XXi @ ((X * e[:, None] ** 2).T @ X) @ XXi * n / (n - k)
    return b, np.sqrt(np.diag(V)), e, X


# Plotting: exactly the figures on the slides
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f"
YL = "output, % (100 x log)"


def band(ax, r, color, label, ls="-"):
    """Response r[:, 0] with a 90% band r[:, 0] +/- 1.645 r[:, 1]."""
    ax.fill_between(hs, r[:, 0] - 1.645 * r[:, 1], r[:, 0] + 1.645 * r[:, 1], color=color, alpha=0.13)
    ax.plot(hs, r[:, 0], color=color, label=label, ls=ls)


def finish(ax, title, ylabel=YL, legend=True, loc="best"):
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("months after the shock")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=9, loc=loc)


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# %% [markdown]
# ## The data (slide "Running example")
# Output trends up and interest rates trend down over 1979–2009. Keep that in mind for L1.

# %%
raw = pd.read_excel(DATA, sheet_name="1").replace(123456789, np.nan)
raw = raw[raw.cloyne.notna()].reset_index(drop=True)      # span of the shock
month = raw.month.to_numpy()
y = 100 * raw.gdp.to_numpy()
u = 100 * raw.unempl.to_numpy()
i = 100 * raw.i_1YR.to_numpy()
z = raw.cloyne.to_numpy()
T = len(y)
one = np.ones(T)
print(f"Sample {month[0]} to {month[-1]}, T = {T}")


def lagged(series, p):
    """p lags of each series in the list, side by side (T x len(series)*p)."""
    return np.column_stack([lag(s, l) for l in range(1, p + 1) for s in series]) \
        if p > 0 else np.empty((T, 0))


dates = pd.PeriodIndex(raw.month.str.replace("m", "-"), freq="M").to_timestamp()
fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))
ax[0].plot(dates, y - y[0], color=BLUE); ax[0].set_title("Output (100 x log, 1979m7 = 0)", fontsize=11)
ax[1].plot(dates, i, color=BLUE, label="1-year rate"); ax[1].plot(dates, u, color=RED, label="unemployment")
ax[1].set_title("Rates (%)", fontsize=11); ax[1].legend(frameon=False, fontsize=9)
ax[2].bar(dates, z, width=25, color=GREEN); ax[2].set_title("Narrative shock z (pp)", fontsize=11)
for a in ax:
    a.xaxis.set_major_locator(matplotlib.dates.YearLocator(10))
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
show(fig, "data")


# %% [markdown]
# # Part A. Local projections
# One regression per horizon $h$:
# $$y_{t+h} = \mu_h + \theta_h\, i_t + \gamma_h' r_t + \delta_h' (\text{lags}) + \xi_{h,t+h}$$
# $\theta_h$ is the response of output $h$ months after a 1pp higher rate. The steps below change the controls $r_t$ and the lags, then the identification.
#
# Every LP is a formula, estimated horizon by horizon:
# - OLS (statsmodels): `smf.ols("y_h12 ~ i + y + u + y_l1 + u_l1 + i_l1", data=df)`
# - 2SLS (linearmodels): `IV2SLS.from_formula("y_h12 ~ 1 + y_l1 + u_l1 + i_l1 + [i ~ z]", data=dh)`

# %% [markdown]
# ### One DataFrame with leads and lags
# Left-hand sides are leads, `y_h12` $= y_{t+12}$. Controls are lags, `y_l1` $= y_{t-1}$. `lag_terms(["y", "u"], 2)` writes the controls as formula text.

# %%
df = pd.DataFrame({"y": y, "u": u, "i": i, "z": z}, index=dates)
lags_ = {f"{v}_l{l}": df[v].shift(l) for l in range(1, 25) for v in ["y", "u", "i", "z"]}   # x_{t-l}
leads_ = {f"{v}_h{h}": df[v].shift(-h) for h in hs for v in ["y", "u"]}                     # x_{t+h}
df = pd.concat([df, pd.DataFrame(lags_), pd.DataFrame(leads_)], axis=1)


def lag_terms(variables, p):
    """The controls as formula text: lag_terms(["y", "i"], 2) = 'y_l1 + i_l1 + y_l2 + i_l2'."""
    return " + ".join(f"{v}_l{l}" for l in range(1, p + 1) for v in variables)


print(df[["y", "y_l1", "y_h12", "i", "i_l1"]].head(3).round(2))

# %% [markdown]
# ## L1. No controls
# $$y_{t+h} = \mu_h + \theta_h\, i_t + \xi_{h,t+h}$$
# The loop is written out once here: one OLS per horizon, keep the coefficient on $i$ and its HC1 standard error. With trending output and falling rates, this gives a spurious −6%.

# %%
rows = []
for h in hs:
    res = smf.ols(f"y_h{h} ~ i", data=df).fit(cov_type="HC1")
    rows.append({"h": h, "b": res.params["i"], "se": res.bse["i"]})
step1 = pd.DataFrame(rows)[["b", "se"]].to_numpy()
print(res.summary().tables[1])                         # the last regression (h = 48), in full
print("\nL1 no controls, h = 0/12/24:", step1[[0, 12, 24], 0].round(2))
# Output trends up and rates trend down over 1979-2009: a spurious -6% "effect".

fig, ax = plt.subplots(figsize=(7, 3.4))
band(ax, step1, GREY, "no controls")
ax.set_ylim(-8, 1.5)
finish(ax, "Output after a 1pp higher 1-year rate", loc="lower left")
show(fig, "l1_none")


# The same loop for any right-hand side: we reuse it from here on.
def run_lp(rhs, dep="y", x="i", cov_type="HC1"):
    """OLS of {dep}_h{h} on the formula rhs for h = 0..H. Returns an (H+1) x 2
    array: the coefficient on x and its standard error."""
    out = np.zeros((H + 1, 2))
    for h in hs:
        res = smf.ols(f"{dep}_h{h} ~ {rhs}", data=df).fit(cov_type=cov_type)
        out[h] = res.params[x], res.bse[x]
    return out


# %% [markdown]
# ## L2. Add lags, then date-$t$ controls
# - (a) one lag of $y, u, i$
# - (b) also $y_t$ and $u_t$: the **recursive** assumption, output and unemployment do not react to the rate within the month, so $\theta_0 = 0$
# - (c) twelve lags
#
# `run_lp(rhs)` is the L1 loop as a function.

# %%
step2 = {
    "+ 1 lag of y, u, i":           run_lp("i + " + lag_terms(["y", "u", "i"], 1)),
    "+ y, u at date t (recursive)": run_lp("i + y + u + " + lag_terms(["y", "u", "i"], 1)),
    "recursive, 12 lags":           run_lp("i + y + u + " + lag_terms(["y", "u", "i"], 12)),
}
for name, out in step2.items():
    print(f"L2 {name:28s} h = 0/12/24/36:", out[[0, 12, 24, 36], 0].round(2))
rec12 = step2["recursive, 12 lags"]

lag1, rec1 = step2["+ 1 lag of y, u, i"], step2["+ y, u at date t (recursive)"]
fig, ax = plt.subplots(figsize=(7, 3.4))                        # L2(a): add one lag
band(ax, step1, GREY, "no controls")
band(ax, lag1, RED, "+ 1 lag of y, u, i")
ax.set_ylim(-8, 1.5)
finish(ax, "Output after a 1pp higher 1-year rate", loc="lower left")
show(fig, "l2_lags")

fig, ax = plt.subplots(figsize=(7, 3.4))                        # L2(b, c): date-t controls, then 12 lags
band(ax, lag1, RED, "+ 1 lag of y, u, i")
band(ax, rec1, BLUE, "+ y, u at date t (recursive)")
band(ax, rec12, GREEN, "recursive, 12 lags")
ax.set_ylim(-2.2, 0.8)
finish(ax, "Output after a 1pp higher 1-year rate", loc="lower left")
show(fig, "l2_rec")

# %% [markdown]
# ## L3. The narrative shock as an instrument
# 2SLS of $y_{t+h}$ on $i_t$, instrumenting $i_t$ with $z_t$, controls: 12 lags of $y, u, i, z$. In linearmodels the endogenous regressor and its instrument go in brackets, `[i ~ z]`.
#
# Also: the first-stage F, and whether $z$ is predictable from its controls.

# %%
def run_lpiv(controls, dep="y", **fit_kw):
    """2SLS of {dep}_h{h} on i (instrument z) and the controls, h = 0..H.
    Rows with missing values are dropped first, as linearmodels expects."""
    fit_kw = fit_kw or {"cov_type": "robust"}
    ctrl = [c.strip() for c in controls.split("+") if c.strip()]
    out = np.zeros((H + 1, 2))
    for h in hs:
        dh = df[[f"{dep}_h{h}", "i", "z"] + ctrl].dropna()
        res = IV2SLS.from_formula(f"{dep}_h{h} ~ 1 + {' + '.join(ctrl + ['[i ~ z]'])}", data=dh) \
                    .fit(debiased=True, **fit_kw)
        out[h] = res.params["i"], res.std_errors["i"]
    return out


controls12 = lag_terms(["y", "u", "i", "z"], 12)
iv12 = run_lpiv(controls12)
iv12_u = run_lpiv(controls12, dep="u")
first = smf.ols(f"i ~ z + {controls12}", data=df).fit(cov_type="HC1")     # first stage
print(f"\nL3 first stage: coefficient on z {first.params['z']:.2f}, F = {first.tvalues['z'] ** 2:.1f}")
print("L3 LP-IV output h = 12/24/36:", iv12[[12, 24, 36], 0].round(2),
      " se", iv12[[12, 24, 36], 1].round(2))
print("L3 LP-IV unemployment h = 12/24/36:", iv12_u[[12, 24, 36], 0].round(2))
r2 = smf.ols(f"z ~ {controls12}", data=df).fit().rsquared              # is the shock predictable?
print(f"L3 R2 of z on its controls = {r2:.2f}")

rec12_u = run_lp("i + y + u + " + lag_terms(["y", "u", "i"], 12), dep="u")   # recursive LP, unemployment

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
band(ax[0], rec12, BLUE, "recursive LP, 12 lags"); band(ax[0], iv12, GREEN, "narrative LP-IV, 12 lags")
finish(ax[0], "Output", loc="lower left")
band(ax[1], rec12_u, BLUE, "recursive LP"); band(ax[1], iv12_u, GREEN, "narrative LP-IV")
finish(ax[1], "Unemployment", "pp", legend=False)
show(fig, "l3_iv")

# %% [markdown]
# ## L4. Lags still matter with an instrument
# The same LP-IV without any controls: still consistent if $z$ is a clean shock, but much less precise.

# %%
iv0 = run_lpiv("")                                   # no controls: y_h{h} ~ 1 + [i ~ z]
print("L4 LP-IV no lags h = 12/24/36:", iv0[[12, 24, 36], 0].round(2),
      " se", iv0[[12, 24, 36], 1].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
band(ax, iv0, GREY, "LP-IV, no lags"); band(ax, iv12, GREEN, "LP-IV, 12 lags")
finish(ax, "Output, LP-IV: lags shrink the bands", loc="lower left")
show(fig, "l4_lags_iv")

# %% [markdown]
# ## L5. Which standard errors?
# LP residuals are serially correlated because the horizons overlap. Two fixes:
# - **Newey–West** (HAC): Bartlett kernel with $h + L_0$ lags, `cov_type="kernel"` in linearmodels, `cov_type="HAC"` in statsmodels;
# - **lag augmentation**: enough lags as controls, then plain HC standard errors are valid (Montiel Olea and Plagborg-Møller 2021).

# %%
L0 = int(0.75 * T ** (1 / 3))
nw12 = np.zeros(H + 1)
ctrl = controls12.split(" + ")
for h in hs:
    dh = df[[f"y_h{h}", "i", "z"] + ctrl].dropna()
    res = IV2SLS.from_formula(f"y_h{h} ~ 1 + {controls12} + [i ~ z]", data=dh) \
                .fit(cov_type="kernel", kernel="bartlett", bandwidth=h + L0, debiased=True)
    nw12[h] = res.std_errors["i"]
print("L5 se at h = 12/24/36  HC:", iv12[[12, 24, 36], 1].round(2),
      " Newey-West:", nw12[[12, 24, 36]].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(hs, iv12[:, 1], color=GREEN, label="HC (valid with lag augmentation)")
ax.plot(hs, nw12, color=BLUE, ls="--", label="Newey-West")
ax.set_ylim(0, None)
finish(ax, "Standard error of the LP-IV output response, 12 lags", "standard error", loc="upper left")
show(fig, "l5_se")

# %% [markdown]
# ## L6. LP variance decomposition
# Gorodnichenko and Lee (2020), three regressions:
# 1. the implicit LP shock $\varepsilon_t$: residual of $i_t$ on $y_t$, $u_t$ and 12 lags (L2(c));
# 2. the forecast error $f$ of $y_{t+h}$: residual of $y_{t+h}$ on the 12 lags;
# 3. the share: $R^2$ of $f$ on $\varepsilon_t, \varepsilon_{t+1}, \dots, \varepsilon_{t+h}$.
#
# The bias-corrected line on the slide needs the VAR simulations, so it is computed in C4.

# %%
lags12 = lag_terms(["y", "u", "i"], 12)
df["eps"] = smf.ols(f"i ~ y + u + {lags12}", data=df).fit().resid      # NaN where lags are missing
eps_leads = pd.DataFrame({f"eps_f{s}": df.eps.shift(-s) for s in hs})
fevd_lp = np.zeros(H + 1)
for h in hs:
    E = list(eps_leads.columns[:h + 1])                                  # eps_t .. eps_{t+h}
    dh = pd.concat([df[[f"y_h{h}"] + lags12.split(" + ")], eps_leads[E]], axis=1).dropna()
    dh["f"] = smf.ols(f"y_h{h} ~ {lags12}", data=dh).fit().resid       # forecast error
    fevd_lp[h] = smf.ols("f ~ " + " + ".join(E), data=dh).fit().rsquared
print("L6 FEVD of output (LP R2), h = 12/24/48:", fevd_lp[[11, 23, 47]].round(3))
# The slide also shows the bias-corrected line. The correction needs the VAR
# simulations, so it is computed in C4 and the full L6 figure is drawn there.

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(hs, fevd_lp, color=GREY, lw=2, label="R² method")
ax.set_ylim(0, 1)
finish(ax, "LP (R² method): share of output forecast error variance", "share", loc="upper left")
show(fig, "l6_fevd_raw")


# %% [markdown]
# # Part B. VARs
# $x_t = (y_t, u_t, i_t)'$, reduced form
# $$x_t = c + A_1 x_{t-1} + \dots + A_p x_{t-p} + u_t$$
# estimated by OLS equation by equation (`var_ols`) and iterated forward (`var_irf`).

# %%
names = ["y", "u", "i"]
data = np.column_stack([y, u, i])
k = 3


def var_ols(data, p):
    """Reduced-form VAR(p) by OLS, equation by equation: A_1..A_p, residuals, Sigma, c."""
    n, kk = data.shape
    Y = data[p:]
    X = np.column_stack([np.ones(n - p)] + [data[p - l:n - l] for l in range(1, p + 1)])
    B = np.linalg.inv(X.T @ X) @ X.T @ Y
    U = Y - X @ B
    A = [B[1 + (l - 1) * kk: 1 + l * kk].T for l in range(1, p + 1)]
    return A, U, U.T @ U / (n - p - X.shape[1]), B[0]


def var_irf(A, impact, H):
    """Iterate the VAR forward: Theta_h = A_1 Theta_{h-1} + ... + A_p Theta_{h-p}."""
    irf = np.zeros((H + 1, len(impact)))
    irf[0] = impact
    for h in range(1, H + 1):
        for l in range(1, min(h, len(A)) + 1):
            irf[h] += A[l - 1] @ irf[h - l]
    return irf


# %% [markdown]
# ## V0. Lag length by information criteria
# $$IC(p) = \ln|\hat\Sigma_p| + \text{penalty} \times \frac{p k^2}{T}$$
# with penalty 2 (AIC), $2 \ln\ln T$ (HQ) or $\ln T$ (BIC), every $p$ on the same sample.

# %%
pmax = 24
n_ic = T - pmax
fit = np.zeros(pmax)
for p in range(1, pmax + 1):
    _, Up, _, _ = var_ols(data[pmax - p:], p)
    fit[p - 1] = np.log(np.linalg.det(Up.T @ Up / n_ic))
ps = np.arange(1, pmax + 1)
ic = {"AIC": fit + 2 * ps * k**2 / n_ic,
      "HQ": fit + 2 * np.log(np.log(n_ic)) * ps * k**2 / n_ic,
      "BIC": fit + np.log(n_ic) * ps * k**2 / n_ic}
print("\nV0 lag length chosen:", {nm: int(v.argmin()) + 1 for nm, v in ic.items()})

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(ps, fit, color=GREY, lw=2)
ax[0].set_title(r"Fit only, $\ln|\hat\Sigma_p|$: always falls", fontsize=11)
for (nm, v), col in zip(ic.items(), [RED, GREEN, BLUE]):
    ax[1].plot(ps, v, color=col, lw=2, label=f"{nm}: p = {v.argmin() + 1}")
    ax[1].plot(v.argmin() + 1, v.min(), "o", color=col, ms=7)
ax[1].set_title("Fit + penalty: pick the minimum", fontsize=11)
ax[1].legend(frameon=False, fontsize=9, loc="upper left")
for a in ax:
    a.set_xlabel("lags p")
show(fig, "v0_ic")

# %% [markdown]
# ## Identification: regress one surprise on the other
# Slides "The simple idea", "The same regression in matrix form" and "One restriction, one answer". With two surprises, $u_y$ and $u_i$: regress $u_i$ on $u_y$. The slope is the Bank's reaction to output news, the residual is the policy shock. That is the Cholesky factor with output ordered first ($b = 0$).

# %%
A, U, Sigma, c = var_ols(data, 12)
S2 = Sigma[np.ix_([0, 2], [0, 2])]                 # Var/Cov of (u_y, u_i)
gamma = S2[0, 1] / S2[0, 0]                        # Bank's reaction to an output surprise
sd_e = np.sqrt(S2[1, 1] - S2[0, 1] ** 2 / S2[0, 0])
print("\nSigma (u_y, u_i):\n", S2.round(3))
print(f"gamma = {gamma:.2f}, sd of the policy shock = {sd_e:.2f}pp")
print("b = 0 (output does not react within the month):\n", np.linalg.cholesky(S2).round(2))
Pr = np.linalg.cholesky(S2[::-1, ::-1])[::-1, ::-1]   # c = 0: rate does not react to output
print("c = 0 (rate does not react to output within the month):\n", Pr.round(2))


# %% [markdown]
# ## V1. Recursive VAR
# $u_t = \Theta_0 \varepsilon_t$ with $\Theta_0 = \text{chol}(\Sigma)$, order $(y, u, i)$: output and unemployment do not react to the policy shock within the month. The impact column is the last column of $\text{chol}(\Sigma)$, scaled so that $i$ rises by 1pp. Bands: residual bootstrap.

# %%
P = np.linalg.cholesky(Sigma)
var12 = var_irf(A, P[:, 2] / P[2, 2], H)
print("\nV1 recursive VAR(12), output h = 0/12/24/36:", var12[[0, 12, 24, 36], 0].round(2))
print("   unemployment peak", var12[:, 1].max().round(2), "at h =", var12[:, 1].argmax())

# Residual bootstrap for 90% bands: simulate from the estimated VAR, re-estimate.
# The simulated samples are kept: C4 reuses them for the L6 bias correction.
B = 500
sims = np.zeros((B, T, k))
draws = np.zeros((B, H + 1, k))
for r in range(B):
    Ustar = U[rng.integers(0, len(U), len(U))]
    sim = sims[r]
    sim[:12] = data[:12]
    for t in range(12, T):
        sim[t] = c + sum(A[l - 1] @ sim[t - l] for l in range(1, 13)) + Ustar[t - 12]
    Ab, Ub, Sb, _ = var_ols(sim, 12)
    Pb = np.linalg.cholesky(Sb)
    draws[r] = var_irf(Ab, Pb[:, 2] / Pb[2, 2], H)
var12_lo, var12_hi = np.percentile(draws, [5, 95], axis=0)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for j, (nm, yl) in enumerate([("Output", "%"), ("Unemployment", "pp"), ("1-year rate", "pp")]):
    ax[j].fill_between(hs, var12_lo[:, j], var12_hi[:, j], color=BLUE, alpha=0.15)
    ax[j].plot(hs, var12[:, j], color=BLUE)
    finish(ax[j], nm, yl, legend=False)
show(fig, "v1_svar")

# %% [markdown]
# ## V2. The ordering is the assumption
# Rate first, order $(i, y, u)$: the policy shock is all of $u_i$, and output and unemployment may react within the month.

# %%
order = [2, 0, 1]
P_first = np.linalg.cholesky(Sigma[np.ix_(order, order)])
impact_first = np.zeros(k)
impact_first[order] = P_first[:, 0] / P_first[0, 0]
var12_first = var_irf(A, impact_first, H)
print("V2 rate ordered first, output h = 0/12/24/36:", var12_first[[0, 12, 24, 36], 0].round(2))
print("   corr(u_y, u_i) =", (Sigma[0, 2] / np.sqrt(Sigma[0, 0] * Sigma[2, 2])).round(3))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.fill_between(hs, var12_lo[:, 0], var12_hi[:, 0], color=BLUE, alpha=0.13)
ax.plot(hs, var12[:, 0], color=BLUE, lw=2, label="rate last: (y, u, i)")
ax.plot(hs, var12_first[:, 0], color=RED, lw=2, ls="--", label="rate first: (i, y, u)")
finish(ax, "Output, VAR(12): two orderings", loc="lower left")
show(fig, "v2_order")

# %% [markdown]
# ## V3. The lag length
# The same recursive VAR with 2, 4, 12 and 24 lags.

# %%
var_by_p = {}
for p in [2, 4, 12, 24]:
    Ap, Up, Sp, _ = var_ols(data, p)
    Pp = np.linalg.cholesky(Sp)
    var_by_p[p] = var_irf(Ap, Pp[:, 2] / Pp[2, 2], H)[:, 0]
    print(f"V3 VAR({p:2d}) output h = 12/24/48:", var_by_p[p][[12, 24, 48]].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
for p, col, ls in [(2, GREY, ":"), (4, RED, "--"), (12, BLUE, "-"), (24, GREEN, "-.")]:
    ax.plot(hs, var_by_p[p], color=col, ls=ls, lw=2, label=f"VAR({p})")
finish(ax, "Output, recursive VAR: lag length", loc="lower left")
show(fig, "v3_lags")

# %% [markdown]
# ## Proxy SVAR as 2SLS on the VAR surprises
# First stage: rate surprise $u_i$ on $z$. Second stage: output surprise $u_y$ on the fitted $u_i$. The impact effect on output is
# $$\hat\beta = \frac{\text{Cov}(u_y, z)}{\text{Cov}(u_i, z)}$$

# %%
zt = z[12:] - z[12:].mean()
b, se, _, _ = ols(U[:, 2], np.column_stack([zt, np.ones(len(zt))]))
print(f"\nFirst stage u_i on z: coefficient {b[0]:.2f}, F = {(b[0] / se[0]) ** 2:.0f}")
cov_yz, cov_iz = U[:, 0] @ zt / len(zt), U[:, 2] @ zt / len(zt)
print(f"Impact on output = {cov_yz:.3f} / {cov_iz:.3f} = {cov_yz / cov_iz:.2f}")

# %% [markdown]
# ## V4. The instrument in a VAR: external vs internal
# - **External** (SVAR-IV): regress each VAR residual on $z_t$; the slopes are the impact column up to scale.
# - **Internal**: put $z$ first in the VAR and identify recursively. Does not need invertibility.

# %%
slopes = U.T @ zt / (zt @ zt)
svar_iv = var_irf(A, slopes / slopes[2], H)
# (b) Internal instrument: put z first in the VAR, recursive, normalise on i.
Az, Uz, Sz, _ = var_ols(np.column_stack([z, y, u, i]), 12)
Pz = np.linalg.cholesky(Sz)
svar_int = var_irf(Az, Pz[:, 0] / Pz[3, 0], H)[:, 1:]
print("V4 output h = 12/24/36  external:", svar_iv[[12, 24, 36], 0].round(2),
      " internal (z first):", svar_int[[12, 24, 36], 0].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(hs, var12[:, 0], color=GREY, ls=":", label="recursive (V1)")
ax.plot(hs, svar_iv[:, 0], color=BLUE, lw=2, label="external: SVAR-IV")
ax.plot(hs, svar_int[:, 0], color=GREEN, lw=2, ls="--", label="internal: z ordered first")
finish(ax, "Output, VAR(12): using the narrative shock", loc="lower left")
show(fig, "v4_iv")

# %% [markdown]
# ## V5. VAR variance decomposition
# Share of the $h$-step forecast error variance of output due to the policy shock, from the one-standard-deviation shocks (columns of $\text{chol}(\Sigma)$).

# %%
Theta = np.stack([var_irf(A, P[:, j], H) for j in range(k)], axis=2)   # [h, var, shock]
contrib = np.cumsum(Theta ** 2, axis=0)
fevd_var = contrib[:, 0, 2] / contrib[:, 0, :].sum(axis=1)
print("V5 FEVD of output, h = 12/24/48:", fevd_var[[11, 23, 47]].round(3))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(hs, fevd_var, color=BLUE, lw=2)
ax.set_ylim(0, 1)
finish(ax, "VAR(12): share of output forecast error variance, policy shock", "share", legend=False)
show(fig, "v5_fevd")


# %% [markdown]
# # Part C. LP vs VAR

# %% [markdown]
# ## C1. Same shock, horizon by horizon
# LP with $y_t$, $u_t$ and $p$ lags vs recursive VAR($p$): equal up to $h = p$ in population, and they can diverge after (Plagborg-Møller and Wolf 2021).

# %%
print("\nC1 LP vs VAR, output response, max |difference|:")
lpvar = {}
for p in [1, 2, 4, 8, 12, 24]:
    Ap, Up, Sp, _ = var_ols(data, p)
    Pp = np.linalg.cholesky(Sp)
    v = var_irf(Ap, Pp[:, 2] / Pp[2, 2], H)[:, 0]
    lp_ = run_lp("i + y + u + " + lag_terms(["y", "u", "i"], p))
    lpvar[p] = (lp_, v)
    print(f"  p = {p:2d}: h <= p {np.abs(lp_[:, 0] - v)[:p + 1].max():.2f},"
          f"  h > p {np.abs(lp_[:, 0] - v)[p + 1:].max():.2f}")

for p, (lp_, v) in lpvar.items():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.axvspan(0, p, color="#f1eee6", zorder=0)              # horizons where they must agree
    band(ax, lp_, RED, f"LP, {p} lag{'s' if p > 1 else ''}")
    ax.plot(hs, v, color=BLUE, lw=2, label=f"VAR({p})")
    ax.axvline(p, color="grey", ls="--", lw=0.8)
    ax.set_ylim(-3, 1.5)
    finish(ax, f"p = {p}: LP and VAR agree for h ≤ {p}", loc="lower left")
    show(fig, f"c1_lpvar_p{p}")

# %% [markdown]
# ## C2. One instrument, two estimators
# The internal-instrument VAR (V4) and LP-IV (L3) estimate the same object.

# %%
print("\nC2 output response h = 12/24/36")
print("  VAR, z ordered first :", svar_int[[12, 24, 36], 0].round(2))
print("  LP-IV (L3)           :", iv12[[12, 24, 36], 0].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
band(ax, iv12, GREEN, "LP-IV")
ax.plot(hs, svar_int[:, 0], color=RED, lw=2, ls="--", label="VAR, internal instrument (z first)")
finish(ax, "Output: one instrument, two estimators", loc="lower left")
show(fig, "c2_iv")

# %% [markdown]
# ## C3. Bias versus variance
# Treat the VAR(12) as the truth. Simulate 500 samples of length $T$, estimate the output response with a VAR(2) and an LP with 2 lags, and compare with the truth. The LP here is in matrix form: 500 × 49 regressions would be slow as formulas.

# %%
truth = var12[:, 0]
est_var, est_lp = np.zeros((500, H + 1)), np.zeros((500, H + 1))
for r in range(500):
    sim = np.zeros((T, k))
    sim[:12] = data[:12]
    for t in range(12, T):
        sim[t] = c + sum(A[l - 1] @ sim[t - l] for l in range(1, 13)) + P @ rng.standard_normal(3)
    A2, U2, S2, _ = var_ols(sim, 2)
    P2 = np.linalg.cholesky(S2)
    est_var[r] = var_irf(A2, P2[:, 2] / P2[2, 2], H)[:, 0]
    ys, us, is_ = sim.T            # the LP in matrix form: 500 x 49 regressions, formulas would be slow
    W2 = np.column_stack([one, ys, us] + [lag(s, l) for l in (1, 2) for s in (ys, us, is_)])
    est_lp[r] = [ols(lead(ys, h), np.column_stack([is_, W2]))[0][0] for h in hs]
far = slice(13, H + 1)
for lab, est in [("VAR(2)", est_var), ("LP(2) ", est_lp)]:
    print(f"C3 {lab}: mean |bias| h>12 {np.abs(est.mean(0) - truth)[far].mean():.2f},"
          f" mean sd {est.std(0)[far].mean():.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
ax[0].plot(hs, est_var.mean(0) - truth, color=BLUE, label="VAR(2)")
ax[0].plot(hs, est_lp.mean(0) - truth, color=RED, label="LP, 2 lags")
finish(ax[0], "Bias (truth: VAR(12))", "bias", loc="upper left")
ax[1].plot(hs, est_var.std(0), color=BLUE); ax[1].plot(hs, est_lp.std(0), color=RED)
finish(ax[1], "Standard deviation", "sd", legend=False)
show(fig, "c3_mc")

# %% [markdown]
# ## C4. Variance decompositions side by side (and the L6 bias correction)
# In data simulated from the VAR(12) the true share is the VAR's. The average LP estimate across the V1 bootstrap samples minus that truth is the small-sample bias (Gorodnichenko and Lee 2020). The L6 regressions run 500 times here, so they are written in matrix form.

# %%
def fevd_r2(y, u, i, p=12):
    W = np.column_stack([np.ones(len(y)), y, u, lagged([y, u, i], p)])
    _, _, e_i, _ = ols(i, W)
    eps = np.full(len(y), np.nan)
    eps[p:] = e_i                                              # implicit shock
    Wlag = np.column_stack([np.ones(len(y)), lagged([y, u, i], p)])
    out = np.zeros(H + 1)
    for h in hs:
        E = np.column_stack([lead(eps, s) for s in range(h + 1)])     # eps_t..eps_{t+h}
        keep = ~np.isnan(np.column_stack([lead(y, h), Wlag, E])).any(axis=1)
        _, _, f, _ = ols(lead(y, h)[keep], Wlag[keep])               # forecast error
        _, _, e, _ = ols(f, np.column_stack([np.ones(keep.sum()), E[keep]]))
        out[h] = 1 - e.var() / f.var()
    return out


fevd_sim = np.array([fevd_r2(*s.T) for s in sims])
fevd_bias = fevd_sim.mean(0) - fevd_var
fevd_lp_bc = np.clip(fevd_lp - fevd_bias, 0, 1)
print("\nC4 FEVD of output at h = 12/24/48")
print("  VAR (V5)            :", fevd_var[[11, 23, 47]].round(3))
print("  LP (L6)             :", fevd_lp[[11, 23, 47]].round(3))
print("  LP, bias-corrected  :", fevd_lp_bc[[11, 23, 47]].round(3))

fig, ax = plt.subplots(figsize=(7, 3.2))                        # the L6 slide
ax.plot(hs, fevd_lp, color=GREY, lw=2, label="R² method")
ax.plot(hs, fevd_lp_bc, color=RED, lw=2, label="bias-corrected")
ax.set_ylim(0, 1)
finish(ax, "LP (R² method): share of output forecast error variance", "share", loc="upper left")
show(fig, "l6_fevd")

fig, ax = plt.subplots(figsize=(7, 3.2))                        # the C4 slide
ax.plot(hs, fevd_var, color=BLUE, lw=2, label="VAR(12)")
ax.plot(hs, fevd_lp_bc, color=RED, ls="--", label="LP, R² method (bias-corrected)")
ax.set_ylim(0, 1)
finish(ax, "Share of output forecast error variance due to the policy shock", "share", loc="upper left")
show(fig, "c4_fevd")

# %% [markdown]
# ## C5 (take-home). Non-invertibility
# $y_t = \eta_t + 2\eta_{t-1}$. A long autoregression cannot recover $\eta_t$ from current and past $y$, so it finds the "invertible twin" with response 1/2. An LP on the observed $\eta_t$ gets it right.

# %%
eta = rng.standard_normal(100_000)
ym = eta + 2 * np.r_[0, eta[:-1]]
Xar = np.column_stack([np.ones(len(ym))] + [lag(ym, l) for l in range(1, 13)])
b, _, e, _ = ols(ym, Xar)
ar = np.zeros(4); ar[0] = 1                    # AR(12) response to its own innovation
for h in range(1, 4):
    ar[h] = sum(b[l] * ar[h - l] for l in range(1, h + 1))
lp_eta = [ols(lead(ym, h), np.column_stack([eta, np.ones(len(ym))]))[0][0] for h in range(4)]
print("\nC5 MA(1) with theta = 2, h = 0..3: truth [1, 2, 0, 0]")
print("  AR(12)      :", ar.round(2))
print("  LP on eta_t :", np.round(lp_eta, 2))
