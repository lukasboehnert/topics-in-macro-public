# %% [markdown]
# # Week 1 lab: local projections and VARs, the fast version
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
# Same cells, same specifications and same numbers as `week1_simple.py`, and the figures are the ones in the deck (saved to `figures/`):
#
# | Part | Cells |
# | --- | --- |
# | A. Local projections | L1 no controls, L2 lags and date-$t$ controls, L3 LP-IV, L4 lags with an instrument, L5 standard errors, L6 variance decomposition |
# | B. VARs | V0 lag length, identification by regression, V1 recursive VAR, V2 ordering, V3 lag length, proxy SVAR as 2SLS, V4 instrument, V5 variance decomposition |
# | C. LP vs VAR | C1 horizon by horizon, C2 one instrument two estimators, C3 bias vs variance, C4 variance decompositions |
# | Take-home | C5 non-invertibility |
#
# **How the code is written.** The same DataFrame of lags as in `week1_simple.py`, and every LP still names its variables like a formula,
#
# `lp("y", "i", controls=["y", "u"] + lag_terms(["y", "u", "i"], 12))`
#
# but four small numpy functions do all the work, so the whole lecture runs in seconds instead of a minute:
#
# - `lp`: one function for every LP (OLS or IV, HC or Newey–West). The regressors are built once; only the left-hand side moves with the horizon.
# - `var`: OLS VAR for one sample, or for 500 simulated samples at once.
# - `irf`: impulse responses by iterating the companion form, again for one VAR or a stack of them.
# - `simulate`: all bootstrap / Monte Carlo samples are simulated together, one time step at a time.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run week1_fast.py`.


# %% [markdown]
# ## Setup
# Packages, horizons ($h = 0, \dots, 48$ months), the four workhorse functions and the plotting functions that draw the slide figures.

# %%
from pathlib import Path

import matplotlib
import matplotlib.dates
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
DATA = HERE / "data" / "CTV_DATA.xlsx"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)
H = 48                       # horizons 0..H (months)
hs = np.arange(H + 1)
rng = np.random.default_rng(1)


def resid(Y, X):
    """Residuals of Y on a constant and X, for one regression (n x k) or a stack of
    them (R x n x k) at once. Demeaning takes care of the constant."""
    Yc, Xc = Y - Y.mean(-2, keepdims=True), X - X.mean(-2, keepdims=True)
    Xt = np.swapaxes(Xc, -1, -2)
    return Yc - Xc @ np.linalg.solve(Xt @ Xc, Xt @ Yc)


def lp(dep, x, controls=(), iv=None, se="HC", data=None, H=H):
    """LP of {dep}_{t+h} on x_t and the controls (constant added), h = 0..H.
    OLS, or 2SLS with instrument iv for x. se: "HC" (HC1) or "NW" (Newey-West,
    Bartlett kernel with h + L0 lags). Returns an (H+1) x 2 array: the
    coefficient on x and its standard error."""
    d = df if data is None else data
    yv = d[dep].to_numpy()
    T = len(yv)
    X = np.column_stack([d[x], np.ones(T), d[list(controls)]])          # built once for all h
    Z = X if iv is None else np.column_stack([d[iv], X[:, 1:]])
    L0 = int(0.75 * T ** (1 / 3))
    ok = ~np.isnan(np.column_stack([X, Z])).any(1)
    out = np.empty((H + 1, 2))
    for h in range(H + 1):
        r = ok.copy()
        r[T - h:] = False                                                # y_{t+h} must exist
        Yh, Xr, Zr = yv[np.flatnonzero(r) + h], X[r], Z[r]
        Xh = Zr @ np.linalg.lstsq(Zr, Xr, rcond=None)[0]                # first stage (= Xr for OLS)
        b = np.linalg.lstsq(Xh, Yh, rcond=None)[0]
        S = Xh * (Yh - Xr @ b)[:, None]
        M = S.T @ S
        if se == "NW":
            for l in range(1, h + L0 + 1):
                G = S[l:].T @ S[:-l]
                M += (1 - l / (h + L0 + 1)) * (G + G.T)
        Ai = np.linalg.inv(Xh.T @ Xh)
        n, k = Xr.shape
        out[h] = b[0], np.sqrt((Ai @ M @ Ai)[0, 0] * n / (n - k))
    return out


def var(Y, p):
    """Reduced-form VAR(p) by OLS for one sample (T x k) or many at once (R x T x k).
    Returns the constant c, the companion matrix F, the residuals U and Sigma."""
    T, k = Y.shape[-2:]
    X = np.concatenate([np.ones(Y.shape[:-2] + (T - p, 1))]
                       + [Y[..., p - l:T - l, :] for l in range(1, p + 1)], axis=-1)
    Q, R = np.linalg.qr(X)
    B = np.linalg.solve(R, np.swapaxes(Q, -1, -2) @ Y[..., p:, :])
    U = Y[..., p:, :] - X @ B
    F = np.zeros(Y.shape[:-2] + (k * p, k * p))
    F[..., :k, :] = np.swapaxes(B[..., 1:, :], -1, -2)                  # [A_1 ... A_p]
    F[..., k:, :-k] = np.eye(k * (p - 1))
    return B[..., 0, :], F, U, np.swapaxes(U, -1, -2) @ U / (T - p - X.shape[-1])


def irf(F, impact, H=H):
    """Responses at h = 0..H to the impact vector (k) or matrix (k x m), by iterating
    the companion form. F and impact may carry a leading stack dimension (R draws)."""
    vector = impact.ndim == F.ndim - 1
    imp = impact[..., None] if vector else impact
    k = imp.shape[-2]
    J = np.zeros(F.shape[:-1] + imp.shape[-1:])
    J[..., :k, :] = imp
    out = []
    for _ in range(H + 1):
        out.append(J[..., :k, :])
        J = F @ J
    out = np.stack(out, axis=-3)                                         # [..., h, var, shock]
    return out[..., 0] if vector else out


def chol_impact(S, j):
    """Recursive identification: column j of chol(S), scaled so variable j moves by 1."""
    P = np.linalg.cholesky(S)
    return P[..., :, j] / P[..., j, j, None]


def simulate(c, F, Y0, shocks):
    """Simulate a VAR for all draws at once from start values Y0 (p x k). shocks: R x T x k."""
    R, T, k = shocks.shape
    state = np.tile(Y0[::-1].reshape(-1), (R, 1))                       # R x kp, newest first
    out = np.empty((R, T, k))
    for t in range(T):
        out[:, t] = c + state @ F[:k].T + shocks[:, t]
        state = np.column_stack([out[:, t], state[:, :-k]])
    return out


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


def show(fig, name=None):
    """Save the figure to figures/{name}.png (the deck's file) and show it; name=None only shows."""
    fig.tight_layout()
    if name:
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
print(f"Sample {month[0]} to {month[-1]}, T = {T}")

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
# ### One DataFrame with lags
# Controls are lags, `y_l1` $= y_{t-1}$. `lag_terms(["y", "u"], 2)` lists them by name. The leads $y_{t+h}$ are taken inside `lp`.

# %%
df = pd.DataFrame({"y": y, "u": u, "i": i, "z": z}, index=dates)
lags_ = {f"{v}_l{l}": df[v].shift(l) for l in range(1, 25) for v in ["y", "u", "i", "z"]}   # x_{t-l}
df = pd.concat([df, pd.DataFrame(lags_)], axis=1)


def lag_terms(variables, p):
    """The lagged controls by name: lag_terms(["y", "i"], 2) = ['y_l1', 'i_l1', 'y_l2', 'i_l2']."""
    return [f"{v}_l{l}" for l in range(1, p + 1) for v in variables]


print(df[["y", "y_l1", "i", "i_l1"]].head(3).round(2))

# %% [markdown]
# ## L1. No controls
# $$y_{t+h} = \mu_h + \theta_h\, i_t + \xi_{h,t+h}$$
# With trending output and falling rates, this gives a spurious −6%.

# %%
step1 = lp("y", "i")
print("L1 no controls, h = 0/12/24:", step1[[0, 12, 24], 0].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
band(ax, step1, GREY, "no controls")
ax.set_ylim(-8, 1.5)
finish(ax, "Output after a 1pp higher 1-year rate", loc="lower left")
show(fig, "l1_none")

# %% [markdown]
# ## L2. Add lags, then date-$t$ controls
# - (a) one lag of $y, u, i$
# - (b) also $y_t$ and $u_t$: the **recursive** assumption, output and unemployment do not react to the rate within the month, so $\theta_0 = 0$
# - (c) twelve lags

# %%
step2 = {
    "+ 1 lag of y, u, i":           lp("y", "i", lag_terms(["y", "u", "i"], 1)),
    "+ y, u at date t (recursive)": lp("y", "i", ["y", "u"] + lag_terms(["y", "u", "i"], 1)),
    "recursive, 12 lags":           lp("y", "i", ["y", "u"] + lag_terms(["y", "u", "i"], 12)),
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
# 2SLS of $y_{t+h}$ on $i_t$, instrumenting $i_t$ with $z_t$, controls: 12 lags of $y, u, i, z$: `lp(..., iv="z")`.
#
# Also: the first-stage F, and whether $z$ is predictable from its controls.

# %%
controls12 = lag_terms(["y", "u", "i", "z"], 12)
iv12 = lp("y", "i", controls12, iv="z")
iv12_u = lp("u", "i", controls12, iv="z")
first = lp("i", "z", controls12, H=0)[0]                                 # first stage
print(f"\nL3 first stage: coefficient on z {first[0]:.2f}, F = {(first[0] / first[1]) ** 2:.1f}")
print("L3 LP-IV output h = 12/24/36:", iv12[[12, 24, 36], 0].round(2),
      " se", iv12[[12, 24, 36], 1].round(2))
print("L3 LP-IV unemployment h = 12/24/36:", iv12_u[[12, 24, 36], 0].round(2))
dz = df[["z"] + controls12].dropna()                                     # is the shock predictable?
e = resid(dz[["z"]].to_numpy(), dz[controls12].to_numpy())
print(f"L3 R2 of z on its controls = {1 - e.var() / dz.z.var(ddof=0):.2f}")

rec12_u = lp("u", "i", ["y", "u"] + lag_terms(["y", "u", "i"], 12))      # recursive LP, unemployment

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
iv0 = lp("y", "i", iv="z")
print("L4 LP-IV no lags h = 12/24/36:", iv0[[12, 24, 36], 0].round(2),
      " se", iv0[[12, 24, 36], 1].round(2))

fig, ax = plt.subplots(figsize=(7, 3.4))
band(ax, iv0, GREY, "LP-IV, no lags"); band(ax, iv12, GREEN, "LP-IV, 12 lags")
finish(ax, "Output, LP-IV: lags shrink the bands", loc="lower left")
show(fig, "l4_lags_iv")

# %% [markdown]
# ## L5. Which standard errors?
# LP residuals are serially correlated because the horizons overlap. Two fixes:
# - **Newey–West** (HAC): Bartlett kernel with $h + L_0$ lags, `se="NW"`;
# - **lag augmentation**: enough lags as controls, then plain HC standard errors are valid (Montiel Olea and Plagborg-Møller 2021).

# %%
nw12 = lp("y", "i", controls12, iv="z", se="NW")[:, 1]
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
# `fevd_r2` takes a stack of samples (R x T x 3), so C4 runs it on all 500 bootstrap samples in one call. The bias-corrected line on the slide needs those samples, so it is computed in C4.

# %%
def fevd_r2(S, p=12):
    """LP variance decomposition (R^2 method) of output for samples S = (y, u, i), R x T x 3."""
    R, T, _ = S.shape
    lagsS = np.concatenate([S[:, p - l:T - l] for l in range(1, p + 1)], axis=2)   # t = p..T-1
    eps = resid(S[:, p:, 2:], np.concatenate([S[:, p:, :2], lagsS], axis=2))[..., 0]
    out = np.empty(H + 1)
    for h in hs:
        n = T - p - h                                                    # rows with y_{t+h} and eps_{t+h}
        f = resid(S[:, p + h:, :1], lagsS[:, :n])                         # forecast error
        E = np.swapaxes(np.lib.stride_tricks.sliding_window_view(eps, n, axis=1), 1, 2)  # eps_t..eps_{t+h}
        e = resid(f, E)
        out[h] = (1 - e.var(axis=1) / f.var(axis=1)).mean()               # average over samples
    return out


data = np.column_stack([y, u, i])
fevd_lp = fevd_r2(data[None])
print("L6 FEVD of output (LP R2), h = 12/24/48:", fevd_lp[[11, 23, 47]].round(3))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(hs, fevd_lp, color=GREY, lw=2, label="R² method")
ax.set_ylim(0, 1)
finish(ax, "LP (R² method): share of output forecast error variance", "share", loc="upper left")
show(fig)                                                                # the slide's version is drawn in C4


# %% [markdown]
# # Part B. VARs
# $x_t = (y_t, u_t, i_t)'$, reduced form
# $$x_t = c + A_1 x_{t-1} + \dots + A_p x_{t-p} + u_t$$
# estimated by OLS (`var`) and iterated forward in companion form (`irf`).

# %%
names = ["y", "u", "i"]
k = 3

# %% [markdown]
# ## V0. Lag length by information criteria
# $$IC(p) = \ln|\hat\Sigma_p| + \text{penalty} \times \frac{p k^2}{T}$$
# with penalty 2 (AIC), $2 \ln\ln T$ (HQ) or $\ln T$ (BIC), every $p$ on the same sample.

# %%
pmax = 24
n_ic = T - pmax
ps = np.arange(1, pmax + 1)
fit = np.array([np.linalg.slogdet((lambda U: U.T @ U / n_ic)(var(data[pmax - p:], p)[2]))[1] for p in ps])
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
c, F, U, Sigma = var(data, 12)
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
# $u_t = \Theta_0 \varepsilon_t$ with $\Theta_0 = \text{chol}(\Sigma)$, order $(y, u, i)$: output and unemployment do not react to the policy shock within the month. The impact column is the last column of $\text{chol}(\Sigma)$, scaled so that $i$ rises by 1pp. Bands: residual bootstrap, all 500 samples simulated and re-estimated at once.

# %%
P = np.linalg.cholesky(Sigma)
var12 = irf(F, chol_impact(Sigma, 2))
print("\nV1 recursive VAR(12), output h = 0/12/24/36:", var12[[0, 12, 24, 36], 0].round(2))
print("   unemployment peak", var12[:, 1].max().round(2), "at h =", var12[:, 1].argmax())

# Residual bootstrap for 90% bands: simulate from the estimated VAR, re-estimate.
# The simulated samples are kept: C4 reuses them for the L6 bias correction.
B = 500
Ustar = U[rng.integers(0, len(U), (B, len(U)))]
sims = np.concatenate([np.tile(data[:12], (B, 1, 1)), simulate(c, F, data[:12], Ustar)], axis=1)
_, Fb, _, Sb = var(sims, 12)
draws = irf(Fb, chol_impact(Sb, 2))
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
impact_first = np.zeros(k)
impact_first[order] = chol_impact(Sigma[np.ix_(order, order)], 0)
var12_first = irf(F, impact_first)
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
    _, Fp, _, Sp = var(data, p)
    var_by_p[p] = irf(Fp, chol_impact(Sp, 2))[:, 0]
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
surprises = pd.DataFrame({"u_y": U[:, 0], "u_u": U[:, 1], "u_i": U[:, 2], "z": zt})
fs = lp("u_i", "z", data=surprises, H=0)[0]
print(f"\nFirst stage u_i on z: coefficient {fs[0]:.2f}, F = {(fs[0] / fs[1]) ** 2:.0f}")
cov_yz, cov_iz = U[:, 0] @ zt / len(zt), U[:, 2] @ zt / len(zt)
print(f"Impact on output = {cov_yz:.3f} / {cov_iz:.3f} = {cov_yz / cov_iz:.2f}")

# %% [markdown]
# ## V4. The instrument in a VAR: external vs internal
# - **External** (SVAR-IV): regress each VAR residual on $z_t$; the slopes are the impact column up to scale.
# - **Internal**: put $z$ first in the VAR and identify recursively. Does not need invertibility.

# %%
slopes = U.T @ zt / (zt @ zt)
svar_iv = irf(F, slopes / slopes[2])
_, Fz, _, Sz = var(np.column_stack([z, data]), 12)                       # z first, normalise on i
Pz = np.linalg.cholesky(Sz)
svar_int = irf(Fz, Pz[:, 0] / Pz[3, 0])[:, 1:]
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
Theta = irf(F, P)                                                         # [h, var, shock]
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
    _, Fp, _, Sp = var(data, p)
    v = irf(Fp, chol_impact(Sp, 2))[:, 0]
    lp_ = lp("y", "i", ["y", "u"] + lag_terms(["y", "u", "i"], p))
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
# Treat the VAR(12) as the truth. Simulate 500 samples of length $T$, estimate the output response with a VAR(2) and an LP with 2 lags, and compare with the truth. All 500 samples are estimated at once: one stacked VAR(2), and per horizon one stacked LP regression.

# %%
R = 500
truth = var12[:, 0]
sim = simulate(c, F, data[:12], rng.standard_normal((R, T - 12, 3)) @ P.T)   # R x (T-12) x 3
_, F2, _, S2s = var(sim, 2)
est_var = irf(F2, chol_impact(S2s, 2))[..., 0]
Ts = sim.shape[1]
X2 = np.concatenate([np.ones((R, Ts - 2, 1)), sim[:, 2:, 2:], sim[:, 2:, :2],      # 1, i_t, y_t, u_t,
                     sim[:, 1:-1], sim[:, :-2]], axis=2)                          # 2 lags of y, u, i
est_lp = np.empty((R, H + 1))
for h in hs:
    Q, Rr = np.linalg.qr(X2[:, :Ts - 2 - h])
    est_lp[:, h] = np.linalg.solve(Rr, np.swapaxes(Q, 1, 2) @ sim[:, 2 + h:, :1])[:, 1, 0]
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
# In data simulated from the VAR(12) the true share is the VAR's. The average LP estimate across the V1 bootstrap samples minus that truth is the small-sample bias (Gorodnichenko and Lee 2020). `fevd_r2` runs on all 500 samples in one call.

# %%
fevd_bias = fevd_r2(sims) - fevd_var
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
ma = pd.DataFrame({"eta": eta, "y": eta + 2 * np.r_[0, eta[:-1]]})
_, Far, _, _ = var(ma[["y"]].to_numpy(), 12)
ar = irf(Far, np.ones(1), H=3)[:, 0]                 # AR(12) response to its own innovation
lp_eta = lp("y", "eta", data=ma, H=3)[:, 0]
print("\nC5 MA(1) with theta = 2, h = 0..3: truth [1, 2, 0, 0]")
print("  AR(12)      :", ar.round(2))
print("  LP on eta_t :", lp_eta.round(2))
