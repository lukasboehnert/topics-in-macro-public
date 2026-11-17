# %% [markdown]
# # Week 4 lab: one monetary policy for countries that differ
#
# **Question.** What does a country give up when it joins a currency union, how should one central bank set
# policy for members that differ, and why does the same interest-rate change move output less in the more
# unequal member countries?
#
# | Part | Cells |
# | --- | --- |
# | A. Monetary policy in a union (Benigno 2004) | B0 the two-region model, B1 an asymmetric shock, B2 which inflation to target |
# | B. Unequal countries in one union (Boehnert, de Ferra, Mitman and Romei 2025) | N0 services are a luxury, N1 a rate rise in a union of identical countries, N2 an equal and an unequal country in one union |
#
# **How the code is written.** In Part A every linear equation is one line: a dictionary of coefficients on next period's
# expected variables (left) and on today's variables (right), $A\, E_t w_{t+1} = B\, w_t$, solved with our own
# QZ solver (Klein 2000). Part B is a non-linear two-period model, solved with `scipy.optimize.root`. `week4_fast.py` runs the same cells without the explanations and writes the slide
# figures to `figures/`; the numbers are identical.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run week4_simple.py` (a few seconds).

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.linalg import ordqz, solve_discrete_lyapunov
from scipy.optimize import brentq, root

try:                                       # VS Code cells / Jupyter: figures appear under each cell
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:                          # plain script: only save the figures
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures_simple"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# %% [markdown]
# ## The solver: a linear rational-expectations solver (Klein 2000)
# Stack the variables in $w_t = (k_t, d_t)$: $k_t$ **predetermined** (known at $t$: the shocks and lagged
# variables), $d_t$ **jumps** (inflation, output, prices...). Every equation is a row of
# $$A\, E_t w_{t+1} = B\, w_t.$$
# The QZ (generalised Schur) decomposition $B = Q S Z'$, $A = Q T Z'$ gives the generalised eigenvalues
# $S_{ii}/T_{ii}$. **Blanchard-Kahn:** a unique stable solution needs exactly as many stable eigenvalues as
# predetermined variables. Then, with the stable ones ordered first,
# $$d_t = F k_t, \quad F = Z_{21} Z_{11}^{-1}; \qquad k_{t+1} = P k_t, \quad P = Z_{11} T_{11}^{-1} S_{11} Z_{11}^{-1}.$$
# Price *levels* under inflation targeting have a unit root; we count roots up to $1 + 10^{-8}$ as stable.

# %%
def solve(A, B, nk):
    """Klein (2000): returns F (d = F k), P (k' = P k) and the generalised eigenvalues."""
    def stable(al, be):
        return np.abs(al) < (1 + 1e-8) * np.abs(be)
    S, T, al, be, Q, Z = ordqz(B, A, sort=stable, output="complex")
    eig = al / np.where(be == 0, 1e-300, be)
    n_stable = int(np.sum(np.abs(eig) < 1 + 1e-8))
    if n_stable != nk:
        raise ValueError(f"Blanchard-Kahn fails: {n_stable} stable roots, {nk} predetermined variables")
    Z11, Z21 = Z[:nk, :nk], Z[nk:, :nk]
    F = (Z21 @ np.linalg.inv(Z11)).real
    P = (Z11 @ np.linalg.solve(T[:nk, :nk], S[:nk, :nk]) @ np.linalg.inv(Z11)).real
    return F, P, eig


# %% [markdown]
# ## B0. A two-region currency union (Benigno 2004)
# Two regions, H (size $n$) and F (size $1-n$), share one currency and one central bank. Complete financial
# markets, so consumption is the same everywhere; no labour mobility; no fiscal policy. Each region produces its
# own good with labour, and firms reset prices with probability $1-\alpha_j$ a quarter (Calvo).
#
# Log utility, inverse Frisch elasticity $\eta$, elasticity $\theta$ between the two regions' goods and
# $\gamma$ between varieties. The **terms of trade** $T_t = p_{F,t} - p_{H,t}$ move relative demand,
# $y_{H,t} - y_{F,t} = \theta T_t$. With flexible prices, a relative productivity gain
# $a_{R,t} = a_{H,t} - a_{F,t}$ lowers the relative price of H goods:
# $$\tilde T_t = \frac{1+\eta}{1+\eta\theta}\, a_{R,t}.$$
# Where this comes from: the disutility of work gives labour supply $w - p = c + \eta l$, so a firm's real
# marginal cost, $mc_H = w_H - p_H - a_H$, rises with output. With $l_H = y_H - a_H$, $c = y_W$ and demand,
# $$mc_{H} = (1+\eta)y_{W} + (1-n)(1+\eta\theta)T - (1+\eta)a_{H}, \qquad
#   mc_{F} = (1+\eta)y_{W} - n(1+\eta\theta)T - (1+\eta)a_{F}.$$
# Flexible prices mean $mc_H = mc_F = 0$: natural union output is $n a_H + (1-n) a_F$ (symmetric shocks) and the
# natural terms of trade are $\tilde T$ above (asymmetric shocks). With Calvo pricing,
# $\pi_j = \beta E\pi_j' + \lambda_j\, mc_j$.
#
# In gaps (union output gap $y_{W,t}$, terms-of-trade gap $\tau_t = T_t - \tilde T_t$), the model is two Phillips
# curves and an identity:
# $$\pi_{H,t} = \beta E_t\pi_{H,t+1} + \kappa_H\left[y_{W,t} + (1-n)\psi\tau_t\right], \qquad
#   \pi_{F,t} = \beta E_t\pi_{F,t+1} + \kappa_F\left[y_{W,t} - n\psi\tau_t\right],$$
# $$\tau_t = \tau_{t-1} + \pi_{F,t} - \pi_{H,t} - \Delta\tilde T_t,$$
# with $\kappa_j = (1+\eta)(1-\alpha_j)(1-\beta\alpha_j)/\alpha_j$ and $\psi = (1+\eta\theta)/(1+\eta)$.
# **In a union the exchange rate is fixed**: the terms of trade can only move through the two inflation rates.
# With two currencies, the identity has $+\Delta e_t$ on the right, and the exchange rate can close $\tau_t$ at once.
#
# **Welfare** (second-order approximation, efficient steady state), per period, in % of consumption:
# $$L = \frac{1+\eta}{2}\left[y_{W}^2 + n(1-n)\theta\psi\,\tau^2 + n\frac{\gamma}{\kappa_H}\pi_H^2
#   + (1-n)\frac{\gamma}{\kappa_F}\pi_F^2\right].$$
# **Policy.** The central bank targets a weighted average of the two inflation rates,
# $\delta\,\pi_{H,t} + (1-\delta)\,\pi_{F,t} = 0$; $\delta = n$ is union-wide inflation.
#
# Calibration (quarterly): $\beta = 0.99$, $\eta = 3$, $\theta = 1$, $\gamma = 6$, $n = 0.5$, $\alpha_H = \alpha_F = 0.75$;
# relative productivity AR(1) with persistence 0.9 and s.d. 1%.

# %%
beta, eta, theta, gam, n = 0.99, 3.0, 1.0, 6.0, 0.5
rho_a, sd_a = 0.9, 0.01
psi = (1 + eta * theta) / (1 + eta)
c_nat = (1 + eta) / (1 + eta * theta)          # natural terms of trade per unit of relative productivity


def kappa(alpha_j):
    return (1 + eta) * (1 - alpha_j) * (1 - beta * alpha_j) / alpha_j


def union_model(delta, aH=0.75, aF=0.75, n=n):
    kH, kF = kappa(aH), kappa(aF)
    K = ["aR", "aR_lag", "tau_lag"]                       # predetermined
    D = ["piH", "piF", "yW", "tau"]                       # jumps
    V = K + D
    eqs = [({"aR": 1}, {"aR": rho_a}),
           ({"aR_lag": 1}, {"aR": 1}),
           ({"tau_lag": 1}, {"tau": 1}),
           ({"piH": beta}, {"piH": 1, "yW": -kH, "tau": -kH * (1 - n) * psi}),     # Phillips curve, H
           ({"piF": beta}, {"piF": 1, "yW": -kF, "tau": kF * n * psi}),           # Phillips curve, F
           ({}, {"tau": 1, "tau_lag": -1, "piF": -1, "piH": 1,                     # terms-of-trade gap, e fixed
                 "aR": c_nat, "aR_lag": -c_nat}),
           ({}, {"piH": delta, "piF": 1 - delta})]                                  # central bank's target
    A, B = np.zeros((len(eqs), len(V))), np.zeros((len(eqs), len(V)))
    for r, (nxt, now) in enumerate(eqs):
        for v, c in nxt.items():
            A[r, V.index(v)] = c
        for v, c in now.items():
            B[r, V.index(v)] = c
    F, P, eig = solve(A, B, len(K))
    return F, P, K, D, (kH, kF)


def union_irf(delta, H=20, **kw):
    F, P, K, D, _ = union_model(delta, **kw)
    k = np.zeros(len(K))
    k[0] = sd_a
    rows = []
    for h in range(H):
        rows.append(np.r_[k, F @ k])
        k = P @ k
    out = pd.DataFrame(100 * np.array(rows), columns=K + D)
    out["T_nat"] = c_nat * out.aR
    out["T"] = out.tau + out.T_nat
    out[["piH", "piF"]] *= 4                                       # annualised
    return out


def union_loss(delta, aH=0.75, aF=0.75, n=n):
    F, P, K, D, (kH, kF) = union_model(delta, aH=aH, aF=aF, n=n)
    Sk = np.zeros((len(K), len(K)))
    Sk[0, 0] = sd_a**2
    Vk = solve_discrete_lyapunov(P, Sk)                            # variance of the states
    v = dict(zip(D, np.diag(F @ Vk @ F.T)))                        # variance of the jumps
    terms = {"union gap": v["yW"], "terms of trade": n * (1 - n) * theta * psi * v["tau"],
             "inflation H": n * gam / kH * v["piH"], "inflation F": (1 - n) * gam / kF * v["piF"]}
    return {k_: 100 * (1 + eta) / 2 * x for k_, x in terms.items()}


print(f"kappa = {kappa(0.75):.3f}, psi = {psi:.2f}, natural terms of trade = {c_nat:.2f} x relative productivity")

# %% [markdown]
# ## B1. An asymmetric shock: what the union cannot do
# Region H becomes 1% more productive than F. With two currencies, H's currency depreciates and the terms of
# trade jump to their natural level at once: no gap, no inflation, **no welfare loss**. In the union, the terms of
# trade can move only if H prices fall and F prices rise, slowly. The central bank keeps union inflation at zero
# ($\delta = n$), so the union output gap is zero, but it cannot do anything about the relative block: with equal
# stickiness the relative Phillips curve $\pi_{R,t} = \beta E_t\pi_{R,t+1} - \kappa\psi\tau_t$
# ($\pi_R = \pi_F - \pi_H$) and the identity do not involve monetary policy at all.
# Together they give $\tau_t = \mu\tau_{t-1} + $ shocks, where $\mu$ solves $\beta\mu^2 - (1+\beta+\kappa\psi)\mu + 1 = 0$:
# $\mu = 0.56$ here, whatever the central bank does.

# %%
irf = union_irf(n)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(irf.T_nat, color=GREY, ls="--", label="two currencies (natural)")
ax[0].plot(irf["T"], color=BLUE, label="currency union")
ax[0].set(title="terms of trade $p_F - p_H$, %")
ax[0].legend(frameon=False, fontsize=9)
ax[1].plot(irf.piH, color=BLUE, label="$\\pi_H$")
ax[1].plot(irf.piF, color=RED, label="$\\pi_F$")
ax[1].plot(n * irf.piH + (1 - n) * irf.piF, color=GREY, ls=":", label="union inflation")
ax[1].set(title="inflation in the union (annualised, %)")
ax[1].legend(frameon=False, fontsize=9)
ax[2].plot(irf.tau, color=BLUE)
ax[2].set(title="terms-of-trade gap $\\tau$, %")
for a_ in ax:
    a_.axhline(0, color=GREY, lw=0.6)
    a_.set(xlabel="quarters")
show(fig, "b1_asym")
L_sym = union_loss(n)
print(f"Relative productivity +1%: H inflation on impact {irf.piH[0]:.2f}%, F inflation {irf.piF[0]:.2f}% (annualised); "
      f"terms-of-trade gap {irf.tau[0]:.2f}%. Half-life of the gap: "
      f"{int(np.argmax(irf.tau.values > irf.tau[0] / 2))} quarters")
print("Welfare loss in the union, % of consumption per period:", {k_: round(float(x), 4) for k_, x in L_sym.items()},
      f"total {sum(L_sym.values()):.4f}; with two currencies: 0")

# %% [markdown]
# ## B2. Which inflation should the union target?
# Now H's prices are stickier ($\alpha_H > \alpha_F$). The relative block is no longer independent of policy:
# the weight $\delta$ decides which region's prices do the adjusting. Benigno's result: **put more weight on the
# region with stickier prices**, $\delta > n$, even though the regions have the same size. Intuition: price
# changes are costlier where prices are sticky (a larger $\gamma/\kappa$), so let the flexible region adjust.

# %%
deltas = np.round(np.linspace(0.2, 0.95, 76), 2)
CASES = {0.75: GREY, 0.85: BLUE, 0.9: RED}                         # alpha_H; alpha_F = 0.75
best = {}
fig, ax = plt.subplots(figsize=(6.2, 3.6))
for aH, col in CASES.items():
    Ls = np.array([sum(union_loss(d, aH=aH).values()) for d in deltas])
    best[aH] = (deltas[np.argmin(Ls)], Ls.min(), sum(union_loss(n, aH=aH).values()))
    ax.plot(deltas, Ls, color=col, label=f"$\\alpha_H$ = {aH}")
    ax.plot(best[aH][0], best[aH][1], "o", color=col)
ax.axvline(n, color=GREY, ls=":", lw=1)
ax.set(title="welfare loss, % of consumption", xlabel="weight on H inflation $\\delta$ ($\\alpha_F$ = 0.75, n = 0.5)")
ax.legend(frameon=False, fontsize=9)
show(fig, "b2_delta")
for aH, (d, Lb, Ln) in best.items():
    print(f"alpha_H = {aH}: best delta = {d:.2f}; loss {Lb:.4f} vs {Ln:.4f} when targeting union inflation")

# %% [markdown]
# ## N0. Services are a luxury: non-homothetic demand and inequality
# Part A's regions differed in price stickiness; their households were identical. Now member countries differ
# in **income inequality**. Each country has two households, Ricardian (R, can save and borrow) and
# hand-to-mouth (H), and two goods, tradable goods (T) and non-tradable services (N). Preferences are
# non-homothetic (Boppart 2014): a household with expenditure $e$ spends the share
# $$\omega^T(e) = \nu \left(\frac{P^N}{e}\right)^{\varepsilon}\left(\frac{P^T}{P^N}\right)^{\gamma},
# \qquad \omega^N(e) = 1 - \omega^T(e)$$
# on goods and services. With $\varepsilon > 0$ the services share rises with expenditure: services are a
# luxury. $\varepsilon = \gamma = 0$ is Cobb-Douglas with constant shares $\nu$ and $1 - \nu$.
#
# **Production and incomes.** $y^s = Z (l^s)^{\alpha_s}$ with $\alpha_N = 1$, $\alpha_T < 1$. The Ricardian
# household receives the share $\kappa_s$ of sector $s$'s income and the hand-to-mouth household $1 - \kappa_s$.
# We keep $\kappa_T = 0.5$ and vary $\kappa_N \ge 0.5$: the higher $\kappa_N$, the more of the services
# sector's income goes to the Ricardian household, and the more unequal the country. (The paper writes G and S
# for goods and services, and $\xi$ for $\kappa$.)
#
# The cell computes the stationary allocation (flexible wages, full employment, $P^T = 1$) for each $\kappa_N$.

# %%
nu, eps_nh, gam = 0.6, 0.18, 0.0          # goods share, non-homotheticity (Boppart 2014), substitution
aT, aN, lbar, kT = 0.43, 1.0, 0.93, 0.5   # labour elasticities, labour endowment, R's share of T income
R_ss, R_shock = 1.01, 1.02                # the nominal rate rises from 1% to 2% for one period
beta_n = 1 / R_ss
Z = 1.0                                   # productivity; set below so that the benchmark services share is 40%


def share_T(e, pT, pN, eps=None):
    """goods share of expenditure; the services share is 1 - share_T"""
    eps = eps_nh if eps is None else eps
    return nu * (pN / e)**eps * (pT / pN)**gam


def stationary(kN, A=0.0, eps=None, x0=None):
    """Stationary allocation of one country. A = Ricardian wealth in units of goods (interest (R-1)A)."""
    def unpack(x):
        yT, yN, pN = np.exp(x)
        WR = (aT * kT * yT + aN * kN * pN * yN) / lbar               # wages clear each type's labour market
        WH = (aT * (1 - kT) * yT + aN * (1 - kN) * pN * yN) / lbar
        eR = kT * yT + kN * pN * yN + (R_ss - 1) * A                 # expenditure = income
        eH = (1 - kT) * yT + (1 - kN) * pN * yN
        return yT, yN, pN, WR, WH, eR, eH

    def f(x):
        yT, yN, pN, WR, WH, eR, eH = unpack(x)
        lTR, lTH = aT * kT * yT / WR, aT * (1 - kT) * yT / WH        # labour demand, by type and sector
        lNR, lNH = aN * kN * pN * yN / WR, aN * (1 - kN) * pN * yN / WH
        sR, sH = share_T(eR, 1, pN, eps), share_T(eH, 1, pN, eps)
        return [yT - Z * (lTH**(1 - kT) * lTR**kT)**aT,              # technology, goods
                yN - Z * (lNH**(1 - kN) * lNR**kN)**aN,              # technology, services
                pN * yN - (1 - sR) * eR - (1 - sH) * eH]             # services market clears at home

    s = root(f, np.log([0.6, 0.4, 1.0]) if x0 is None else x0, tol=1e-13)
    assert max(abs(np.array(f(s.x)))) < 1e-9
    yT, yN, pN, WR, WH, eR, eH = unpack(s.x)
    return dict(x=s.x, yT=yT, yN=yN, pN=pN, eR=eR, eH=eH, cTR=share_T(eR, 1, pN, eps) * eR,
                shareN=pN * yN / (yT + pN * yN), shareN_R=1 - share_T(eR, 1, pN, eps),
                shareN_H=1 - share_T(eH, 1, pN, eps))


def _gap(z):
    global Z
    Z = z
    s = stationary(0.5)
    return s["eR"] / s["pN"] - 1


Z = brentq(_gap, 0.5, 5.0)                 # benchmark (kappa_N = 0.5): e / P^N = 1, services share = 1 - nu

kNs = np.linspace(0.5, 0.98, 25)
SS = [stationary(k) for k in kNs]
SS_cd = [stationary(k, eps=0.0) for k in kNs]
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axes[0]
ax.plot([s["eR"] / s["pN"] for s in SS], [100 * s["shareN_R"] for s in SS], "o", color=BLUE, ms=4, label="Ricardian")
ax.plot([s["eH"] / s["pN"] for s in SS], [100 * s["shareN_H"] for s in SS], "o", color=RED, ms=4, label="hand-to-mouth")
ax.plot([s["eR"] / s["pN"] for s in SS_cd] + [s["eH"] / s["pN"] for s in SS_cd], [100 * (1 - nu)] * (2 * len(kNs)),
        color=GREY, lw=1, label="Cobb-Douglas")
ax.set(title="household services share, %", xlabel="household expenditure (benchmark = 1)")
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
ax.plot(kNs, [100 * s["shareN"] for s in SS], color=BLUE, lw=2, label="non-homothetic")
ax.plot(kNs, [100 * s["shareN"] for s in SS_cd], color=GREY, lw=1, label="Cobb-Douglas")
ax.set(title="country's services share, %", xlabel="R's share of services income $\\kappa_N$ (inequality)")
ax.legend(frameon=False, fontsize=9)
show(fig, "n0_shares")
s_eq, s_un = stationary(0.5), stationary(0.9)
print(f"kappa_N = 0.5: e_R/e_H = {s_eq['eR'] / s_eq['eH']:.2f}, services share {100 * s_eq['shareN']:.1f}%.  "
      f"kappa_N = 0.9: e_R/e_H = {s_un['eR'] / s_un['eH']:.2f}, services share {100 * s_un['shareN']:.1f}% "
      f"(R {100 * s_un['shareN_R']:.1f}%, H {100 * s_un['shareN_H']:.1f}%)")

# %% [markdown]
# **What to see.** Left: every household is on the same upward-sloping curve, so richer households spend a
# larger share on services (Fact 1). Right: the curve is concave, but services *spending* is convex in
# expenditure, so spreading expenditure unequally raises the country's services share (Fact 2). With
# Cobb-Douglas preferences both lines are flat.

# %% [markdown]
# ## N1. A rate rise in a union of identical countries
# Two periods. In period 1 the central bank raises the nominal rate and **nominal wages are fixed**; from
# period 2 on wages are flexible and the economy is back at the stationary allocation. Three equations pin down
# period 1:
#
# * **Services price fixed.** $\alpha_N = 1$ and fixed wages give $P^N_1 = P^N$.
# * **Goods supply.** Fixed wages and $\alpha_T < 1$: $y^T_1 = y^T (P^T_1)^{\alpha_T/(1-\alpha_T)}$, an
#   upward-sloping supply curve.
# * **Ricardian demand.** The Euler equation with the goods-inflation target ($\gamma = 0$):
#   $c^T_{R,1} = c^T_{R,2} / (\beta R^n_1)$. The Ricardian household's goods consumption falls one for one with
#   the rate.
#
# The hand-to-mouth household spends its income, $e_H = (1-\kappa_T) P^T y^T + (1-\kappa_N) P^N y^N$. Markets
# for goods and services clear. `period1` solves this for a union of one or more (equally sized) country types;
# with several types the Ricardian households trade a bond, and $B$ is the country's trade balance.

# %%
def period1(kN_list, R1=R_shock, eps=None):
    """Period-1 equilibrium of a union of equally sized country types with services-income shares kN_list."""
    S = [stationary(k, eps=eps) for k in kN_list]
    n_c = len(kN_list)

    def parts(x):
        pT, out = np.exp(x[0]), []
        for i, (k, ss) in enumerate(zip(kN_list, S)):
            yN, B = np.exp(x[1 + 2 * i]), x[2 + 2 * i]
            yT, pN = ss["yT"] * pT**(aT / (1 - aT)), ss["pN"]        # goods supply; services price fixed
            eR = kT * pT * yT + k * pN * yN - pT * B                 # R saves B (in goods) abroad
            eH = (1 - kT) * pT * yT + (1 - k) * pN * yN
            out.append((yT, yN, pN, eR, eH, B, pT))
        return out

    def f(x):
        res, P = [], parts(x)
        for (yT, yN, pN, eR, eH, B, pT), k, ss in zip(P, kN_list, S):
            sR, sH = share_T(eR, pT, pN, eps), share_T(eH, pT, pN, eps)
            c2 = stationary(k, A=R1 * B, eps=eps, x0=ss["x"])["cTR"] if n_c > 1 else ss["cTR"]
            res += [pN * yN - (1 - sR) * eR - (1 - sH) * eH,         # services market, country by country
                    sR * eR / pT - c2 / (beta_n * R1)]               # Ricardian Euler equation
        res += [sum(p[5] for p in P)]                                # bonds in zero net supply = goods market
        if n_c == 1:
            res[-1] = P[0][5]                                        # one country type: no trade, B = 0
        return res

    x0 = [np.log(0.99)] + sum([[np.log(ss["yN"] * 0.97), 0.0] for ss in S], [])
    s = root(f, x0, tol=1e-12)
    assert max(abs(np.array(f(s.x)))) < 1e-8
    out = []
    for (yT, yN, pN, eR, eH, B, pT), ss in zip(parts(s.x), S):
        y0 = ss["yT"] + ss["pN"] * ss["yN"]                          # real output at stationary prices
        out.append(dict(y=100 * ((yT + pN * yN) / y0 - 1), yT=100 * (yT / ss["yT"] - 1),
                        yN=100 * (yN / ss["yN"] - 1), pT=100 * (pT - 1),
                        cTR=100 * (1 / (beta_n * R1) - 1) if n_c == 1 else np.nan,
                        cT=100 * ((yT - B) / ss["yT"] - 1), tb=100 * B / ss["yT"],
                        rel=100 * (eR / eH / (ss["eR"] / ss["eH"]) - 1)))
    return out


closed = pd.DataFrame([period1([k])[0] for k in kNs], index=kNs)
closed_cd = pd.DataFrame([period1([k], eps=0.0)[0] for k in kNs], index=kNs)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, col, title in zip(axes, ["yT", "rel", "y"],
                          ["goods output, %", "relative expenditure $e_R/e_H$, %", "total output, %"]):
    ax.plot(kNs, closed[col], color=BLUE, lw=2, label="non-homothetic")
    ax.plot(kNs, closed_cd[col], color=GREY, lw=1, label="Cobb-Douglas")
    ax.set(title=title, xlabel="$\\kappa_N$ (inequality)")
axes[0].axhline(closed["cTR"].iloc[0], color=RED, ls="--", lw=1)
axes[0].text(0.62, closed["cTR"].iloc[0] + 0.004, "Ricardian goods demand", color=RED, fontsize=8)
axes[0].legend(frameon=False, fontsize=8, loc="center right")
show(fig, "n1_closed")
a, b = period1([0.5])[0], period1([0.9])[0]
print(f"Rate 1% -> 2%. kappa_N = 0.5: goods output {a['yT']:.2f}%, goods price {a['pT']:.2f}%, services {a['yN']:.2f}%, "
      f"total {a['y']:.2f}%, e_R/e_H {a['rel']:.2f}%")
print(f"               kappa_N = 0.9: goods output {b['yT']:.2f}%, goods price {b['pT']:.2f}%, services {b['yN']:.2f}%, "
      f"total {b['y']:.2f}%, e_R/e_H {b['rel']:.2f}%")

# %% [markdown]
# **What to see.** The Ricardian household's goods demand always falls by the same amount (dashed line): that
# is the direct effect of the rate. With $\kappa_N = 0.5$ the two households are identical and total goods
# output falls by exactly that amount. With $\kappa_N > 0.5$ the recession hits services hardest (they are the
# luxury), services income goes mostly to the Ricardian household, so $e_R/e_H$ **falls**: inequality is
# pro-cyclical. The hand-to-mouth household, which spends everything and mostly on goods, loses less income,
# and output falls less. Both ingredients are needed: with Cobb-Douglas preferences the relative value of
# services is constant, $e_R/e_H$ does not move, and the output response is the same for every $\kappa_N$;
# with $\kappa_N = \kappa_T$ the two households' incomes move together. (Levels differ a little from the paper's figures because
# the normalisation of the stationary allocation is not identical; the comparative statics are the same.)

# %% [markdown]
# ## N2. An equal and an unequal country in one union
# Now half of the union's countries are **equal** ($\kappa_N = 0.5$) and half **unequal** ($\kappa_N > 0.5$).
# They share the nominal rate and the goods price $P^T_1$; services clear at home. Ricardian households can
# lend to each other across countries, so goods consumption can differ from goods output: the difference is
# the trade balance.

# %%
kU = np.linspace(0.5, 0.98, 13)
EQ = pd.DataFrame([period1([0.5, k])[0] for k in kU], index=kU)
UN = pd.DataFrame([period1([0.5, k])[1] for k in kU], index=kU)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, col, title in zip(axes, ["y", "cT", "tb"],
                          ["total output, %", "goods consumption, %", "trade balance, % of goods output"]):
    ax.plot(kU, EQ[col], color=BLUE, lw=2, label="equal country")
    ax.plot(kU, UN[col], color=RED, lw=2, ls="--", label="unequal country")
    ax.set(title=title, xlabel="$\\kappa_N$ in the unequal country")
axes[1].plot(kU, EQ["yT"], color=GREY, lw=1, label="goods output (both)")
axes[1].legend(frameon=False, fontsize=8)
axes[0].legend(frameon=False, fontsize=8)
show(fig, "n2_union")
e, u = period1([0.5, 0.9])
print(f"Union of kappa_N = 0.5 and 0.9: goods price {e['pT']:.2f}% (closed economies: {a['pT']:.2f}% and {b['pT']:.2f}%), "
      f"goods output {e['yT']:.2f}% in both")
print(f"  equal country:   total output {e['y']:.2f}%, goods consumption {e['cT']:.2f}%, trade balance {e['tb']:+.3f}% of goods output")
print(f"  unequal country: total output {u['y']:.2f}%, goods consumption {u['cT']:.2f}%, trade balance {u['tb']:+.3f}% of goods output")

# %% [markdown]
# **What to see.** The union's goods price settles between the two closed-economy prices, so goods output
# falls by the same amount everywhere. Goods *consumption* falls less in the unequal country, where
# pro-cyclical inequality protects the hand-to-mouth household: it imports the difference and runs a **trade
# deficit**; the equal country runs a surplus. Services output, and so total output, falls less in the unequal
# country. One interest rate, different recessions, and trade imbalances inside the union.
