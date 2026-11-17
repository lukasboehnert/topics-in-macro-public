# Week 4 lab (fast version): monetary policy in a currency union, and a union of unequal countries.
# The same cells, calibration and numbers as `week4_simple.py`, without the explanations.
# Writes the lab's figures to `figures/`. A few seconds.
# %% Setup

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
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# %% The solver: a linear rational-expectations solver (Klein 2000)

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


# %% B0. A two-region currency union (Benigno 2004)

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

# %% B1. An asymmetric shock: what the union cannot do

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

# %% B2. Which inflation should the union target?

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

# %% N0. Services are a luxury: non-homothetic demand and inequality

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

# %% N1. A rate rise in a union of identical countries

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

# %% N2. An equal and an unequal country in one union

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

