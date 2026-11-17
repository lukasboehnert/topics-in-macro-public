# Week 4 class code (fast version): monetary policy in a currency union with unequal countries
# (Boehnert, de Ferra, Mitman and Romei 2025). The same steps, calibration and numbers as
# `unequal_union_simple.py`, without the explanations. Writes the figures to `figures/`.
# %% Setup and calibration

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
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


nu, eps, gam = 0.6, 0.18, 0.0         # preferences: goods weight, non-homotheticity, substitution
aT, aN = 0.43, 1.0                    # output elasticity to labour: goods, services
lbar = 0.93                           # labour endowment of each household (inelastic supply)
kT_base = 0.5                         # R's share of goods income, the same in every country
R_ss, R_shock = 1.01, 1.02            # gross nominal rate: steady state and period 1
beta = 1 / R_ss                       # discount factor consistent with the steady-state rate

# %% Step 1. Preferences: services are a luxury

# %%
def share_T(e, pT=1.0, pN=1.0, eps_=None):
    """Goods (tradable) share of expenditure e. The services share is 1 - share_T."""
    eps_ = eps if eps_ is None else eps_
    return nu * (pN / e)**eps_ * (pT / pN)**gam


e_grid = np.linspace(0.6, 1.6, 101)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(e_grid, 100 * (1 - share_T(e_grid, eps_=0.0)), color=GREY, label="Cobb-Douglas ($\\varepsilon$ = 0)")
ax.plot(e_grid, 100 * (1 - share_T(e_grid)), color=BLUE, lw=2, label="non-homothetic ($\\varepsilon$ = 0.18)")
ax.set(title="services share of expenditure, %", xlabel="expenditure $e / P^N$")
ax.legend(frameon=False, fontsize=9)
show(fig, "u1_shares")

# Lemma 1, checked numerically at e = 1.2: the slope of the services share equals eps (1 - omega_N) / e
e0, h = 1.2, 1e-6
slope = ((1 - share_T(e0 + h)) - (1 - share_T(e0 - h))) / (2 * h)
print(f"Services share at e = {e0}: {100 * (1 - share_T(e0)):.1f}%. Slope {slope:.4f}; "
      f"Lemma 1 formula eps * omega_T / e = {eps * share_T(e0) / e0:.4f}")

# %% Step 2. Incomes and the stationary allocation

# %%
Z = 1.0


def stationary(kN, kT=kT_base, A=0.0, eps_=None, x0=None):
    """Stationary allocation of one country. A = R's net foreign assets in goods; it earns (R_ss - 1) A."""
    def unpack(x):
        yT, yN, pN = np.exp(x)
        WR = (aT * kT * yT + aN * kN * pN * yN) / lbar              # R's wage: its labour market clears
        WH = (aT * (1 - kT) * yT + aN * (1 - kN) * pN * yN) / lbar  # H's wage
        eR = kT * yT + kN * pN * yN + (R_ss - 1) * A                # income = expenditure
        eH = (1 - kT) * yT + (1 - kN) * pN * yN
        return yT, yN, pN, WR, WH, eR, eH

    def equations(x):
        yT, yN, pN, WR, WH, eR, eH = unpack(x)
        lTR, lTH = aT * kT * yT / WR, aT * (1 - kT) * yT / WH       # labour demand, goods
        lNR, lNH = aN * kN * pN * yN / WR, aN * (1 - kN) * pN * yN / WH
        sR, sH = share_T(eR, 1, pN, eps_), share_T(eH, 1, pN, eps_)
        return [yT - Z * (lTH**(1 - kT) * lTR**kT)**aT,             # technology, goods
                yN - Z * (lNH**(1 - kN) * lNR**kN)**aN,             # technology, services
                pN * yN - (1 - sR) * eR - (1 - sH) * eH]            # services market clears at home

    sol = root(equations, np.log([0.6, 0.4, 1.0]) if x0 is None else x0, tol=1e-13)
    assert max(abs(np.array(equations(sol.x)))) < 1e-9, "stationary allocation not found"
    yT, yN, pN, WR, WH, eR, eH = unpack(sol.x)
    return dict(x=sol.x, yT=yT, yN=yN, pN=pN, eR=eR, eH=eH,
                cTR=share_T(eR, 1, pN, eps_) * eR, cTH=share_T(eH, 1, pN, eps_) * eH,
                shareN=pN * yN / (yT + pN * yN),                    # country's services share
                shareN_R=1 - share_T(eR, 1, pN, eps_), shareN_H=1 - share_T(eH, 1, pN, eps_),
                gini=abs(eR - eH) / (2 * (eR + eH)))               # Gini with two equal-sized groups


def _benchmark_gap(z):
    global Z
    Z = z
    s = stationary(0.5)
    return s["eR"] / s["pN"] - 1


Z = brentq(_benchmark_gap, 0.5, 5.0)
kN_grid = np.round(np.linspace(0.5, 0.98, 25), 2)
SS = pd.DataFrame([stationary(k) for k in kN_grid], index=kN_grid).drop(columns="x")
print(f"Z = {Z:.3f}")
print(SS.loc[[0.5, 0.7, 0.9], ["eR", "eH", "gini", "shareN_R", "shareN_H", "shareN"]].round(3))

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
ax = axes[0]
ax.plot(SS["eR"] / SS["pN"], 100 * SS["shareN_R"], "o", color=BLUE, ms=4, label="Ricardian")
ax.plot(SS["eH"] / SS["pN"], 100 * SS["shareN_H"], "o", color=RED, ms=4, label="hand-to-mouth")
ax.set(title="household services share, %", xlabel="household expenditure $e / P^N$")
ax.legend(frameon=False, fontsize=9)
ax = axes[1]
ax.plot(SS["gini"], 100 * SS["shareN"], color=BLUE, lw=2)
ax.set(title="country's services share, %", xlabel="Gini of household income")
show(fig, "u2_inequality")

# %% Step 3. The experiment: a rate rise with rigid wages

# %%
def period1_closed(kN, kT=kT_base, R1=R_shock, eps_=None):
    """Period 1 after the rate rise, one country or a union of identical countries."""
    ss = stationary(kN, kT, eps_=eps_)
    pN = ss["pN"]                                                   # services price fixed

    def unpack(x):
        pT, yN = np.exp(x)
        yT = ss["yT"] * pT**(aT / (1 - aT))                         # goods supply with a fixed wage
        eR = kT * pT * yT + kN * pN * yN
        eH = (1 - kT) * pT * yT + (1 - kN) * pN * yN
        return pT, yT, yN, eR, eH

    def equations(x):
        pT, yT, yN, eR, eH = unpack(x)
        sR, sH = share_T(eR, pT, pN, eps_), share_T(eH, pT, pN, eps_)
        return [pN * yN - (1 - sR) * eR - (1 - sH) * eH,            # services market
                sR * eR / pT - ss["cTR"] / (beta * R1)]             # R's Euler equation

    sol = root(equations, np.log([0.99, 0.97 * ss["yN"]]), tol=1e-13)
    assert max(abs(np.array(equations(sol.x)))) < 1e-10, "period 1 not found"
    pT, yT, yN, eR, eH = unpack(sol.x)
    cTR, cTH = share_T(eR, pT, pN, eps_) * eR / pT, share_T(eH, pT, pN, eps_) * eH / pT
    pct = lambda new, old: 100 * (new / old - 1)
    y0 = ss["yT"] + pN * ss["yN"]                                   # real output at stationary prices
    return dict(pT=pct(pT, 1), yT=pct(yT, ss["yT"]), yN=pct(yN, ss["yN"]), y=pct(yT + pN * yN, y0),
                cTR=pct(cTR, ss["cTR"]), cTH=pct(cTH, ss["cTH"]), eR=pct(eR, ss["eR"]), eH=pct(eH, ss["eH"]),
                rel=pct(eR / eH, ss["eR"] / ss["eH"]), shareN=ss["shareN"],
                relvalue0=pN * ss["yN"] / ss["yT"], relvalue1=pN * yN / (pT * yT))


cases = {"equal, kappa_N = 0.5": 0.5, "unequal, kappa_N = 0.9": 0.9}
table = pd.DataFrame({name: period1_closed(k) for name, k in cases.items()}).T
print("Rate from 1% to 2% for one period; % change from the stationary allocation:")
print(table[["pT", "yT", "yN", "y", "cTR", "cTH", "eR", "eH"]].round(2))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
labels = ["R's goods\n(direct effect)", "H's goods", "total goods", "services", "total output"]
cols = ["cTR", "cTH", "yT", "yN", "y"]
x = np.arange(len(cols))
for i, ((name, row), col) in enumerate(zip(table.iterrows(), [GREY, BLUE])):
    ax.bar(x + (i - 0.5) * 0.38, row[cols], width=0.36, color=col, label=name.replace("kappa_N", "$\\kappa_N$"))
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, labels, fontsize=9)
ax.set(title="period 1, % change")
ax.legend(frameon=False, fontsize=9)
show(fig, "u3_experiment")

# %% Step 4. The mechanism: pro-cyclical inequality

# %%
def er_eh(relvalue, kN, kT=kT_base):
    return (kN * relvalue + kT) / ((1 - kN) * relvalue + (1 - kT))


for name, k in cases.items():
    for eps_ in (0.0, eps):
        r = period1_closed(k, eps_=eps_)
        before, after = er_eh(r["relvalue0"], k), er_eh(r["relvalue1"], k)
        print(f"{name}, eps = {eps_:.2f}: P^N y^N / P^T y^T {r['relvalue0']:.4f} -> {r['relvalue1']:.4f}; "
              f"e_R/e_H {before:.4f} -> {after:.4f} ({100 * (after / before - 1):+.2f}%, "
              f"model {r['rel']:+.2f}%)")

# %% Step 5. More inequality, weaker transmission

# %%
CL = pd.DataFrame([period1_closed(k) for k in kN_grid], index=kN_grid)
CL_cd = pd.DataFrame([period1_closed(k, eps_=0.0) for k in kN_grid], index=kN_grid)

fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
ax = axes[0, 0]
ax.plot(100 * CL["shareN"], CL["y"], color=BLUE, lw=2)
ax.set(title="total output, %", xlabel="services share of the country, %")
ax = axes[0, 1]
ax.plot(kN_grid, CL["yT"], color=BLUE, lw=2, label="total goods, $c^T_R + c^T_H$")
ax.plot(kN_grid, CL["cTR"], color=RED, lw=2, ls="--", label="Ricardian goods, $c^T_R$")
ax.set(title="goods consumption, %", xlabel="$\\kappa_N$")
ax.legend(frameon=False, fontsize=9)
ax = axes[1, 0]
ax.plot(kN_grid, CL["rel"], color=BLUE, lw=2, label="non-homothetic")
ax.plot(kN_grid, CL_cd["rel"], color=GREY, lw=1, label="Cobb-Douglas")
ax.set(title="$e_R / e_H$, %", xlabel="$\\kappa_N$")
ax.legend(frameon=False, fontsize=9)
ax = axes[1, 1]
ax.plot(kN_grid, CL["pT"], color=BLUE, lw=2, label="goods price")
ax.plot(kN_grid, CL["yN"], color=GREEN, lw=2, label="services output")
ax.set(title="% change", xlabel="$\\kappa_N$")
ax.legend(frameon=False, fontsize=9)
show(fig, "u5_transmission")
print(f"Total output: {CL['y'].iloc[0]:.2f}% (kappa_N = 0.5, services share {100 * CL['shareN'].iloc[0]:.1f}%) "
      f"to {CL['y'].iloc[-1]:.2f}% (kappa_N = {kN_grid[-1]}, services share {100 * CL['shareN'].iloc[-1]:.1f}%)")

# %% Step 6. Which ingredient matters?

# %%
rows = {}
for label, eps_ in [("Cobb-Douglas, eps = 0", 0.0), ("benchmark, eps = 0.18", eps), ("eps = 0.22", 0.22)]:
    a, b = period1_closed(0.5, eps_=eps_), period1_closed(0.9, eps_=eps_)
    rows[label] = dict(equal=a["y"], unequal=b["y"], gap=b["y"] - a["y"], rel_unequal=b["rel"])
a, b = period1_closed(0.5), period1_closed(0.5, kT=0.9)
rows["inequality from goods, kappa_T = 0.9"] = dict(equal=a["y"], unequal=b["y"], gap=b["y"] - a["y"],
                                                     rel_unequal=b["rel"])
print("Total output, % change; gap = unequal minus equal (positive: weaker transmission in the unequal country)")
print(pd.DataFrame(rows).T.round(2))

# %% Step 7. An equal and an unequal country in one union

# %%
def period1_union(kN_list, kT=kT_base, R1=R_shock, eps_=None):
    """Period 1 of a union of equally sized country types; returns one dictionary per type."""
    S = [stationary(k, kT, eps_=eps_) for k in kN_list]

    def unpack(x):
        pT, out = np.exp(x[0]), []
        for i, (k, ss) in enumerate(zip(kN_list, S)):
            yN, B = np.exp(x[1 + 2 * i]), x[2 + 2 * i]
            yT = ss["yT"] * pT**(aT / (1 - aT))                      # same price, same goods supply
            eR = kT * pT * yT + k * ss["pN"] * yN - pT * B           # R saves B abroad
            eH = (1 - kT) * pT * yT + (1 - k) * ss["pN"] * yN
            out.append((yT, yN, ss["pN"], eR, eH, B))
        return pT, out

    def equations(x):
        pT, out = unpack(x)
        res = []
        for (yT, yN, pN, eR, eH, B), k, ss in zip(out, kN_list, S):
            sR, sH = share_T(eR, pT, pN, eps_), share_T(eH, pT, pN, eps_)
            c2 = stationary(k, kT, A=R1 * B, eps_=eps_, x0=ss["x"])["cTR"]
            res += [pN * yN - (1 - sR) * eR - (1 - sH) * eH,         # services market, at home
                    sR * eR / pT - c2 / (beta * R1)]                 # R's Euler equation
        return res + [sum(o[5] for o in out)]                        # bonds in zero net supply

    x0 = [np.log(0.99)] + sum([[np.log(0.97 * ss["yN"]), 0.0] for ss in S], [])
    sol = root(equations, x0, tol=1e-12)
    assert max(abs(np.array(equations(sol.x)))) < 1e-8, "union equilibrium not found"
    pT, out = unpack(sol.x)
    res = []
    for (yT, yN, pN, eR, eH, B), ss in zip(out, S):
        y0 = ss["yT"] + pN * ss["yN"]
        res.append(dict(pT=100 * (pT - 1), yT=100 * (yT / ss["yT"] - 1), yN=100 * (yN / ss["yN"] - 1),
                        y=100 * ((yT + pN * yN) / y0 - 1), cT=100 * ((yT - B) / ss["yT"] - 1),
                        tb=100 * B / ss["yT"]))
    return res


eq, un = period1_union([0.5, 0.9])
print(f"Goods price, %: equal country alone {period1_closed(0.5)['pT']:.2f}, union {eq['pT']:.2f}, "
      f"unequal country alone {period1_closed(0.9)['pT']:.2f}   (the paper's Lemma 6)")
print(pd.DataFrame({"equal, kappa_N = 0.5": eq, "unequal, kappa_N = 0.9": un}).T
      [["yT", "cT", "tb", "yN", "y"]].round(3))

kU = np.round(np.linspace(0.5, 0.98, 13), 2)
U = [period1_union([0.5, k]) for k in kU]
EQ = pd.DataFrame([u[0] for u in U], index=kU)
UN = pd.DataFrame([u[1] for u in U], index=kU)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
for ax, col, title in zip(axes, ["y", "tb"], ["total output, %", "trade balance, % of goods output"]):
    ax.plot(kU, EQ[col], color=BLUE, lw=2, label="equal country")
    ax.plot(kU, UN[col], color=RED, lw=2, ls="--", label="unequal country")
    ax.set(title=title, xlabel="$\\kappa_N$ in the unequal country")
axes[0].legend(frameon=False, fontsize=9)
show(fig, "u7_union")

