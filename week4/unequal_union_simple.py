# %% [markdown]
# # Week 4 class code: monetary policy in a currency union with unequal countries
#
# **Paper.** Boehnert, de Ferra, Mitman and Romei (2025), *Monetary Policy in Currency Unions with Unequal
# Countries*. We build the paper's two-agent, two-sector, two-period model from scratch, one step per cell, in
# the order of the slides:
#
# | Step | What we do | Slide |
# | --- | --- | --- |
# | 1 | Preferences: services are a luxury | Preferences: services are a luxury |
# | 2 | Incomes and the stationary allocation; inequality and the services share | Incomes: who earns what |
# | 3 | The experiment: a rate rise with rigid wages, one country | The experiment |
# | 4 | The mechanism: pro-cyclical inequality, checked with the $e_R/e_H$ formula | The mechanism |
# | 5 | More inequality, weaker transmission | Identical countries |
# | 6 | Which ingredient matters? | (in-class problem) |
# | 7 | An equal and an unequal country in one union: trade imbalances | A union of an equal and an unequal country |
#
# **Notation.** T = tradable goods, N = non-tradable services (the paper writes G and S). R = Ricardian
# household (saves and borrows), H = hand-to-mouth household (spends its income). $\kappa_s$ = R's share of
# sector $s$'s income (the paper's $\xi_s$).
#
# **How the code is written.** Everything is non-linear and solved with `scipy.optimize.root`; every function
# returns a dictionary of levels, and the experiments report % changes. No linearisation, no solver of our own.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run unequal_union_simple.py`
# (about ten seconds). `unequal_union_fast.py` is the same code without the explanations.

# %% [markdown]
# ## Setup and calibration
# The calibration is the paper's. $\nu$ sets the goods share at the benchmark (60%, so 40% services),
# $\varepsilon$ the strength of the income effect (Boppart 2014), $\gamma = 0$ for analytical convenience.
# Services are linear in labour ($\alpha_N = 1$), goods have decreasing returns ($\alpha_T = 0.43$): together a
# labour share of about two thirds. The shock raises the nominal rate from 1% to 2% for one period.

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
FIGS = HERE / "figures_simple"
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

# %% [markdown]
# ## Step 1. Preferences: services are a luxury
# Boppart (2014) indirect utility over expenditure $e$ and prices:
# $$v(e, P^T, P^N) = \frac{1}{\varepsilon}\Big[\Big(\frac{e}{P^N}\Big)^{\varepsilon} - 1\Big]
#   - \frac{\nu}{\gamma}\Big[\Big(\frac{P^T}{P^N}\Big)^{\gamma} - 1\Big].$$
# Roy's identity gives the expenditure shares
# $$\omega^T(e) = \nu \Big(\frac{P^N}{e}\Big)^{\varepsilon}\Big(\frac{P^T}{P^N}\Big)^{\gamma},
#   \qquad \omega^N(e) = 1 - \omega^T(e).$$
# With $\varepsilon > 0$ the goods share falls, and the services share rises, with expenditure (the paper's
# Lemma 1: $\partial\omega^N/\partial e = \varepsilon\,(1 - \omega^N)/e > 0$). With $\varepsilon = \gamma = 0$
# both shares are constant: Cobb-Douglas.

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

# %% [markdown]
# **What to see.** One curve for everybody: a household's services share depends only on how much it
# spends. A household that spends 50% more than the benchmark spends about 44% on services instead of 40%.
# This is Fact 1.

# %% [markdown]
# ## Step 2. Incomes and the stationary allocation
# **Technology.** $y^s = Z\,(l^s)^{\alpha_s}$, where $l^s$ is a Cobb-Douglas bundle of the two households'
# labour, $l^s = (l^s_H)^{1-\kappa_s}(l^s_R)^{\kappa_s}$. With perfect competition and profits paid in
# proportion to labour income, **R receives the share $\kappa_s$ of sector $s$'s revenue**:
# $$e_R = \kappa_T P^T y^T + \kappa_N P^N y^N, \qquad e_H = (1-\kappa_T) P^T y^T + (1-\kappa_N) P^N y^N.$$
# **Inequality.** $\kappa_T = 1/2$ everywhere; the more unequal the country, the higher $\kappa_N$.
#
# **Stationary allocation** (flexible wages, full employment, $P^T = 1$, no assets). Three unknowns
# $(y^T, y^N, P^N)$ and three equations: the two technologies, with each household's labour demand
# $l^s_j = \alpha_s \kappa^j_s P^s y^s / W_j$ and wages that clear each household's labour market,
# and the services market, which clears at home: $P^N y^N = \omega^N_R e_R + \omega^N_H e_H$.
# We pick $Z$ so that at the benchmark ($\kappa_N = 1/2$) $e / P^N = 1$ and the services share is $1 - \nu$.

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

# %% [markdown]
# **What to see.** Left: each country ($\kappa_N$ from 0.5 to 0.98) contributes one R dot and one H dot,
# all on the curve of Step 1 (Fact 1). Right: the more unequal the country, the higher its services share
# (Fact 2). Why: services *spending*, $e - \nu e^{1-\varepsilon}(P^N)^{\varepsilon}$, is convex in $e$, so
# by Jensen's inequality spreading a given total expenditure unequally raises total services spending
# (the paper's Lemma 3).

# %% [markdown]
# ## Step 3. The experiment: a rate rise with rigid wages
# **Period 1:** the nominal rate rises to $R^n_1$ and nominal wages stay at their stationary level.
# **Period 2:** wages are flexible again and the economy is back at the stationary allocation; the central
# bank keeps the goods price at 1 from then on. With rigid wages:
#
# * **Services price fixed.** $\alpha_N = 1$, so $P^N_1 = W^N / Z$ does not move: services output is set by demand.
# * **Goods supply.** $\alpha_T < 1$ and a fixed wage: $P^T_1 = \frac{W^T}{\alpha_T Z^{1/\alpha_T}} (y^T_1)^{(1-\alpha_T)/\alpha_T}$,
#   i.e. $y^T_1 = y^T (P^T_1)^{\alpha_T/(1-\alpha_T)}$: more goods only at a higher price.
# * **Ricardian demand.** The Euler equation in goods with $\gamma = 0$ and $P^T_2 = 1$:
#   $c^T_{R,1} = c^T_{R,2}/(\beta R^n_1)$. R's goods consumption falls by the same proportion in every country:
#   the **direct effect** of monetary policy.
#
# One country (or a union of identical countries): no trade, so goods output = goods consumption. Two
# unknowns, $P^T_1$ and $y^N_1$, and two equations: the services market and R's Euler equation. H spends its
# income; total goods demand is R's plus H's.

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

# %% [markdown]
# **What to see.** R's goods consumption falls by $1 - 1/(\beta R^n_1) \approx 0.98\%$ in both countries:
# the direct effect does not depend on inequality. With $\kappa_N = 0.5$ the two households are identical
# and everything falls together. With $\kappa_N = 0.9$ H's goods consumption falls **less** than R's, so
# total goods demand, the goods price and output fall less. Services fall more than goods in both: they are
# the luxury, so spending moves away from them in a recession.

# %% [markdown]
# ## Step 4. The mechanism: pro-cyclical inequality
# Divide R's income by H's. The only endogenous object is the relative value of services output:
# $$\frac{e_R}{e_H} = \frac{\kappa_N \frac{P^N y^N}{P^T y^T} + \kappa_T}{(1-\kappa_N)\frac{P^N y^N}{P^T y^T} + (1-\kappa_T)}.$$
# Two conditions make $e_R/e_H$ fall in a recession:
# 1. **non-homothetic preferences**: in a recession everybody is poorer and spends less on the luxury, so
#    $P^N y^N / P^T y^T$ falls (with Cobb-Douglas it stays at $(1-\nu)/\nu$);
# 2. **$\kappa_N > \kappa_T$**: the formula is increasing in $P^N y^N / P^T y^T$ only if R's income is more
#    exposed to services.
#
# Below we plug the model's relative value of services, before and after the shock, into the formula and
# compare with the model's own $e_R/e_H$.

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

# %% [markdown]
# **What to see.** The formula reproduces the model. With Cobb-Douglas the relative value of services does
# not move, so $e_R/e_H$ does not move whatever $\kappa_N$. With non-homothetic preferences it falls; with
# $\kappa_N = 0.5$ that does not matter (both households earn the same mix), with $\kappa_N = 0.9$ R's
# relative income falls by about half a percent. **Inequality falls in the recession: it is pro-cyclical.**
# H, who spends all of its income and mostly on goods, is partly insulated, and total demand falls by less
# than R's: the paper's Proposition 2.

# %% [markdown]
# ## Step 5. More inequality, weaker transmission
# Repeat the experiment for every $\kappa_N$ and plot the responses against the country's services share,
# as in the data (Fact 3).

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

# %% [markdown]
# **What to see.** Top left: the higher the services share, the smaller the fall in output (the paper's
# Figure 8, and Fact 3). Top right: R's goods demand always falls by 0.98%; total goods demand falls less,
# the more so the higher $\kappa_N$ (Figure 9, left). Bottom left: the fall in $e_R/e_H$, which is zero
# under Cobb-Douglas (Figure 9, right). Bottom right: goods prices and services output also fall less in
# more unequal countries (Figure 10). The services share is a *symptom*; the cause is who earns the services
# income. (Levels differ a little from the paper's figures because the stationary allocation is normalised
# differently; the comparative statics are the same.)

# %% [markdown]
# ## Step 6. Which ingredient matters?
# Switch each ingredient off in turn and compare an equal ($\kappa_N = 0.5$) with an unequal country
# ($\kappa_N = 0.9$):
# * Cobb-Douglas preferences, $\varepsilon = 0$;
# * the benchmark, $\varepsilon = 0.18$;
# * stronger non-homotheticity, $\varepsilon = 0.22$ (the paper's Figure 13);
# * inequality from the **goods** sector instead: $\kappa_N = 0.5$, $\kappa_T = 0.9$ (the paper's Figure 14).

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

# %% [markdown]
# **What to see.** Without non-homotheticity there is no difference between the countries. More
# non-homotheticity deepens the recession everywhere (luxuries collapse) *and* widens the gap between
# countries. If instead the rich earn their income from **goods**, the sign flips: $e_R/e_H$ now *rises* in
# the recession, H loses most, and transmission is **stronger** in the unequal country. The data (Fact 3)
# therefore tell us where inequality comes from: services income.

# %% [markdown]
# ## Step 7. An equal and an unequal country in one union
# Half of the union's countries are equal ($\kappa_N = 0.5$), half unequal ($\kappa_N > 0.5$). They share
# the nominal rate and the goods price; services clear at home. Now R households can lend across borders:
# R in country $i$ buys bonds $B_i$ in period 1, so $e_{R,i} = \kappa_T P^T y^T_i + \kappa_N P^N y^N_i - P^T B_i$
# and its period-2 allocation includes the interest. Goods consumption can differ from goods output; the
# difference is the **trade balance**, $y^T_i - c^T_i = B_i$.
#
# Unknowns: the common $P^T_1$, and per country $y^N_{i,1}$ and $B_i$. Equations: per country the services
# market and R's Euler equation (with period-2 consumption from the stationary allocation with assets
# $R^n_1 B_i$); for the union, bonds in zero net supply, $\sum_i B_i = 0$, which is goods-market clearing.

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

# %% [markdown]
# **What to see.** One goods price for the union, between the two closed-economy prices (Lemma 6). Same
# price, same goods supply: goods **output** falls by the same amount in both countries. Goods
# **consumption** falls less in the unequal country, where H is insulated: it imports the difference and
# runs a **trade deficit**; the equal country runs a surplus (Proposition 4). Services clear at home, so
# total output also falls less in the unequal country (Proposition 3). The figure is the paper's Figure 11.
# One interest rate, different recessions, and trade imbalances inside the union, with the sign we see in
# the data.
#
# **Try this.** (i) Make the shock larger, `R_shock = 1.03`: do the imbalances scale with it? (ii) Put a
# third, very unequal country type in the union, `period1_union([0.5, 0.75, 0.98])`. (iii) Union of an
# equal country and one whose inequality comes from goods: change `period1_union` so that each country
# has its own $\kappa_T$ (take-home).
