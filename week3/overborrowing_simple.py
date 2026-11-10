# %% [markdown]
# # Week 3 lab, part B: overborrowing and the planner (Bianchi 2011), step by step
#
# **Question.** Why does a small open economy borrow "too much" in good times, and how large should a tax on
# borrowing be to prevent it?
#
# **Model** (Bianchi 2011): the economy consumes tradables and nontradables, borrows abroad at a fixed world rate,
# and can borrow only up to a fraction $\kappa$ of its income valued at today's prices. When tradable consumption
# falls, the relative price of nontradables falls (a real depreciation), the value of collateral falls, and the
# economy must cut borrowing further. Households take prices as given, so they ignore their own effect on
# everybody's collateral: a **pecuniary externality**. A planner who internalises it borrows less.
#
# | Cell | What |
# | --- | --- |
# | B0 | The model, and why the price in the constraint creates an externality |
# | B1 | Competitive equilibrium by time iteration |
# | B2 | Simulation and the definition of a crisis |
# | B3 | The constrained-efficient planner |
# | B4 | Market vs planner: debt, crises, the anatomy of a crisis |
# | B5 | The optimal tax on debt, and welfare |
#
# **How the code is written.** Time iteration on the Euler equation: guess tomorrow's marginal utility, solve
# today's Euler equation at every grid point by bisection, update, repeat. `overborrowing_fast.py` solves the same
# model with the endogenous grid method and gives the same numbers.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run overborrowing_simple.py`.

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
# ## B0. The model, and the externality
# Bonds $b$ (assets; $b<0$ is debt) at the world rate $r$, price $q = 1/(1+r)$. Preferences and constraints:
#
# $$\max E_0 \sum_t \beta^t \frac{c_t^{1-\sigma}}{1-\sigma}, \qquad
#   c = \left[\omega (c^T)^{-\eta} + (1-\omega)(c^N)^{-\eta}\right]^{-1/\eta}$$
# $$q\, b' + c^T + p^N c^N = b + y^T + p^N y^N \qquad \text{(budget)}$$
# $$q\, b' \ge -\kappa \left(y^T + p^N y^N\right) \qquad \text{(collateral)}$$
#
# Nontradables clear at home, $c^N = y^N$, and the household's first-order condition prices them:
# $$p^N = \frac{1-\omega}{\omega}\left(\frac{c^T}{y^N}\right)^{1+\eta}.$$
# So when $c^T$ falls, $p^N$ falls, and so does collateral. This is the amplification.
#
# **Calibration** (annual), from Bianchi's replication code: $r = 4\%$, $\beta = 0.906$, elasticity of
# substitution $1/(1+\eta) = 0.83$, $\sigma = 1 + \eta$ (utility separable in $T$ and $N$), $\omega = 0.307$,
# $\kappa = 0.3235$, $y^N = 1$; $\log y^T$ AR(1) with $\rho = 0.5$ and s.d. 5.8\%, 9 Rouwenhorst states.

# %%
R = 1.04                     # gross world interest rate
beta = 0.906                 # discount factor (beta * R < 1: households are impatient and borrow)
eta = 1 / 0.83 - 1           # elasticity of substitution T/N = 1 / (1 + eta) = 0.83
sigma = 1 + eta              # risk aversion; sigma = 1 + eta makes utility separable in T and N
omega = 0.307                # weight on tradables
kappa = 0.3235               # collateral: share of income that can be pledged
q = 1 / R
yN = 1.0
rho = 0.5
sd_e = 0.058 * np.sqrt(1 - 0.5**2) # s.d. of log y^T: 5.8%
NY = 9   
NB = 800                     # grid points for b


def rouwenhorst(n, rho, sd):
    """Markov chain (grid, transition matrix) for an AR(1) with persistence rho and innovation s.d. sd."""
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


logy, P = rouwenhorst(NY, rho, sd_e)
yT = np.exp(logy)                         # tradable endowment in the 9 states
B = np.linspace(-1.4, 0.0, NB)            # grid for b
print("y^T states:", yT.round(3))


def cagg(cT):                             # the CES consumption bundle, with c^N = y^N
    return (omega * cT**-eta + (1 - omega) * yN**-eta) ** (-1 / eta)


def u(cT):                                # period utility
    return cagg(cT) ** (1 - sigma) / (1 - sigma)


def uT(cT):                               # marginal utility of tradable consumption, du/dc^T
    return omega * cagg(cT) ** (1 + eta - sigma) * cT ** (-1 - eta)


def price(cT):                            # p^N
    return (1 - omega) / omega * (cT / yN) ** (1 + eta)


def psi(cT):                              # kappa * d(p^N y^N) / dc^T, used by the planner (B3)
    return kappa * (1 + eta) * price(cT) * yN / cT


# %% [markdown]
# **The borrowing limit.** For given $(b, y^T)$, the lowest $b'$ that satisfies the constraint. The price inside
# depends on $c^T = b + y^T - q b'$, so on $b'$ itself: find it by bisection. "slack" is positive when the
# constraint holds.

# %%
def slack(bp, b, y):
    cT = np.maximum(b + y - q * bp, 1e-10)
    return q * bp + kappa * (y + price(cT) * yN)


def bisect(f, lo, hi, n=60):
    """Vectorised bisection: f(lo) < 0 <= f(hi) elementwise. Returns hi at the crossing."""
    for _ in range(n):
        mid = (lo + hi) / 2
        ok = f(mid) >= 0
        lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
    return hi


bp_lim = np.empty((NB, NY))               # bp_lim[i, j]: borrowing limit at b = B[i], y^T = yT[j]
for j in range(NY):
    lo = np.full(NB, B[0])
    hi = (B + yT[j]) / q - 1e-8           # c^T just above zero: slack > 0 here
    lim = bisect(lambda x: slack(x, B, yT[j]), lo, hi)
    bp_lim[:, j] = np.where(slack(lo, B, yT[j]) >= 0, B[0], lim)

# Figure: the price in the constraint. Collateral kappa (y^T + p^N y^N) moves with aggregate c^T through p^N.
cT_fix = np.linspace(0.6, 1.4, 200)
coll = kappa * (1.0 + price(cT_fix) * yN)              # at y^T = 1
coll_fixed = kappa * (1.0 + price(1.0) * yN) * np.ones_like(cT_fix)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(cT_fix, price(cT_fix), color=BLUE)
ax[0].set(title="Relative price of nontradables", xlabel="aggregate $c^T$", ylabel="$p^N$")
ax[1].plot(cT_fix, coll_fixed, "--", color=GREY, label="what each household sees ($p^N$ given)")
ax[1].plot(cT_fix, coll, "-", color=RED, label="in equilibrium ($p^N$ moves with $c^T$)")
ax[1].set(title="Borrowing limit $\\kappa(y^T + p^N y^N)$, $y^T = 1$", xlabel="aggregate $c^T$")
ax[1].legend(frameon=False, fontsize=9)
show(fig, "b0_model")
print(f"A 10% fall in aggregate c^T lowers p^N by {100 * (1 - price(0.9) / price(1.0)):.1f}% and the borrowing "
      f"limit by {100 * (1 - (1 + price(0.9)) / (1 + price(1.0))):.1f}%")

# %% [markdown]
# ## B1. Competitive equilibrium by time iteration
# The household's Euler equation, with multiplier $\mu \ge 0$ on the constraint:
# $$u_T(c^T) = \beta R\, E\left[u_T(c^{T\prime})\right] + \mu.$$
# **The idea.** Guess what the future looks like (tomorrow's marginal utility), work out what households do
# today given that guess, use today's choices as the new guess for the future, and repeat until the guess stops
# changing.
#
# **Step 0, before the loop: the borrowing limit** $b'_{lim}(b, y^T)$ (computed in B0). It is a small fixed point
# of its own: collateral depends on $p^N$, $p^N$ on $c^T$, and $c^T = b + y^T - q b'$ on $b'$. Households take
# $p^N$ as given, but in equilibrium it must be consistent with aggregate $c^T$. With identical households,
# individual and aggregate choices coincide, so the limit is the same in every iteration.
#
# **The loop.** Given tomorrow's marginal utility $M(b', y')$ on the grid:
# 1. *Expectations.* $E[M(b', y') \mid y^T] = \sum_{y'} P(y^T, y') M(b', y')$: the expected marginal value of
#    one more unit of wealth tomorrow, for every candidate $b'$.
# 2. *Euler residual.* At each $(b, y^T)$, $F(b') = u_T(b + y^T - q b') - \beta R\, E[M(b', y') \mid y^T]$.
#    $F$ rises with $b'$: saving more lowers $c^T$ today and raises today's marginal utility.
# 3. *Corner first.* If $F \ge 0$ already at the limit, households would like to borrow even more but cannot:
#    the constraint binds, $b' = b'_{lim}$, and $\mu = F(b'_{lim})$ is its shadow cost.
# 4. *Interior otherwise.* Find the $b'$ with $F(b') = 0$ by bisection; $\mu = 0$.
# 5. *Update.* Compute $c^T$ from the budget constraint, set $M = u_T(c^T)$, and go back to 1 until $M$ stops
#    changing.
#
# **Why this is the market outcome.** In step 5, $M$ is just private marginal utility. Households ignore that
# borrowing more today tightens everybody's constraint tomorrow through a lower $p^N$. The planner (B3) runs the
# same loop but adds $\mu\psi$ to $M$: that term is the pecuniary externality.
#
# Once it converges, the policy $b'(b, y^T)$ is all we need to simulate the economy (B2).

# %%
def solve(planner, tax=None, tol=1e-10, damp=0.5, maxit=5000):
    """Time iteration. planner=False: competitive equilibrium. planner=True: B3.
    tax: a tax on debt for each (b', y), used in B5."""
    M = uT(np.maximum(B[:, None] + yT[None, :] - q * B[:, None], 0.05))      # initial guess
    for it in range(maxit):
        EM = M @ P.T                    # EM[i, j] = E[M(B[i], y') | y = yT[j]]
        if tax is not None:
            EM = EM * (1 + tax)         # the tax raises the cost of borrowing
        bp = np.empty((NB, NY))
        mu = np.empty((NB, NY))
        for j in range(NY):

            def F(x):                   # Euler residual at candidate b' = x, for all b on the grid
                return uT(np.maximum(B + yT[j] - q * x, 1e-10)) - beta * R * np.interp(x, B, EM[:, j])

            F_lim = F(bp_lim[:, j])
            hi = np.minimum((B + yT[j]) / q - 1e-6, B[-1])
            root = bisect(F, bp_lim[:, j].copy(), hi, n=45)
            binding = F_lim >= 0
            bp[:, j] = np.where(binding, bp_lim[:, j], root)
            mu[:, j] = np.where(binding, F_lim, 0.0)
        cT = B[:, None] + yT[None, :] - q * bp
        if planner:
            mu = np.maximum(mu / (1 - psi(cT)), 0.0)
            Mn = uT(cT) + mu * psi(cT)
        else:
            Mn = uT(cT)
        dist = np.max(np.abs(np.log(Mn) - np.log(M)))
        M = np.exp(damp * np.log(Mn) + (1 - damp) * np.log(M)) if planner else Mn
        if dist < tol:
            break
    return dict(bp=bp, cT=cT, mu=mu, M=M, it=it)


ce = solve(planner=False)
print(f"CE converged in {ce['it']} iterations")

lab = {0: "low $y^T$", NY // 2: "mean $y^T$", NY - 1: "high $y^T$"}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for j, c in zip(lab, [RED, BLUE, GREEN]):
    ax[0].plot(B, ce["bp"][:, j], color=c, label=lab[j])
    ax[0].plot(B, bp_lim[:, j], ":", color=c, lw=1)
    ax[1].plot(B, ce["cT"][:, j], color=c, label=lab[j])
ax[0].plot(B, B, color=GREY, lw=0.8)
ax[0].set(title="Bond policy $b'(b, y^T)$ (dotted: limit)", xlabel="$b$", xlim=(-1.05, -0.3), ylim=(-1.05, -0.3))
ax[1].set(title="Tradable consumption $c^T(b, y^T)$", xlabel="$b$", xlim=(-1.05, -0.3))
ax[0].legend(frameon=False)
show(fig, "b1_policy")

# %% [markdown]
# ## B2. Simulation and the definition of a crisis
# Draw 100,000 years of $y^T$ from the Markov chain, then follow $b_{t+1} = b'(b_t, y^T_t)$.
# A **sudden stop**: the constraint binds *and* the current account (change in net foreign assets, % of GDP) is
# more than one standard deviation above its mean. GDP: $y^T + p^N y^N$, in units of tradables.

# %%
T, BURN = 100_000, 1000
rng = np.random.default_rng(3)
draws = rng.random(T + BURN)
cumP = np.cumsum(P, axis=1)
s = np.empty(T + BURN, dtype=int)            # index of the y^T state
s[0] = NY // 2
for t in range(1, T + BURN):
    s[t] = min(np.searchsorted(cumP[s[t - 1]], draws[t]), NY - 1)


def simulate(sol):
    b = np.empty(T + BURN + 1)
    b[0] = -0.8
    for t in range(T + BURN):
        b[t + 1] = np.interp(b[t], B, sol["bp"][:, s[t]])
    sim = pd.DataFrame({"b": b[BURN:-1], "bp": b[BURN + 1:], "s": s[BURN:]})
    sim["yT"] = yT[sim.s]
    sim["cT"] = sim.b + sim.yT - q * sim.bp
    sim["p"] = price(sim.cT)
    sim["gdp"] = sim.yT + sim.p * yN
    sim["lim"] = [np.interp(bt, B, bp_lim[:, st]) for bt, st in zip(sim.b, sim.s)]
    sim["bind"] = sim.bp <= sim.lim + 1e-6
    sim["ca"] = 100 * (sim.bp - sim.b) / sim.gdp
    sim["crisis"] = sim.bind & (sim.ca > sim.ca.mean() + sim.ca.std())
    sim["debt_gdp"] = -100 * sim.b / sim.gdp
    return sim


sim_ce = simulate(ce)
print(f"CE: debt/GDP {sim_ce.debt_gdp.mean():.1f}%, constraint binds {100 * sim_ce.bind.mean():.1f}% "
      f"of years, sudden stops {100 * sim_ce.crisis.mean():.2f}% of years")

K = 5


def window(sim, var):
    idx = np.flatnonzero(sim.crisis.values)
    idx = idx[(idx >= K) & (idx < len(sim) - K)]
    x = sim[var].values
    return np.array([x[idx + k].mean() for k in range(-K, K + 1)])


def event_panel(sims, labels, colors, name):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    ks = np.arange(-K, K + 1)
    for sim, labl, col in zip(sims, labels, colors):
        ax[0].plot(ks, 100 * (window(sim, "yT") - 1), "o-", ms=3, color=col, label=labl)
        ax[1].plot(ks, 100 * np.log(window(sim, "cT")), "o-", ms=3, color=col)
        ax[2].plot(ks, 100 * np.log(window(sim, "p") / sim.p.mean()), "o-", ms=3, color=col)
        ax[3].plot(ks, window(sim, "ca"), "o-", ms=3, color=col)
    for a, ttl in zip(ax, ["$y^T$, % from mean", "$c^T$, log x 100", "$p^N$, % from mean",
                           "current account, % GDP"]):
        a.axvline(0, color=GREY, lw=0.8)
        a.set(title=ttl, xlabel="years from crisis")
    ax[0].legend(frameon=False)
    show(fig, name)


cr = sim_ce[sim_ce.crisis]
print(f"In a CE sudden stop: c^T falls {100 * (1 - cr.cT.mean() / sim_ce.cT.mean()):.1f}% below its mean, "
      f"p^N {100 * (1 - cr.p.mean() / sim_ce.p.mean()):.1f}% below; current account {cr.ca.mean():.1f}% of GDP")

# %% [markdown]
# ## B3. The planner (constrained efficient)
# The planner faces the same constraint but internalises that $p^N$ rises with aggregate $c^T$. With
# $\psi = \kappa\, \frac{\partial p^N}{\partial c^T}\, y^N$, its Euler equation is
# $$u_T(c^T) = \beta R\, E\left[u_T(c^{T\prime}) + \mu'\psi'\right] + \mu\,(1 - \psi).$$
# Same `solve()`: $M$ now includes $\mu\psi$, and in binding states $\mu$ is divided by $1-\psi$. The planner's
# iteration is damped (`damp=0.5`): without it the binding region flips back and forth between iterations.

# %%
sp = solve(planner=True)
print(f"Planner converged in {sp['it']} iterations")
jL = 2                                               # y^T 8% below its mean (about -1.4 s.d.)
con = sp["bp"] <= bp_lim + 1e-12                     # states where the planner's constraint binds


def shade(ax, mask, color, label):
    ax.fill_between(B, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color=color, alpha=0.15, lw=0,
                    label=label)


fig, ax = plt.subplots(figsize=(6.5, 4.2))
shade(ax, con[:, jL], GREY, "constrained region")
ax.plot(B, ce["bp"][:, jL], color=RED, label="market (CE)")
ax.plot(B, sp["bp"][:, jL], "--", color=BLUE, label="planner")
ax.set(title=f"Bond policy $b'(b)$ when $y^T$ is {100 * (1 - yT[jL]):.0f}% below its mean",
       xlabel="current bond holdings $b$", ylabel="next period's bond holdings $b'$",
       xlim=(-1.05, -0.55), ylim=(-1.0, -0.72))
ax.legend(frameon=False, loc="lower right")
show(fig, "b3_planner")

# %% [markdown]
# ## B4. Market vs planner
# Same shocks, planner's policy.

# %%
sim_sp = simulate(sp)
fig, ax = plt.subplots(figsize=(6, 3.6))
bins = np.linspace(*np.percentile(np.r_[sim_ce.debt_gdp, sim_sp.debt_gdp], [0.1, 99.9]), 50)
ax.hist(sim_ce.debt_gdp, bins=bins, color=RED, alpha=0.5, density=True, label="CE")
ax.hist(sim_sp.debt_gdp, bins=bins, color=BLUE, alpha=0.5, density=True, label="planner")
ax.set_yscale("log")
ax.set(title="Ergodic distribution of debt (log scale)", xlabel="debt / GDP, %", ylabel="density")
ax.legend(frameon=False)
show(fig, "b4_dist")
event_panel([sim_ce, sim_sp], ["CE", "planner"], [RED, BLUE], "b4_event")

# %% [markdown]
# ## B5. The optimal tax on debt, and welfare
# A tax $\tau$ on borrowing, rebated lump sum, changes the CE Euler equation to
# $u_T = \beta R (1+\tau)\, E u_T' + \mu$. It reproduces the planner when
# $$\tau(b, y) = \frac{E\left[\mu'\psi'\right]}{E\left[u_T(c^{T\prime})\right]}
#   \quad \text{at the planner's } b'.$$
# Where the planner is constrained the market is constrained too (it wants to borrow even more), so
# $\tau = 0$ there, as in Bianchi's code. The check: give households the tax schedule $\tau(b', y)$, solve
# the market again with the same `solve()`, and compare its $b'$ with the planner's.
# Welfare: value functions from $V = u(c) + \beta E V(b', y')$ (policy evaluation), and
# the permanent consumption increase that makes the CE as good as the planner,
# $\lambda = (V^{SP}/V^{CE})^{1/(1-\sigma)} - 1$, averaged over the CE's ergodic distribution.

# %%
EX = (sp["mu"] * psi(sp["cT"])) @ P.T          # E[mu' psi' | y] on the b' grid
EU = uT(sp["cT"]) @ P.T                       # E[u_T(c^T') | y] on the b' grid
tax_bp = EX / EU                              # the wedge, for every b' on the grid and every y
tau = np.column_stack([np.interp(sp["bp"][:, j], B, tax_bp[:, j]) for j in range(NY)])   # at the planner's b'
tau[con] = 0.0                                 # planner constrained: the market is constrained too, no tax needed

# Check: households facing this tax (and getting it back lump sum) choose exactly the planner's b'.
tx = solve(planner=False, tax=tax_bp)
print(f"Market with the tax vs planner: max |b' difference| = {np.abs(tx['bp'] - sp['bp']).max():.1e}")


def value(sol, tol=1e-10):
    flow = u(sol["cT"])
    V = flow / (1 - beta)
    while True:
        EV = V @ P.T
        Vn = np.column_stack([flow[:, j] + beta * np.interp(sol["bp"][:, j], B, EV[:, j]) for j in range(NY)])
        if np.max(np.abs(Vn - V)) < tol:
            return Vn
        V = Vn


V_ce, V_sp = value(ce), value(sp)
gain = (V_sp / V_ce) ** (1 / (1 - sigma)) - 1
tau_sim = np.array([np.interp(bt, B, tau[:, st]) for bt, st in zip(sim_sp.b, sim_sp.s)])
gain_sim = np.array([np.interp(bt, B, gain[:, st]) for bt, st in zip(sim_ce.b, sim_ce.s)])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
shade(ax[0], con[:, jL], GREY, "constrained")
shade(ax[0], (tau[:, jL] > 1e-4) & ~con[:, jL], RED, "tax region")
ax[0].plot(B, 100 * tau[:, jL], color=BLUE)
ax[0].set(title=f"Optimal tax on debt, $y^T$ {100 * (1 - yT[jL]):.0f}% below mean", ylabel="%",
          xlabel="current bond holdings $b$", xlim=(-1.05, -0.55))
ax[0].legend(frameon=False, loc="upper right")
ax[1].hist(100 * tau_sim, bins=50, color=BLUE, alpha=0.8, density=True)
ax[1].set(title="Tax in the planner's ergodic distribution", xlabel="$\\tau$, %")
show(fig, "b5_tax")

# %% [markdown]
# ## Summary: market vs planner

# %%
def moments(sim):
    cr = sim[sim.crisis]
    return {"debt / GDP, %": sim.debt_gdp.mean(),
            "constraint binds, % of years": 100 * sim.bind.mean(),
            "sudden stops, % of years": 100 * sim.crisis.mean(),
            "c^T in a crisis, % below mean": 100 * (1 - cr.cT.mean() / sim.cT.mean()),
            "p^N in a crisis, % below mean": 100 * (1 - cr.p.mean() / sim.p.mean()),
            "current account in a crisis, % GDP": cr.ca.mean()}


table = pd.DataFrame({"CE": moments(sim_ce), "planner": moments(sim_sp)})
print(table.round(2))
print(f"Mean tax on debt {100 * tau_sim.mean():.2f}% (s.d. {100 * tau_sim.std():.2f}); "
      f"welfare gain {100 * gain_sim.mean():.3f}% of consumption")
