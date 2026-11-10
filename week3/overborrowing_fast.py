# %% [markdown]
# # Week 3 lab, part B (fast version): overborrowing and the planner (Bianchi 2011)
#
# The same cells, calibration and numbers as `overborrowing_simple.py`, written for speed: the household problem
# is solved with the **endogenous grid method** (EGM, Carroll 2006) instead of time iteration with bisection,
# and everything that does not need a loop is vectorised. Every number agrees with the simple script to the
# digits printed. Writes the slide figures to `figures/`. A few seconds.
#
# Cells: B0 the model and the externality, B1 competitive equilibrium, B2 simulation and crises, B3 the planner,
# B4 market vs planner, B5 the optimal tax and welfare.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path
import time

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
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY, ORANGE = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f", "#d08a2c"


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


T0 = time.time()

# %% [markdown]
# ## B0. The model, and the externality
# Small open economy, one-period bond $b$ (assets; $b<0$ is debt) at world rate $r$, price $q=1/(1+r)$.
#
# $$\max E_0 \sum_t \beta^t \frac{c_t^{1-\sigma}}{1-\sigma}, \qquad
#   c = \left[\omega (c^T)^{-\eta} + (1-\omega)(c^N)^{-\eta}\right]^{-1/\eta}$$
# $$q\, b' + c^T + p^N c^N = b + y^T + p^N y^N, \qquad
#   q\, b' \ge -\kappa \left(y^T + p^N y^N\right)$$
#
# Market clearing $c^N = y^N$ and the household's FOC give the relative price of nontradables (the inverse of
# the real exchange rate): $p^N = \frac{1-\omega}{\omega}\left(c^T / y^N\right)^{1+\eta}$.
# **The externality:** collateral depends on $p^N$, and $p^N$ on aggregate $c^T$. Each household takes $p^N$ as
# given; the planner does not.
#
# **Calibration** (annual), from Bianchi's replication code: $r = 4\%$, $\beta = 0.906$, elasticity of
# substitution $1/(1+\eta) = 0.83$, $\sigma = 1 + \eta$ (utility separable in $T$ and $N$), $\omega = 0.307$,
# $\kappa = 0.3235$, $y^N = 1$; $\log y^T$ AR(1) with $\rho = 0.5$ and s.d. 5.8\%, 9 Rouwenhorst states.

# %%
R, beta, eta, omega, kappa = 1.04, 0.906, 1 / 0.83 - 1, 0.307, 0.3235
sigma = 1 + eta                                     # separable in T and N, as in Bianchi's code
q, yN, A = 1 / R, 1.0, (1 - omega) / omega
rho, sd_e, NY, NB = 0.5, 0.058 * np.sqrt(1 - 0.5**2), 9, 800


def rouwenhorst(n, rho, sd):
    """Markov chain for an AR(1) with persistence rho and innovation s.d. sd (Kopecky and Suen 2010)."""
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
yT = np.exp(logy)
B = np.linspace(-1.4, 0.0, NB)                       # grid for bond holdings b (b < 0: debt)
bb, YY = np.meshgrid(B, yT, indexing="ij")          # (NB, NY) arrays of the state (b, y^T)


def cagg(cT):
    return (omega * cT**-eta + (1 - omega) * yN**-eta) ** (-1 / eta)


def u(cT):
    return cagg(cT) ** (1 - sigma) / (1 - sigma)


def uT(cT):                                          # marginal utility of tradable consumption
    return omega * cagg(cT) ** (1 + eta - sigma) * cT ** (-1 - eta)


def price(cT):                                       # p^N from the nontradables market
    return A * (cT / yN) ** (1 + eta)


def psi(cT):                                         # kappa * d(p^N y^N)/dc^T: what the planner internalises
    return kappa * (1 + eta) * price(cT) * yN / cT


_c = np.exp(np.linspace(np.log(1e-3), np.log(20), 20000))
_lu = np.log(uT(_c))                                 # uT is decreasing: invert it with a table


def uT_inv(m):
    return np.exp(np.interp(-np.log(m), -_lu, np.log(_c)))


# The borrowing limit: the lowest b' with q b' + kappa (y^T + p^N(c^T) y^N) >= 0, c^T = b + y^T - q b'.
# p^N depends on c^T, so the limit depends on b' itself: solve for it by bisection, once.
def slack(bp, b, y):
    cT = np.maximum(b + y - q * bp, 1e-10)
    return q * bp + kappa * (y + price(cT) * yN)


lo, hi = np.full_like(bb, B[0]), (bb + YY) / q - 1e-8
for _ in range(60):
    mid = (lo + hi) / 2
    ok = slack(mid, bb, YY) >= 0
    lo, hi = np.where(ok, lo, mid), np.where(ok, mid, hi)
bp_lim = np.where(slack(np.full_like(bb, B[0]), bb, YY) >= 0, B[0], hi)

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
# ## B1. Competitive equilibrium, solved with EGM
# Euler equation with multiplier $\mu \ge 0$ on the constraint:
# $u_T(c^T) = \beta (1+r)\, E\, u_T(c^{T\prime}) + \mu$.
#
# EGM step, for every $b'$ on the grid and every $y^T$:
# 1. $EM(b', y) = \sum_{y'} P(y, y') M(b', y')$, with $M = u_T(c^T)$ from the last iteration.
# 2. Unconstrained $c^T = u_T^{-1}(\beta R\, EM)$ and the $b$ that leads there: $b = q b' + c^T - y^T$.
# 3. Interpolate back to the $b$ grid, then impose the limit: $b' = \max(b'_{unc}, b'_{lim})$.

# %%
def solve(planner, tax=None, tol=1e-10, damp=0.5, maxit=5000):
    """planner=False: market (CE). planner=True: B3. tax: a tax on debt for each (b', y), used in B5."""
    M = uT(np.maximum(bb + YY - q * bb, 0.05))
    for it in range(maxit):
        EM = M @ P.T                                             # E[M(b', y') | y], shape (NB, NY)
        if tax is not None:
            EM = EM * (1 + tax)                                  # the tax raises the cost of borrowing
        b_endo = q * B[:, None] + uT_inv(beta * R * EM) - YY      # endogenous grid
        bp_unc = np.column_stack([np.interp(B, b_endo[:, j], B) for j in range(NY)])
        bp = np.minimum(np.maximum(bp_unc, bp_lim), B[-1])
        cT = bb + YY - q * bp
        binding = bp_lim > bp_unc
        EMb = np.column_stack([np.interp(bp[:, j], B, EM[:, j]) for j in range(NY)])
        gap = np.where(binding, uT(cT) - beta * R * EMb, 0.0)    # mu (CE) or mu (1 - psi) (planner)
        if planner:
            mu = np.maximum(gap / (1 - psi(cT)), 0.0)
            Mn = uT(cT) + mu * psi(cT)                           # the planner values the collateral effect
        else:
            mu = np.maximum(gap, 0.0)
            Mn = uT(cT)
        dist = np.max(np.abs(np.log(Mn) - np.log(M)))
        M = np.exp(damp * np.log(Mn) + (1 - damp) * np.log(M)) if planner else Mn
        if dist < tol:
            break
    return dict(bp=bp, cT=cT, mu=mu, binding=binding, M=M, it=it)


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
# Simulate 100,000 years. A **sudden stop**: the constraint binds and the current account (the change in net
# foreign assets, % of GDP) rises by more than one standard deviation above its mean. GDP in tradables:
# $y^T + p^N y^N$.

# %%
T, BURN = 100_000, 1000
rng = np.random.default_rng(3)
draws = rng.random(T + BURN)
cumP = np.cumsum(P, axis=1)
s = np.empty(T + BURN, dtype=int)
s[0] = NY // 2
for t in range(1, T + BURN):
    s[t] = min(np.searchsorted(cumP[s[t - 1]], draws[t]), NY - 1)


def simulate(sol):
    b = np.empty(T + BURN + 1)
    b[0] = -0.8
    for t in range(T + BURN):
        b[t + 1] = np.interp(b[t], B, sol["bp"][:, s[t]])
    b, bp, st = b[BURN:-1], b[BURN + 1:], s[BURN:]
    y = yT[st]
    cT = b + y - q * bp
    p = price(cT)
    gdp = y + p * yN
    lim = np.empty(T)
    for j in range(NY):
        m = st == j
        lim[m] = np.interp(b[m], B, bp_lim[:, j])
    bind = bp <= lim + 1e-6
    ca = 100 * (bp - b) / gdp
    crisis = bind & (ca > ca.mean() + ca.std())
    return pd.DataFrame(dict(b=b, bp=bp, s=st, yT=y, cT=cT, p=p, gdp=gdp, ca=ca, bind=bind, crisis=crisis,
                             debt_gdp=-100 * b / gdp))


sim_ce = simulate(ce)
print(f"CE: debt/GDP {sim_ce.debt_gdp.mean():.1f}%, constraint binds {100 * sim_ce.bind.mean():.1f}% "
      f"of years, sudden stops {100 * sim_ce.crisis.mean():.2f}% of years")

K = 5


def window(sim, var):
    idx = np.flatnonzero(sim.crisis.values)
    idx = idx[(idx >= K) & (idx < len(sim) - K)]
    x = sim[var].values
    return np.array([x[idx + k] for k in range(-K, K + 1)]).mean(axis=1)


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


crisis_rows = sim_ce[sim_ce.crisis]
print(f"In a CE sudden stop: c^T falls {100 * (1 - crisis_rows.cT.mean() / sim_ce.cT.mean()):.1f}% below its "
      f"mean, p^N {100 * (1 - crisis_rows.p.mean() / sim_ce.p.mean()):.1f}% below; current account "
      f"{crisis_rows.ca.mean():.1f}% of GDP")

# %% [markdown]
# ## B3. The planner (constrained efficient)
# The planner chooses $b'$ subject to the *same* constraint but knows that $p^N$ moves with $c^T$. Define
# $\psi = \kappa\, \partial p^N/\partial c^T\, y^N$. The Euler equation becomes
# $$u_T(c^T) = \beta R\, E\left[u_T(c^{T\prime}) + \mu' \psi'\right] + \mu (1 - \psi).$$
# The extra term $\mu'\psi'$: borrowing today raises the chance of a binding constraint tomorrow, when lower
# $c^T$ also lowers everyone's collateral. Same EGM code; $M$ now includes $\mu\psi$.

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
# A tax $\tau$ on borrowing (proceeds rebated lump sum) makes the CE Euler equation
# $u_T = \beta R (1+\tau) E u_T' + \mu$. Matching the planner's gives
# $$\tau(b, y) = \frac{E\left[\mu'\psi'\right]}{E\left[u_T(c^{T\prime})\right]},$$
# evaluated at the planner's choice $b'$. Where the planner is constrained the market is constrained too (it wants to borrow even more), so $\tau = 0$ there, as in Bianchi's code. Implementation: the tax is levied on aggregate borrowing, so each household takes it as given; we pass the schedule $\tau(b', y)$ to the same `solve()` and check that the market then chooses the planner's $b'$. Welfare: the permanent consumption increase that makes a household in
# the CE as well off as under the planner, $\lambda = (V^{SP}/V^{CE})^{1/(1-\sigma)} - 1$, averaged over the
# CE's ergodic distribution.

# %%
EX = (sp["mu"] * psi(sp["cT"])) @ P.T
EU = uT(sp["cT"]) @ P.T
tax_bp = EX / EU                              # the wedge, for every b' on the grid and every y
tau = np.column_stack([np.interp(sp["bp"][:, j], B, tax_bp[:, j]) for j in range(NY)])   # at the planner's b'
tau[con] = 0.0                                 # planner constrained: the market is constrained too, no tax needed

# Check: households facing this tax (and getting it back lump sum) choose exactly the planner's b'.
tx = solve(planner=False, tax=tax_bp)
print(f"Market with the tax vs planner: max |b' difference| = {np.abs(tx['bp'] - sp['bp']).max():.1e}")


def value(sol, tol=1e-10):
    """Policy evaluation: V = u(c) + beta E V(b', y'), with linear interpolation in b'."""
    i = np.clip(np.searchsorted(B, sol["bp"]) - 1, 0, NB - 2)
    w = (sol["bp"] - B[i]) / (B[i + 1] - B[i])
    flow = u(sol["cT"])
    V = flow / (1 - beta)
    while True:
        EV = V @ P.T
        cols = np.arange(NY)[None, :]
        Vn = flow + beta * ((1 - w) * EV[i, cols] + w * EV[i + 1, cols])
        if np.max(np.abs(Vn - V)) < tol:
            return Vn
        V = Vn


V_ce, V_sp = value(ce), value(sp)
gain = (V_sp / V_ce) ** (1 / (1 - sigma)) - 1


def at_sim(sim, X):
    out = np.empty(len(sim))
    for j in range(NY):
        m = sim.s.values == j
        out[m] = np.interp(sim.b.values[m], B, X[:, j])
    return out


tau_sim = at_sim(sim_sp, tau)
gain_sim = at_sim(sim_ce, gain)

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
print(f"Total time {time.time() - T0:.1f} s")
