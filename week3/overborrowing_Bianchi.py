# %% [markdown]
# # Week 3 lab, part B: Bianchi (2011) the way Bianchi wrote it
#
# A line-by-line Python port of Javier Bianchi's replication code (`main.m`, "Overborrowing and Systemic
# Externalities in the Business Cycle", AER 2011). Same sections, same variable names, same loops, so you can
# put the two files side by side. `overborrowing_simple.py` solves the same model in a different way (bisection
# on b', a 9-state Rouwenhorst chain, a finer grid); the economics is identical.
#
# | Section | What | `main.m` |
# | --- | --- | --- |
# | 1 | Parameters | 21-39 |
# | 2 | Shocks (Tauchen), grid, initial guess | 41-111 |
# | 3 | Decentralized equilibrium (the market) | 114-222 |
# | 4 | Social planner | 223-332 |
# | 5 | Welfare | 333-368 |
# | 6 | Optimal tax on debt | 370-384 |
# | 7 | Simulation and crises | 398-584 |
#
# **The idea of the algorithm.** Guess consumption c(b, y) and next period's bonds b'(b, y) on the grid. Then
# repeat until nothing changes:
# 1. expected marginal utility tomorrow, at the guessed b';
# 2. if consuming at the borrowing limit leaves marginal utility *above* that, the household would like to
#    borrow more than it can: the constraint **binds**, consume at the limit;
# 3. otherwise solve the Euler equation for c. Utility is separable in T and N (sigma = 1 + ita, Bianchi's
#    `sep_prefs = 1`), so this is one line: c = (emu/omega)^(-1/sigma);
# 4. update the price of nontradables from the new c, and with it the borrowing limit;
# 5. move the guess only a little toward the new values (damping, `uptd`).
#
# **Timing.** Bianchi writes the budget as c^T + b' = (1+r) b + y^T and the constraint as b' >= -kappa (p^N y^N + y^T)
# (b < 0 is debt). The simple script writes q b' + c^T = b + y^T; the two b's differ by the factor 1+r.
#
# **Run** cell by cell in VS Code (Shift+Enter), or top to bottom with `uv run overborrowing_Bianchi.py`
# (about a minute: the planner is updated slowly, like in the original).

# %% [markdown]
# ## 0. Housekeeping

# %%
from math import erfc, sqrt
from pathlib import Path

import matplotlib
import numpy as np

try:                                       # VS Code cells / Jupyter: figures appear under each cell
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:                          # plain script: only save the figures
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures_bianchi"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
DE_COL, SP_COL = "#c0504d", "#1f4e79"     # decentralized equilibrium red, social planner blue

# %% [markdown]
# ## 1. Model parameter values

# %%
ita = (1 / 0.83) - 1       # elasticity parameter (elasticity of substitution T/N is 0.83)
sigma = ita + 1            # risk aversion. sigma = 1 + ita makes utility separable in T and N, so that
                           # marginal utility of tradables is simply u_T(c) = omega c^(-sigma)
beta = 0.906               # discount factor
omega = 0.3070             # share for tradables
kappa = 0.3235             # collateral: share of income that can be pledged
r = 0.04
R = 1 + r

print(f"MODEL PARAMETERS: sigma {sigma:.1f}  elasticity {1 / (1 + ita):.2f}  beta {beta:.2f}  "
      f"omega {omega:.2f}  kappa {kappa:.2f}  r {r}")

# %% [markdown]
# ## 2. Numerical parameters
# log y^T follows an AR(1) with persistence 0.5 and s.d. 5.8%. Tauchen's method turns it into a 9-state Markov
# chain: `yT[j]` is income in state j and `Prob[j, k]` the probability of moving from state j to state k.

# %%
def cdf_normal(x):
    return 0.5 * erfc(-x / sqrt(2))


def tauchen(N, mu, rho, sigma, m):
    """Markov chain for z' = (1-rho) mu + rho z + eps, eps ~ N(0, sigma^2), with N nodes spanning +- m s.d."""
    Z = np.zeros(N)
    Zprob = np.zeros((N, N))
    a = (1 - rho) * mu

    Z[N - 1] = m * sqrt(sigma**2 / (1 - rho**2))
    Z[0] = -Z[N - 1]
    zstep = (Z[N - 1] - Z[0]) / (N - 1)
    for i in range(1, N - 1):
        Z[i] = Z[0] + zstep * i
    Z = Z + a / (1 - rho)

    for j in range(N):
        for k in range(N):
            if k == 0:
                Zprob[j, k] = cdf_normal((Z[0] - a - rho * Z[j] + zstep / 2) / sigma)
            elif k == N - 1:
                Zprob[j, k] = 1 - cdf_normal((Z[N - 1] - a - rho * Z[j] - zstep / 2) / sigma)
            else:
                Zprob[j, k] = (cdf_normal((Z[k] - a - rho * Z[j] + zstep / 2) / sigma)
                               - cdf_normal((Z[k] - a - rho * Z[j] - zstep / 2) / sigma))
    return Z, Zprob


yn = 1
sigma_y = 0.058                                  # s.d. of log y^T
rho = 0.5
sigma_eps = sqrt(sigma_y**2 * (1 - rho**2))      # s.d. of the innovation
NSS = 9                                          # number of income states
m = 2
Z, Prob = tauchen(NSS, 0, rho, sigma_eps, m)
yT = np.exp(Z)
yN = np.full(NSS, yn)

NB = 100                                         # grid points for bonds
bmin = -1.02
bmax = -0.2000
B = np.linspace(bmin, bmax, NB)

# Every object below is an NB x NSS matrix: row i is bond holdings B[i], column j is income state j.
SR = np.ones((NB, NSS)) * (1 + r)               # gross interest rate
YT = np.tile(yT, (NB, 1))
YN = np.tile(yN, (NB, 1))
KAPPAS = kappa * np.ones((NB, NSS))
b = np.tile(B[:, None], (1, NSS))

# initialize: guess b' = b (stay put)
bp = b.copy()
c = b * SR + YT - bp
price = (1 - omega) / omega * c**(1 + ita)       # p^N from the household's first-order condition
EE = np.zeros((NB, NSS))                         # Euler equation residual
bmax_collat = -kappa * (price * YN + YT)         # the borrowing limit on b'
cbind = b * SR + YT - bmax_collat                # consumption if the constraint binds
emu = np.full((NB, NSS), np.nan)                 # expected marginal utility tomorrow
Ibind = np.ones((NB, NSS))                       # 1 where the constraint binds

uptd = 0.2          # damping, market (change to 0.1 if policy functions fluctuate)
uptdSP = 0.02       # damping, planner
outfreq = 50        # display frequency
iter_tol = 50000    # maximum number of iterations
Tol_eulerDE = 1.0e-7
Tol_eulerSP = 1.0e-7
tol = 1.0e-5        # numerical tolerance

T = 79000           # number of simulated years

print(f"tol {tol:10.2e}  bmin {bmin:.2f} bmax {bmax:.2f}  NB {NB:3d} NS {NSS:2d}")
print("y^T states:", yT.round(3))

# %% [markdown]
# ## 3. Decentralized equilibrium
# Euler equation of the household, with multiplier mu >= 0 on the constraint:
# $$u_T(c) = \beta R\, E\left[u_T(c')\right] + \mu, \qquad u_T(c) = \omega c^{-\sigma}.$$

# %%
it = 0
d2 = 100
print("DE Iter      Norm         Updt")

while d2 > tol and it < iter_tol:
    oldc = c.copy()
    oldbp = bp.copy()

    mup = omega * c**(-sigma)                    # marginal utility today, mup(b, y)

    # EMU is expected marginal utility tomorrow on today's grid.
    # Emup[:, j] = sum_k Prob[j, k] mup[:, k]: expected marginal utility at every grid point b', given y_j today.
    Emup = mup @ Prob.T
    for i in range(NB):
        for j in range(NSS):
            emu[i, j] = beta * SR[i, j] * np.interp(bp[i, j], B, Emup[:, j])

    for i in range(NB):
        for j in range(NSS):
            EE[i, j] = omega * cbind[i, j]**(-sigma) - emu[i, j]

            if EE[i, j] > Tol_eulerDE:           # BORROWING CONSTRAINT BINDS
                bp[i, j] = bmax_collat[i, j]
                c[i, j] = cbind[i, j]
                Ibind[i, j] = 1
            else:                                # EULER EQUATION
                c[i, j] = (emu[i, j] / omega)**(-1 / sigma)
                EE[i, j] = 0
                Ibind[i, j] = 0

    bp = SR * b + YT - c
    bp = np.minimum(bmax, np.maximum(bp, bmin))

    price = (1 - omega) / omega * (c / YN)**(1 + ita)

    c = SR * b + YT - np.maximum(bp, -KAPPAS * (price * YN + YT))   # update consumption based on update for pN

    d2 = max(np.max(np.abs(c - oldc)), np.max(np.abs(bp - oldbp)))  # metric

    bmax_collat = -KAPPAS * (price * YN + YT)
    bmax_collat = np.minimum(bmax, np.maximum(bmax_collat, bmin))
    cbind = SR * b + YT - bmax_collat
    cbind[cbind < 0] = np.nan

    # ===================== Updating rule. Must be slow, important ==============
    bp = uptd * bp + (1 - uptd) * oldbp
    c = uptd * c + (1 - uptd) * oldc
    # ===========================================================================

    it += 1
    if it % outfreq == 0 or it == 1:
        print(f"{it:3d}          {d2:10.2e}   {uptd:.2f}")

print(f"No. Iterations: {it:3d}  Metric  {d2:10.2e}")

LagrangeDE = EE.copy()
LagrangeDE[Ibind == 0] = 0

bpDE = bp.copy()
cDE = c.copy()
priceDE = price.copy()
IbindDE = Ibind.copy()

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].plot(B, cDE)
ax[0].set(title="c", xlabel="B")
ax[1].plot(B, bpDE)
ax[1].set(title="bond policy", xlabel="B")
ax[2].plot(B, priceDE)
ax[2].set(title="pN", xlabel="B")
fig.tight_layout()
fig.savefig(FIGS / "de_policies.png")
plt.show() if INTERACTIVE else plt.close(fig)

# %% [markdown]
# ## 4. Planner problem
# The planner internalises that p^N, and with it everybody's borrowing limit, rises with c. With
# psi = kappa y^N dp^N/dc, its Euler equation is
# $$u_T(c) + \mu\psi = \beta R\, E\left[u_T(c') + \mu'\psi'\right] + \mu.$$
# Same loop as the market: the only changes are that tomorrow's marginal utility includes mu' psi', and that
# the multiplier mu is updated from the equation above.

# %%
bp = bpDE.copy()
c = cDE.copy()
psi = (1 - omega) / omega * (ita + 1) * KAPPAS * (c / YN)**ita
price = priceDE.copy()
LagrangeSP = LagrangeDE / (1 - psi)              # guess for SP
IbindSP = IbindDE.copy()

EESP = np.full((NB, NSS), np.nan)
RHS_SP = np.zeros((NB, NSS))

uptd = uptdSP
d2 = 100
it = 0
print("SPP Iter      Norm       Uptd")

while d2 > tol and it < iter_tol * 2:
    oldc = c.copy()
    oldbp = bp.copy()

    mup = omega * c**(-sigma)                    # mup(b, y)

    psi = (1 - omega) / omega * (ita + 1) * KAPPAS * (c / YN)**ita
    LFH_SP = mup + LagrangeSP * psi

    ELFH = LFH_SP @ Prob.T                       # expected u_T + mu psi tomorrow, at every grid point b'
    for i in range(NB):
        for j in range(NSS):
            RHS_SP[i, j] = beta * SR[i, j] * np.interp(bp[i, j], B, ELFH[:, j])   # beta R E[u_T' + mu' psi']

    # THIS IS EQUATION u_T + mu psi = beta R E[u_T' + mu' psi'] + mu.
    # main.m line 262 writes LFH_SP - beta*SR.*RHS_SP, but RHS_SP already contains beta*R (line 258), so the
    # original multiplies by beta*R twice. Here it enters once, as in the equation.
    LagrangeSP = LFH_SP - RHS_SP

    LagrangeSP[IbindSP == 0] = 0

    for i in range(NB):
        for j in range(NSS):
            if LagrangeSP[i, j] >= Tol_eulerSP:   # BORROWING CONSTRAINT BINDS
                c[i, j] = cbind[i, j]
                IbindSP[i, j] = 1
                EESP[i, j] = LagrangeSP[i, j]
            else:                                 # EULER EQUATION
                c[i, j] = (RHS_SP[i, j] / omega)**(-1 / sigma)
                EESP[i, j] = 0
                IbindSP[i, j] = 0

    bp = SR * b + YT - c
    bp = np.minimum(bmax, np.maximum(bp, bmin))
    price = (1 - omega) / omega * (c / YN)**(1 + ita)

    # =============== Check collateral constraint ==============================
    c = SR * b + YT - np.maximum(bp, -KAPPAS * (price * YN + YT))
    price = (1 - omega) / omega * (c / YN)**(1 + ita)
    bp = SR * b + YT - c
    # ==========================================================================

    d2 = np.max(np.abs(c - oldc))

    # ===================== Updating rule. Must be slow, important ==============
    bp = uptd * bp + (1 - uptd) * oldbp
    c = uptd * c + (1 - uptd) * oldc
    # ===========================================================================

    bmax_collat = -KAPPAS * (price * YN + YT)
    cbind = SR * b + YT - bmax_collat

    it += 1
    if it % outfreq == 0 or it == 1:
        print(f"{it:3d}          {d2:10.2e}   {uptd:.2f}")

print(f"No. Iterations: {it:3d}  Metric  {d2:10.2e}")

mupSP = mup.copy()
bpSP = bp.copy()
priceSP = price.copy()
cSP = c.copy()

# %% [markdown]
# ## 5. Welfare calculation
# Value of following each policy forever: V = U + beta E V(b', y'). The welfare gain Wf is the permanent
# percentage increase in consumption that makes a household in the market economy as well off as under the planner.

# %%
def totalc(cT):                                  # inside of the CES aggregator, with c^N = y^N
    return omega * cT**(-ita) + (1 - omega) * YN**(-ita)


V = np.zeros((NB, NSS))
VSP = np.zeros((NB, NSS))

U = (totalc(cDE)**(-1 / ita))**(1 - sigma) / (1 - sigma)
USP = (totalc(cSP)**(-1 / ita))**(1 - sigma) / (1 - sigma)

EV = U.copy()
EVSP = USP.copy()

d3 = 100
it = 0
print("Value Iter     Norm")
while d3 > tol and it < iter_tol:
    EVk = V @ Prob.T                             # expected value tomorrow at every grid point b'
    EVSPk = VSP @ Prob.T
    for i in range(NB):
        for j in range(NSS):
            EV[i, j] = beta * np.interp(bpDE[i, j], B, EVk[:, j])   # EV is expected value tomorrow on today's grid
            EVSP[i, j] = beta * np.interp(bpSP[i, j], B, EVSPk[:, j])

    V_new = U + EV
    VSP_new = USP + EVSP

    with np.errstate(divide="ignore", invalid="ignore"):            # V = 0 in the first iteration
        d3 = max(np.max(np.abs((V_new - V) / V)), np.max(np.abs((VSP_new - VSP) / VSP)))

    it += 1
    V = V_new
    VSP = VSP_new
    if it % outfreq == 0:
        print(f"{it}          {d3:.7f}")

Wf = (VSP / V)**(1 / (1 - sigma)) - 1

# %% [markdown]
# ## 6. Optimal tax
# A tax tau on borrowing turns the household's Euler equation into u_T = (1 + tau) beta R E u_T' + mu. At the
# planner's allocation, 1 + tau = u_T / (beta R E u_T'). No tax where the planner is constrained.

# %%
emuSP = np.zeros((NB, NSS))
EmupSP = mupSP @ Prob.T
for i in range(NB):
    for j in range(NSS):
        emuSP[i, j] = beta * SR[i, j] * np.interp(bpSP[i, j], B, EmupSP[:, j])
tau = mupSP / emuSP - 1

tau[IbindSP == 1] = 0
tau[tau < 0] = 0

# %% [markdown]
# ## 7. Simulation
# Draw T years of income from the Markov chain and follow both economies along the *same* shocks.

# %%
def markov(Prob, n, s0, rng):
    """Simulate n periods of a Markov chain with transition matrix Prob, starting in state s0."""
    cum = np.cumsum(Prob, axis=1)
    S = np.zeros(n, dtype=int)
    S[0] = s0
    X = rng.random(n)
    for t in range(1, n):
        S[t] = min(np.searchsorted(cum[S[t - 1]], X[t]), NSS - 1)
    return S


print("simulating the model")
b0_DE = -0.9443
b0_SP = -0.9443
rng = np.random.default_rng(0)
S_index = markov(Prob, T, 0, rng)

# SHOCKS
ySIM = yT[S_index]
yNSIM = yN[S_index]
RSIM = SR[0, S_index]
KAPPASIM = KAPPAS[0, S_index]

# INITIALIZING VECTORS
bpSIM = np.zeros(T)          # bond holdings at the start of year t (market)
WfSIM = np.zeros(T)
cSIM = np.zeros(T)
bpSP_SIM = np.zeros(T)       # same, planner
cSP_SIM = np.zeros(T)
Tau_SIM = np.zeros(T)

# Initial conditions
bpSIM[0] = b0_DE
bpSP_SIM[0] = b0_SP

for i in range(T):
    bptemp = np.interp(bpSIM[i], B, bpDE[:, S_index[i]])
    cSIM[i] = bpSIM[i] * RSIM[i] + ySIM[i] - bptemp

    WfSIM[i] = np.interp(bpSIM[i], B, Wf[:, S_index[i]])

    bptempSP = np.interp(bpSP_SIM[i], B, bpSP[:, S_index[i]])
    cSP_SIM[i] = bpSP_SIM[i] * RSIM[i] + ySIM[i] - bptempSP
    Tau_SIM[i] = np.interp(bpSP_SIM[i], B, tau[:, S_index[i]])
    if i < T - 1:
        bpSIM[i + 1] = bptemp
        bpSP_SIM[i + 1] = bptempSP

print(f"welfare gain is: {100 * WfSIM.mean():.6f}")

pSIM = (1 - omega) / omega * (cSIM / yNSIM)**(1 + ita)
pSP_SIM = (1 - omega) / omega * (cSP_SIM / yNSIM)**(1 + ita)

# Borrowing limit on b_{t+1}, chosen in year t; shift by one so that entry t is the limit on bpSIM[t].
bplim_SIM0 = -KAPPASIM * (pSIM * yNSIM + ySIM)
bplimSP_SIM0 = -KAPPASIM * (pSP_SIM * yNSIM + ySIM)
bplim_SIM = bplim_SIM0.copy()
bplimSP_SIM = bplimSP_SIM0.copy()
bplim_SIM[1:] = bplim_SIM0[:-1]
bplimSP_SIM[1:] = bplimSP_SIM0[:-1]

# IMPORTANT TO USE A TOLERANCE, NOT <=
bindSIM = ((bpSIM - bplim_SIM) <= 1e-5).astype(int)
bindSP_SIM = ((bpSP_SIM - bplimSP_SIM) <= 1e-5).astype(int)
bindSIM[0] = 0                # not binding in the first period
bindSP_SIM[0] = 0

bpSIM[bindSIM == 1] = bplim_SIM[bindSIM == 1]
bpSP_SIM[bindSP_SIM == 1] = bplimSP_SIM[bindSP_SIM == 1]

CA_SIM = np.zeros(T)          # change in net foreign assets, > 0 is a capital outflow
CASP_SIM = np.zeros(T)
CA_SIM[:-1] = bpSIM[1:] - bpSIM[:-1]
CASP_SIM[:-1] = bpSP_SIM[1:] - bpSP_SIM[:-1]
CA_SIM[-1] = np.interp(bpSIM[-1], B, bpDE[:, S_index[-1]]) - bpSIM[-1]
CASP_SIM[-1] = np.interp(bpSP_SIM[-1], B, bpSP[:, S_index[-1]]) - bpSP_SIM[-1]

Y_SIM = pSIM * yNSIM + ySIM   # income, in units of tradables
YSP_SIM = pSP_SIM * yNSIM + ySIM

CAY_SIM = CA_SIM / Y_SIM      # current account / income
CAYSP_SIM = CASP_SIM / YSP_SIM

Lev_SIM = bpSIM / Y_SIM       # leverage (negative: debt)
LevSP_SIM = bpSP_SIM / YSP_SIM

CtSIM = (omega * cSIM**(-ita) + (1 - omega) * yNSIM**(-ita))**(-1 / ita)      # total consumption
CtSPSIM = (omega * cSP_SIM**(-ita) + (1 - omega) * yNSIM**(-ita))**(-1 / ita)

RER_SIM = (omega**(1 / (1 + ita)) + (1 - omega)**(1 / (1 + ita)) * pSIM**(ita / (1 + ita)))**((1 + ita) / ita)
RERSP_SIM = (omega**(1 / (1 + ita)) + (1 - omega)**(1 / (1 + ita)) * pSP_SIM**(ita / (1 + ita)))**((1 + ita) / ita)

# %% [markdown]
# ### Event analysis
# A **crisis** (sudden stop) in year t: the constraint binds on the choice made in t, *and* the current account
# is more than one standard deviation above its mean. As in `main.m`, the planner's crises use the market's
# threshold, and the planner is compared with the market *on the market's crisis dates*.

# %%
nbd = 3                       # periods before and after a crisis

print("Event analysis")
bindSIM0 = np.zeros(T)
bindSIM0[:-1] = bindSIM[1:]   # entry t: does the choice made in year t bind?
bindSP_SIM0 = np.zeros(T)
bindSP_SIM0[:-1] = bindSP_SIM[1:]

thresh = CA_SIM.mean() + CA_SIM.std(ddof=1)
SS = ((CA_SIM > thresh) & (bindSIM0 > 0)).astype(int)
SS_SP = ((CASP_SIM > thresh) & (bindSP_SIM0 > 0)).astype(int)

print(f"The long-run probability of crisis in DE is: {SS.mean():.4f}")
print(f"The long-run probability of crisis in SP is: {SS_SP.mean():.4f}")
print(f"std of CA/Y DE is: {CAY_SIM.std(ddof=1):.4f}")
print(f"std of CA/Y SP is: {CAYSP_SIM.std(ddof=1):.4f}")
print(f"mean debt/Y level in DE is: {Lev_SIM.mean():.4f}")
print(f"mean debt/Y level in SP is: {LevSP_SIM.mean():.4f}")
print(f"welfare gain is: {100 * WfSIM.mean():.6f}")

Cdev = (CtSIM - CtSIM.mean()) / CtSIM.mean()
CSPdev = (CtSPSIM - CtSPSIM.mean()) / CtSPSIM.mean()
RERdev = (RER_SIM - RER_SIM.mean()) / RER_SIM.mean()
RERSPdev = (RERSP_SIM - RERSP_SIM.mean()) / RERSP_SIM.mean()

SS_Index = np.flatnonzero(SS == 1)
SS_Index = SS_Index[nbd:-nbd]   # drop a few events at the ends so that windows stay inside the sample

print(f"C drop in DE is: {Cdev[SS_Index].mean():.4f}")
print(f"C drop in SP is: {CSPdev[SS_Index].mean():.4f}")
print(f"RER depreciation in DE is: {RERdev[SS_Index].mean():.4f}")
print(f"RER depreciation in SP is: {RERSPdev[SS_Index].mean():.4f}")
print(f"CA/Y | SS in DE is: {CAY_SIM[SS_Index].mean():.4f}")
print(f"CA/Y | SS in SP is: {CAYSP_SIM[SS_Index].mean():.4f}")
print(f"mean tax on debt (planner's ergodic distribution): {100 * Tau_SIM.mean():.2f}%")

# %% [markdown]
# ## Figures from the paper
# Income state 3 of 9 (index 2 here, y^T about 6% below its mean). Grey: where the planner is constrained.
# `plotspaper.m` uses state 6, but with this shock process the planner is constrained only in the four
# lowest income states.

# %%
shok = 2
CONSTRAINTp = (LagrangeSP >= 1e-2).astype(int)
ktightp = bpSP / (-KAPPAS * (priceSP * YN + YT))   # planner's debt relative to its limit
grey = CONSTRAINTp[:, shok] == 1


def shade(ax):
    ax.fill_between(B, 0, 1, where=grey, transform=ax.get_xaxis_transform(), color="0.8", lw=0)


fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))
shade(ax[0, 0])
ax[0, 0].plot(B, bpDE[:, shok], color=DE_COL, lw=2.5, label="Decentralized Equilibrium")
ax[0, 0].plot(B, bpSP[:, shok], "-.", color=SP_COL, lw=2, label="Social Planner")
ax[0, 0].plot(B, B, color="k", lw=0.5)
ax[0, 0].set(title="Bond policy b'(b)", xlabel="Current Bond Holdings", xlim=(-1.05, -0.6), ylim=(-1.05, -0.6))
ax[0, 0].legend(frameon=False)

bins = np.linspace(-1.02, -0.6, 60)
ax[0, 1].hist(bpSIM, bins=bins, density=True, color=DE_COL, alpha=0.5, label="Decentralized Equilibrium")
ax[0, 1].hist(bpSP_SIM, bins=bins, density=True, color=SP_COL, alpha=0.5, label="Social Planner")
ax[0, 1].set(title="Ergodic distribution of bonds", xlabel="Bond Holdings")
ax[0, 1].legend(frameon=False)

shade(ax[1, 0])
ax[1, 0].plot(B, 100 * tau[:, shok], color=SP_COL, lw=2.5)
ax[1, 0].set(title="Implied Tax on Debt", xlabel="Current Bond Holdings", ylabel="Percentage", xlim=(-1.05, -0.6))

shade(ax[1, 1])
ax[1, 1].plot(B, 100 * Wf[:, shok], color=SP_COL, lw=2.5)
ax[1, 1].set(title="Welfare gain from the planner", xlabel="Current Bond Holdings",
             ylabel="% of permanent consumption", xlim=(-1.05, -0.6))
fig.tight_layout()
fig.savefig(FIGS / "bianchi_figures.png")
plt.show() if INTERACTIVE else plt.close(fig)
