# Week 3 Lab: Financial Crises, Collateral and Macroprudential Policy

Topics in Macroeconomics, PhD elective. Two parts. **Part A (sudden stops):** a small open economy that borrows abroad against the market value of an asset, d' ≤ κ q, the collateral constraint of Mendoza (2010) stripped down to an endowment economy. We solve it under perfect foresight and with uncertainty, simulate it, and measure amplification. **Part B (overborrowing):** Bianchi (2011), where the price in the constraint is the relative price of nontradables. We solve the market and the constrained-efficient planner, compare crises, and compute the optimal tax on debt. Every number and figure on the slides comes from these scripts.

## Files

- `week3_simple.py`, `week3_fast.py`: Part A, cells F0 and M0–M4 (M2, Euler equation errors, is a side check that is not on the slides). **Time iteration**: for every state, solve the Euler equation for tomorrow's debt by bisection; if that violates d' ≤ κ q(d'), solve d' = κ q(d') instead; update consumption, the asset price and the multiplier; repeat. The simple version loops over the nine income states and spells everything out (about 15 seconds); the fast version is vectorised and writes the slide figures to `figures/` (about 10 seconds). Same output.
- `overborrowing_simple.py`, `overborrowing_fast.py`: Part B, cells B0–B5. The simple version uses time iteration with bisection, the fast version the endogenous grid method; same output, a few seconds each.
- `overborrowing_Bianchi.py`: Part B the way Bianchi wrote it, a line-by-line port of his replication code (`main.m`, AER 2011) with the same sections, variable names and loops, to read side by side with the original. Tauchen shocks and Bianchi's grid, so numbers differ slightly from the scripts above; the economics is identical. About a minute.
- `exercise_SS_student.py`: the in-class exercise on Part A's model (below). The solution, `exercise_SS.py`, is posted after class.
- `data/fetch_jst.py`: downloads the Jordà-Schularick-Taylor Macrohistory database (`JSTdatasetR6.xlsx`) used in F0, a credit-boom event study (Mendoza and Terrones 2012). Run it once: `uv run data/fetch_jst.py`. Without the file, the scripts skip F0. F0 is not on the slides.

## Part A: the model

    max E sum beta^t c^(1-sigma)/(1-sigma)
    c_t = A_t + d_{t+1} - (1+r) d_t              (the asset is in fixed supply k = 1; it pays alpha A_t)
    d_{t+1} <= kappa q_t                          (collateral: the asset held today)
    u'(c_t) = beta (1+r) E u'(c_{t+1}) + mu_t
    q_t u'(c_t) = beta E[ u'(c_{t+1}) (alpha A_{t+1} + q_{t+1}) + kappa mu_{t+1} q_{t+1} ]

Calibration (annual): sigma = 2, beta = 0.91, r = 4%, alpha = 0.3, kappa = 0.14; log A AR(1) with rho = 0.85, sd = 0.05, 9-state Rouwenhorst chain. M0 is the perfect-foresight version (log utility, beta(1+r) = 1, kappa = 0.10, d0 = 0.6); its figure shows borrowing, the asset price and consumption against the size of the output loss.

**Multiple equilibria.** More borrowing raises consumption today and so the asset price, so d' ≤ κ q(d') can hold at several values of d' (Schmitt-Grohé and Uribe 2021). The solver takes the largest debt consistent with the constraint. With a tax on debt (exercise, Part 2 and the take-home) time iteration can end in a two-cycle at a few grid points; `model()` then averages the last two iterates and says so (moments move by less than 0.5 pp).

## Part B: Bianchi (2011)

    max E sum beta^t c^(1-sigma)/(1-sigma),   c = [omega cT^(-eta) + (1-omega) cN^(-eta)]^(-1/eta)
    q b' + cT + pN cN = b + yT + pN yN,       q b' >= -kappa (yT + pN yN),     cN = yN
    pN = (1-omega)/omega (cT/yN)^(1+eta)
    market:   uT(cT) = beta R E uT(cT') + mu
    planner:  uT(cT) + mu psi = beta R E[uT(cT') + mu' psi'] + mu,      psi = kappa yN dpN/dcT
    tax:      tau = E[mu' psi'] / E[uT(cT')]  at the planner's b'

Calibration from Bianchi's replication code: r = 4%, beta = 0.906, elasticity 1/(1+eta) = 0.83, sigma = 1 + eta (separable), omega = 0.307, kappa = 0.3235, yN = 1; log yT AR(1) with rho = 0.5 and s.d. 5.8%, 9 Rouwenhorst states (Bianchi's code uses Tauchen).

## Setup

See [Getting started](../README.md#getting-started) in the course README. Then from this folder:

```
uv run week3_simple.py
uv run overborrowing_simple.py
uv run exercise_SS_student.py
```

Or open a script in VS Code, pick the `.venv` interpreter, and run it cell by cell (Shift+Enter).

## In class (about 20 minutes)

Run the simple scripts cell by cell alongside the slides.

| Cell | Slide | Try this |
| --- | --- | --- |
| M0 | Small vs large shocks | Set d0 = 0.5. Where is the threshold now? Then kappa = 0.12. |
| M1 | Competitive equilibrium | Plot the policy functions for kappa = 0.10. Where is the cliff? |
| M3 | Simulation | Define a crisis as "constraint binds" only. How often does it happen? |
| M4 | Anatomy | Raise the shock threshold to 2 s.d. Does the SS/NSS gap grow? |
| B3 | Planner | Lower kappa to 0.28. Does the planner save more or less than the market? |
| B4 | Market vs planner | Double the volatility of yT. What happens to the gap in crisis frequency? |
| B5 | Tax on debt | Plot the tax against yT at a fixed debt level. Is it higher in good or bad times? |

## In-class exercise: your turn (about 30 minutes)

`exercise_SS_student.py`, five parts on Part A's model. Part 1 is the perfect-foresight model on paper and in three lines of code (the threshold and the quadratic for c1/c2). Part 2: write the asset-pricing equation inside `model(...)`, reproduce the moments and the gain from a simple 2.5% tax on debt. Part 3: vary κ. Part 4: value the collateral at a fixed price; the constraint then binds almost always, there are no Fisherian collapses, and the tax only costs welfare: without a price in the constraint there is no externality. Part 5: one experiment of their own. Gaps are `# TODO` lines with `...`; a `check(...)` line after each gap prints ✓ when the result matches.

## Take-home (due at the start of Week 4)

**Part 1. The size of the externality.** In Bianchi's model, compute the mean optimal tax, the welfare gain and the crisis frequency (market and planner) for κ ∈ {0.28, 0.32, 0.36} and an s.d. of yT of 4%, 5.8% and 8%. Where is the case for macroprudential policy strongest? Why is the welfare gain small even when crises fall by two thirds?

**Part 2. A simple rule.** Replace the optimal, state-contingent tax by a constant tax on debt in the market economy (add `(1 + tau)` to the Euler equation). Which constant tax does best, and how much of the planner's welfare gain does it achieve?

**Part 3. Back to the data.** Run F0 for the euro periphery (PRT, ESP, ITA, GRC are in JST) and for the rest. Are booms in the periphery larger, and do they end in crises more often? Relate the event windows to M4.

**Extension.** Portugal in 2011 had no nominal exchange rate. Which price played the role of p^N or q in Portugal's collateral constraints (think about housing and bank funding), and how would you write the model to capture it?

Hand in your code and two pages of figures and text.
