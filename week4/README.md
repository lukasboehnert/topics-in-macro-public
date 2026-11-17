# Week 4 Lab: Currency Unions and Heterogeneous Households

Topics in Macroeconomics, PhD elective. The lab follows the lecture in two parts. **B0–B2:** a two-region currency union with sticky prices (Benigno 2004): what the union cannot do after an asymmetric shock, and which inflation rate the central bank should target when one region's prices are stickier. **N0–N2:** a union of countries that differ in inequality (Boehnert, de Ferra, Mitman and Romei 2025): non-homothetic demand for services, a two-agent, two-sector, two-period model of a rate rise, and the trade imbalances it creates between an equal and an unequal member. Linear models are solved with our own rational-expectations solver (QZ and the Blanchard-Kahn check, Klein 2000), not Dynare. 

## Files

- `week4_simple.py`: one `# %%` cell per step (B0–B2, N0–N2), with the model written out above each cell. In B0–B2 every linear equation is one line of coefficients, `({next-period terms}, {today's terms})`, stacked into `A E_t w_{t+1} = B w_t`. N0–N2 is non-linear and solved with `scipy.optimize.root`. Read this one first. Figures go to `figures_simple/`.
- `week4_fast.py`: the same code without the explanations; writes the slide figures to `figures/`. A few seconds.
- `unequal_union_simple.py`: **the class code for Part 3.** The model of Boehnert, de Ferra, Mitman and Romei (2025) built from scratch in seven steps, in the order of the slides: preferences, incomes and the stationary allocation, the rate rise, the mechanism (checked with the e_R/e_H formula), inequality and transmission, which ingredient matters, and the union of an equal and an unequal country. Reproduces the shape of the paper's Figures 6 to 11, 13 and 14. Figures go to `figures_simple/`; about ten seconds.
- `unequal_union_fast.py`: the same steps without the explanations; figures to `figures/`.

No data.

## B0–B2. A two-region currency union (Benigno 2004)

    piH_t = beta E piH_{t+1} + kH [yW_t + (1 - n) psi tau_t]       (Phillips curve, H)
    piF_t = beta E piF_{t+1} + kF [yW_t - n psi tau_t]             (Phillips curve, F)
    tau_t = tau_{t-1} + piF_t - piH_t - dT~_t                       (terms-of-trade gap; exchange rate fixed)
    delta piH_t + (1 - delta) piF_t = 0                             (central bank's target)

kj = (1 + eta)(1 − αj)(1 − β αj)/αj, psi = (1 + eta θ)/(1 + eta), natural terms of trade T~ = (1 + eta)/(1 + eta θ) aR. Loss per period, % of consumption: (1 + eta)/2 [yW² + n(1 − n) θ psi tau² + n γ/kH piH² + (1 − n) γ/kF piF²] (log utility, efficient steady state). Calibration: β = 0.99, eta = 3, θ = 1, γ = 6, n = 0.5, αH = αF = 0.75; relative productivity AR(1) 0.9, s.d. 1%. Variances from the Lyapunov equation.

**Main numbers.** Relative productivity +1%: H inflation −0.78%, F +0.78% (annualised) on impact, terms-of-trade gap −0.61%, half-life 2 quarters; loss 0.0206% of consumption per period (two currencies: 0). With αH = 0.85 or 0.9 (αF = 0.75), the best δ is 0.69 or 0.80, and the loss falls from 0.0272 to 0.0233, or from 0.0347 to 0.0234, relative to targeting union inflation (δ = n).

## N0–N2. A union of unequal countries (Boehnert, de Ferra, Mitman and Romei 2025)

Each country has a Ricardian (R) and a hand-to-mouth (H) household, tradable goods (T) and non-tradable services (N). Boppart (2014) preferences: goods share ω^T(e) = ν (P^N/e)^ε (P^T/P^N)^γ, so services are a luxury when ε > 0. Technology y^s = Z (l^s)^α_s with α_N = 1, α_T = 0.43; R receives the share κ_s of sector s's income, κ_T = 0.5, κ_N ≥ 0.5 measures inequality. Calibration as in the paper: ν = 0.6, ε = 0.18, γ = 0, labour endowment 0.93, Z set so that the services share is 40% at κ_N = 0.5. Experiment: in period 1 the nominal rate rises from 1% to 2% and nominal wages are fixed; period 2 is the stationary allocation. Then P^N_1 is fixed, y^T_1 = y^T (P^T_1)^(α_T/(1−α_T)), and R's goods consumption falls by 1/(βR^n_1).

**Main numbers.** Identical countries: total output −1.98% with κ_N = 0.5, −1.73% with κ_N = 0.9 (goods −0.98% vs −0.84%; e_R/e_H −0.47%). Cobb-Douglas (ε = 0): −1.49% for every κ_N. Union of κ_N = 0.5 and 0.9: goods price −1.20% (alone −1.30% and −1.11%), goods output −0.90% in both; trade balance +0.05% (equal) and −0.05% (unequal) of goods output; total output −1.86% and −1.83%. Levels differ a little from the paper's figures (normalisation of the stationary allocation); the comparative statics match.

## Setup

See [Getting started](../README.md#getting-started) in the course README. This week needs `scipy`; it is in this folder's `pyproject.toml`. Then from this folder:

```
uv run week4_simple.py
uv run week4_fast.py
```

Or open a script in VS Code, pick the `.venv` interpreter, and run it cell by cell (Shift+Enter).

## In class (about 15 minutes, then the in-class problem)

Run `week4_simple.py` (B1, B2) cell by cell alongside Part 1, and go through `unequal_union_simple.py` step by step alongside Part 3 (about 25 minutes).

| Cell | Slide | Try this |
| --- | --- | --- |
| B1 | A productivity gain in one region | Make F prices more flexible (αF = 0.5). Which region now does the adjusting? |
| B2 | Which inflation to target? | Make H small (n = 0.2). Is the best δ still above n? |
| Step 2 | Incomes: who earns what | Set `eps = 0.3`. How much does the services share of the most unequal country rise? |
| Step 4 | The mechanism | Why does e_R/e_H not move with κ_N = 0.5, although services lose value? |
| Step 6 | Which ingredient matters? | Inequality from goods (κ_T = 0.9): why does the sign flip? |
| Step 7 | Equal and unequal country | Make the shock larger (`R_shock = 1.03`). Do the trade imbalances scale with it? |

**In-class problem: one union, two regions.** (1) Derive the union and relative blocks; show the interest rate cannot affect τ. (2) Portugal in the euro: n = 0.05, αH = 0.85, αF = 0.75; the best δ. (3) N1 with Cobb-Douglas (ε = 0) and with ε = 0.22: which ingredient matters? (4) N2 with κ_N = 0.5 and 0.75: the union's goods price and who runs the deficit.

## Take-home (due at the start of Week 5, with the one-page project proposal)

**Part 1. A fiscal instrument for the union.** Add a regional fiscal instrument to B0, for example government spending in H that shifts H's marginal cost and demand, with a rule g_H,t = −φ τ_t. How much of the union's loss can it remove, and at what φ? Compare with Ferrero (2009).

**Part 2. Where does inequality come from?** Step 6 of `unequal_union_simple.py` shows that a country whose inequality comes from the goods sector (κ_T > 0.5) responds *more* to a rate rise. Extend `period1_union` so that each country has its own κ_T, and put an equal country in a union with such a country. Which country runs the deficit? Explain with the e_R/e_H formula, and say what Fact 3 and the trade-balance projection then tell us about where inequality comes from.

Hand in your code and two pages of figures and text.
