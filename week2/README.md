# Week 2 Lab: Shift-Share IV and Event Studies

Topics in Macroeconomics, PhD elective. The lab is the lecture's running example: how much manufacturing employment did US local labour markets lose because of import competition from China? Every number and figure on the slides comes from these two scripts.

## Files

- `week2_simple.py`: the whole lecture, one `# %%` cell per slide step (S0–S7, E0–E2). Every regression is a formula on a DataFrame (`IV2SLS.from_formula("d_sh_empl_mfg ~ 1 + t2 + [shock ~ z]", df, weights=df.weights)` with linearmodels, `smf.ols(...)` with statsmodels); the instrument and the shock-level data are pandas merges and group-bys. Read this one first. This is the code style the exercise uses. Its figures are saved to `figures_simple/`. About a minute (the placebo is 200 draws of three regressions).
- `week2_fast.py`: the same cells, specifications and numbers in numpy: the CZ × shock share matrix `Sh` (1444 × 794), `z = Sh @ g`, the shock-level aggregation as one matrix product, and the placebo with 1000 draws at once. It writes the figures the slides use to `figures/`. About 5 seconds.
- `exercise_ADH_student.py`: the in-class exercise (below). The solution, `exercise_ADH.py`, is posted after class.
- `data/`: the ADH files from the Goldsmith-Pinkham, Sorkin and Swift (2020) replication package ([github.com/paulgp/bartik-weight](https://github.com/paulgp/bartik-weight)).

The two lecture scripts give the same estimates. Only the placebo (S4) uses random draws (200 vs 1000), so its rejection rates differ slightly. The event-study simulation uses its own seed and is identical in both.

## Data

- `ADHdata_AKM.csv`: 722 commuting zones × 2 periods (`t2` = 0 for 1990–2000, 1 for 2000–2007). Outcomes `d_sh_empl_mfg`, `d_sh_empl_nmfg`, `d_sh_empl` (change in manufacturing, non-manufacturing and total employment / working-age population, pp per decade); treatment `shock` (change in US imports from China per worker, $1000); ADH's instrument `IV`; population weights `weights`; `statefip`, `czone`, `division`; ADH's controls (`l_shind_manuf_cbp`, `l_sh_popedu_c`, `l_sh_popfborn`, `l_sh_empl_f`, `l_sh_routine33`, `l_task_outsource`).
- `Lshares.dta`: `czone`, `year` (1990 / 2000: the start of each period; the shares themselves are lagged, as in ADH), `sic87dd` (397 manufacturing industries), `ind_share`. They sum to the CZ's manufacturing share, `S`, not to one.
- `shocks.dta`: `year`, `sic87dd`, `g` (growth of imports from China in eight other high-income countries, $1000 per US worker), `g_importsUSA` (the same for the US), and the eight countries separately.
- `sic_code_desc.dta`: industry names.

Check: `z = Σ_n s_ln g_n` reproduces the `IV` column to 1e-6.

## Setup

See [Getting started](../README.md#getting-started) in the course README. Then run any script from this folder:

```
cd week2
uv run week2_simple.py
uv run week2_fast.py
uv run exercise_ADH_student.py
```

Or open the script in VS Code, pick the `.venv` interpreter, and run it cell by cell (Shift+Enter).

## In class (about 25 minutes)

Run `week2_simple.py` cell by cell alongside the slides.

| Cell | Slide | Try this |
| --- | --- | --- |
| S0 | Running example | Which industries have the largest shocks? In which period? |
| S1 | OLS | Add ADH's controls. Why does OLS move towards zero? |
| S2 | The ADH instrument | Drop the weights. Compare with ADH's unweighted number. |
| S3 | Incomplete shares | Replace `S_90 + S_00` by `S` alone. Why is that not enough? |
| S4 | Standard errors | Cluster the shock-level IV by industry (`sic87dd`) instead of SIC3. |
| S4 | Placebo | Draw the placebo shocks with a common mean of 1 and drop `S_90 + S_00`. What happens to the rejection rates? |
| S5 | Shock-level view | Compute the effective number of shocks for each period separately. |
| S6 | Rotemberg weights | How much weight do the 1990s shocks get in total? |
| S7 | Balance | Add ADH's controls to the balance regressions (except the variable on the left). |
| E1 | Static TWFE | Add a never-treated share of 50%. Do the negative weights go away? |
| E2 | Event studies | Drop the never-treated units. What happens to the TWFE event study, and to LP-DiD at long horizons? |

## In-class exercise: your turn (about 25 minutes)

`exercise_ADH_student.py`: six short parts on the same data, each changing one thing. Build the instrument yourself; total and non-manufacturing employment (where did the jobs go?); US imports instead of other countries' imports as the shift; exposure-robust SEs; leaving out the computer industry (the largest Rotemberg weight); and, in the simulation, when static TWFE is fine. Gaps are `# TODO` lines with `...`; a `check(...)` line after each gap prints ✓ when the result matches the solution.

## Take-home (due at the start of Week 3)

**Part 1. The two checklists.** Write one page applying the Borusyak, Hull and Jaravel (2025) checklist for exogenous shifts to the China shock: the ideal experiment, the gap between observed and ideal shifts, the incomplete-share control, lagged shares, shock-level descriptives (cells S5, S6), balance (S7) and exposure-robust inference (S4). Which step is the weakest?

**Part 2. Spillovers.** Using the preferred specification, estimate the effect on non-manufacturing and total employment with exposure-robust SEs (exercise Part 2 and 4). Compare with the manufacturing estimate and discuss what the local-labour-markets design does and does not capture (aggregate effects, the "missing intercept").

**Part 3. Event study.** In the simulation, make the effect constant over time but different across cohorts (c = 1.5, 1, 0.5, slope 0). Is static TWFE still biased? Compare the TWFE event study, the imputation estimator and LP-DiD, and explain the differences.

**Extension.** A shift-share design for Portugal or the euro area: pick an outcome, the exposure shares and the shifts (e.g. Portuguese regions' export exposure to sectors hit by Chinese competition, or regions' exposure to the ECB's rate changes through sectoral debt shares). Write down the shift-level experiment and which of the two identification routes you rely on.

Hand in your code and two pages of figures and text.
