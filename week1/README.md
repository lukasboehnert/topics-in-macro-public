# Week 1 Lab: Local Projections vs VARs

The lab is the lecture's running example: what happens to UK output after the 1-year interest rate rises by 1pp? Every number and figure on the slides comes from these two scripts.

## Files

- `week1_simple.py`: the whole lecture, one `# %%` cell per slide. Each cell estimates one step and draws the figure on that slide. Every local projection is a regression formula on one DataFrame of leads and lags (`smf.ols("y_h12 ~ i + y + u + y_l1 + ...", data=df)` with statsmodels, `IV2SLS.from_formula("y_h12 ~ 1 + ... + [i ~ z]", ...)` with linearmodels), and the VARs are iterated by hand. Read this one first. This is the code style the exercises use. Its figures are saved to `figures_simple/`.
- `week1_fast.py`: the same cells, specifications and numbers, written to run fast (about 15 seconds instead of a minute): one numpy LP function that still names its variables like a formula (`lp("y", "i", ["y", "u"] + lag_terms(["y", "u", "i"], 12))`), companion-form VARs, and the bootstrap, Monte Carlo and FEVD bias correction run on all 500 samples at once. Its figures are saved to `figures/`.
- `exercise_PT_student.py`: the in-class exercise (below). The solution, `exercise_PT.py`, is posted after class.
- `data/fetch_data_PT.py`: downloads the data for the exercise (see Data).

The two lecture scripts give the same estimates. Only the bootstrap bands and the Monte Carlo use random draws, so those differ slightly.

## Data

- `CTV_DATA.xlsx` (in `data/`): UK monthly data from Cesa-Bianchi, Thwaites and Vicondoa (2020), for the lecture scripts. We use output (`gdp`), unemployment (`unempl`), the 1-year gilt yield (`i_1YR`) and the Cloyne and Hürtgen (2016) narrative monetary policy shock (`cloyne`), 1979m7 to 2009m2 (T = 356). Missing values are coded 123456789.
- `pt_monthly.csv`, `pt_gdp_quarterly.csv` (in `data/`): Portuguese data for the exercise, made by `data/fetch_data_PT.py`. It downloads industrial production, HICP and unemployment from Eurostat, the Euribor and the deposit facility rate from the ECB, and the Jarociński–Karadi (2020) shocks from [github.com/marekjarocinski/jkshocks_update_ecb](https://github.com/marekjarocinski/jkshocks_update_ecb). You don't need to run it: the `check(...)` values in the exercise are computed from the data in the repository, and newer data may be revised. To see how the data are built, or to get the latest months for your own work:

  ```
  cd week1/data
  uv run fetch_data_PT.py
  ```

## Setup

See [Getting started](../README.md#getting-started) in the course README. Then run any script from this folder:

```
cd week1
uv run week1_simple.py
uv run week1_fast.py
uv run exercise_PT_student.py
```

Or open the script in VS Code, pick the `.venv` interpreter that `uv sync` created, and run it cell by cell (Shift+Enter): each figure appears under its cell. Run as a plain script, the figures are only saved to disk.

## In class (about 25 minutes)

Run `week1_simple.py` cell by cell alongside the slides. LPs first (L1–L6), then VARs (V0–V5), then the comparison (C1–C4).

| Cell | Slide | Try this |
| --- | --- | --- |
| Data | Running example | Which series trend? Why does that matter for L1? |
| L1 | No controls | Add a linear trend as a control. What happens to the −6%? |
| L2 | Lags and date-t controls | Drop `u` from the date-t controls. Change 12 lags to 4 and 24. |
| L3 | Narrative LP-IV | Leave lags of `z` out of the controls. Look at unemployment instead of output. |
| L4 | Lags with an instrument | Compare the standard errors with and without lags. |
| L5 | Standard errors | Compare HC and Newey-West at 12, 24 and 36 months. |
| L6 | LP variance decomposition | Use the narrative shock instead of the implicit shock. Why is the share smaller? |
| V0 | Information criteria | Change `pmax` to 36. Does the choice move? |
| Identification | Regress one surprise on the other | Check that γ̂ and the residual s.d. give the Cholesky factor. |
| V1 | Recursive VAR | Add 68% bands next to the 90% ones. How wide are they at 3 years? |
| V2 | Ordering | Put unemployment last. What changes, and why so little? |
| V3 | Lag length | Which p would you report? What did V0 say? |
| Proxy SVAR | 2SLS on the VAR surprises | Why does output rise on impact? |
| V4 | Instrument in a VAR | Compare the impact columns from the external and the internal instrument. |
| V5 | VAR variance decomposition | How much do unemployment forecast errors owe to the policy shock? |
| C1 | LP vs VAR by lag length | Up to which horizon do they agree for each p, and why? |
| C2 | One instrument, two estimators | Where do LP-IV and the internal-instrument VAR differ, and why? |
| C3 | Bias vs variance | Replace the VAR(2) and LP(2) by 4 lags. What happens to the bias? |
| C4 | Variance decompositions | How large is the bias correction at 12, 24 and 48 months? |

## Take-home (due at the start of Week 2)

**Part 5. Bias versus variance (cell C3).** Treat the VAR(12) as the truth, simulate 500 samples, and estimate the output response with a VAR(2) and an LP with 2 lags. Plot bias and standard deviation by horizon. In half a page, explain the pattern using the bias-variance results from the lecture. Why is the LP(2) not unbiased here?

**Part 6. Non-invertibility (cell C5).** For y_t = η_t + 2η_{t−1}, compare an AR(12) with an LP on the observed η_t at h = 0 to 3. Explain why the AR gets h = 1 wrong, and what this implies for SVARs with news or anticipated shocks.

**Part 7. Variance decompositions (cells L6, V5, C4).** How much of the forecast error variance of output does the policy shock explain at 12, 24 and 48 months, in the VAR and with the LP R² method of Gorodnichenko and Lee (2020)? Why do the two differ at 1 to 2 years but not at 4?

**Extension.** Add a price index to the VAR. Does the price puzzle appear?

Hand in your code and two pages of figures and text.

## In-class exercise: the ECB and Portugal (about 25 minutes)

Same tools, new question: what does an ECB monetary policy shock do to Portugal? The shocks are Jarociński and Karadi's (2020) high-frequency ECB surprises, split into a pure policy shock (MP: rates up, stocks down) and a central bank information shock (CBI: rates up, stocks up). They are a weak instrument for the 1-year Euribor, so we use them directly as observed shocks: an LP of each outcome on the shock, with lags as controls.

`exercise_PT_student.py` covers IP, HICP, unemployment and the Euribor after an MP shock, a CBI shock, and with the poor man's sign split. Some lines are left for you to write (`# TODO` lines with `...`): build the leads, write the shock-test and LP formulas, store the coefficient and standard error, and call the LP function for the CBI and poor man's shocks. A `check(...)` line after each gap prints ✓ when your result matches the solution. Figures go to `figures_exercise/`.

Is the shock unpredictable? Do MP and CBI move output and prices in opposite directions? Why does the Euribor fall a few months after an MP tightening?
