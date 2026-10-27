# %% [markdown]
# # In-class exercise, part 1: the ECB and Portugal
#
# **Question.** What happens to Portuguese industrial production, prices and unemployment after an ECB monetary policy shock? And does it matter whether the ECB surprises markets with a pure policy move or with news about the economy?
#
# **Shocks.** Jarociński and Karadi (2020, *AEJ: Macro*), updated. They split the high-frequency surprises around ECB announcements in two, using the co-movement of interest rates and stock prices:
#
# - **MP** (monetary policy): rates up, stocks down. A pure tightening.
# - **CBI** (central bank information): rates up, stocks up. The ECB revealed that the economy is stronger than markets thought.
#
# Two ways to do the split, both in the data:
#
# - **median**: the median of all rotations that satisfy the sign restrictions;
# - **poor man's sign**: each announcement goes entirely to MP if the rate and stock surprises have opposite signs, entirely to CBI if they have the same sign.
#
# **Method.** Unlike the UK narrative shock, the JK shock is a weak instrument for the 1-year Euribor (first-stage F around 1), so LP-IV is unreliable. We use the shock directly as an **observed shock**: an LP of each outcome on the shock, with lags as controls. The code is the same as Part A of `week1_simple.py`: one DataFrame with leads and lags, one formula per horizon.
#
# **Data.** `data/pt_monthly.csv` (made by `data/fetch_data_PT.py`), 2000m1 to 2019m12.
#
# **Run** cell by cell in VS Code, or `uv run --with numpy,pandas,matplotlib,statsmodels exercise_PT.py`


# %% [markdown]
# ## Setup
# Horizons $h = 0, \dots, 36$ months, $P = 6$ lags as controls (six variables with 12 lags would be 72 regressors on 240 months), sample 2000–2019 (before COVID).

# %%
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:                                       # VS Code cells / Jupyter: figures appear under each cell
    get_ipython()  # noqa: F821
    INTERACTIVE = True
except NameError:                          # plain script: only save the figures
    INTERACTIVE = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures_exercise"
FIGS.mkdir(exist_ok=True)
H = 36                                     # horizons 0..H (months)
hs = np.arange(H + 1)
P = 6                                      # lags as controls: 6 variables x 12 lags would be 72
                                           # regressors on 240 months, so we use 6 (bias vs variance)
START, END = "2000-01", "2019-12"          # pre-COVID: 2020 brings the lockdown collapse

plt.rcParams.update({"figure.dpi": 110, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, GREEN, GREY = "#1f4e79", "#c0504d", "#4f8f5b", "#7f7f7f"


def band(ax, r, label, color, ls="-"):
    """Response r[:, 0] with a 90% band r[:, 0] +/- 1.645 r[:, 1]."""
    ax.fill_between(hs, r[:, 0] - 1.645 * r[:, 1], r[:, 0] + 1.645 * r[:, 1], color=color, alpha=0.13)
    ax.plot(hs, r[:, 0], color=color, ls=ls, label=label)


def show(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.png")
    plt.show() if INTERACTIVE else plt.close(fig)


# %% [markdown]
# ## 0. A first look at the raw data
# Before estimating anything: what is in `data/pt_monthly.csv`? One row per month, one column per series:
#
# | column | what it is | unit |
# | --- | --- | --- |
# | `ip` | industrial production, Portugal, industry excl. construction (Eurostat `sts_inpr_m`), seasonally adjusted | index |
# | `hicp` | HICP all items, Portugal (Eurostat `prc_hicp_midx`), **not** seasonally adjusted | index |
# | `hicp_sa` | `hicp` seasonally adjusted by us (average log change of each calendar month removed) | index |
# | `core_sa` | HICP excluding energy, food, alcohol and tobacco, adjusted the same way | index |
# | `unemp` | unemployment rate, seasonally adjusted (Eurostat `une_rt_m`) | % of labour force |
# | `euribor1y`, `euribor3m` | 1-year and 3-month Euribor, monthly average (ECB) | % |
# | `dfr` | ECB deposit facility rate, end of month (ECB) | % |
# | `mp_median`, `cbi_median` | Jarociński–Karadi MP and CBI shocks, median rotation | pp, monthly sum |
# | `mp_pm`, `cbi_pm` | the same, poor man's sign split | pp, monthly sum |
# | `pc1` | first principal component of the high-frequency rate surprises (before the split) | pp |
#
# The euro-area series (Euribor, deposit rate, shocks) are the same for every country; only `ip`, the HICPs and `unemp` are Portuguese. The shocks are summed over the ECB announcements in a month, so they are zero in a month without one.

# %%
pd.set_option("display.width", 160, "display.max_columns", 20)   # print all columns
data = pd.read_csv(HERE / "data" / "pt_monthly.csv", parse_dates=["date"], index_col="date")
print(f"{data.shape[0]} months x {data.shape[1]} columns, {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}\n")
print(data.head(), "\n")                                # the first rows
print(data.tail(), "\n")                                # the last rows
print(data.describe().T.round(2), "\n")                 # mean, sd, min, max per column

# When does each series start and end, and how many months are missing?
coverage = pd.DataFrame({"first": data.apply(lambda c: c.first_valid_index()).dt.strftime("%Y-%m"),
                         "last": data.apply(lambda c: c.last_valid_index()).dt.strftime("%Y-%m"),
                         "missing": data.isna().sum()})
print(coverage)

# How sparse are the shocks? Share of months with a non-zero shock, and the largest ones.
print(f"\nNon-zero MP shock in {(data.mp_median.fillna(0) != 0).mean():.0%} of months")
print("Largest MP shocks (pp):")
print(data.mp_median.abs().nlargest(5).index.strftime("%Y-%m").tolist(),
      data.mp_median.loc[data.mp_median.abs().nlargest(5).index].round(3).tolist())

fig, axes = plt.subplots(2, 3, figsize=(12, 5.5))
for ax, (c, title) in zip(axes.flat, [("ip", "Industrial production (index)"),
                                      ("hicp", "HICP, raw and seasonally adjusted (index)"),
                                      ("unemp", "Unemployment (%)"),
                                      ("euribor1y", "1-year Euribor and deposit rate (%)"),
                                      ("mp_median", "MP shock, median (pp)"),
                                      ("mp_pm", "MP shock, poor man's (pp)")]):
    if c.startswith("mp"):
        ax.bar(data.index, data[c], width=25, color=BLUE)
    else:
        ax.plot(data.index, data[c], color=BLUE)
    ax.set_title(title, fontsize=9)
axes[0, 1].plot(data.index, data.hicp_sa, color=RED, lw=0.8)
axes[1, 0].plot(data.index, data.dfr, color=RED, lw=0.8)
show(fig, "pt_raw")
# Things to notice: the level shift in IP in 2005-2008, COVID in 2020, the
# unemployment peak in the sovereign debt crisis, negative rates after 2014, and
# how small the shocks are (a few basis points). Our sample is 2000-2019.


# %% [markdown]
# ## 1. Data: one DataFrame with leads and lags
# Outcomes: 1-year Euribor `e1y`, industrial production `ip` and HICP `p` (100 × log), unemployment `u`. Shocks are divided by their standard deviation, so every response is to a **one-standard-deviation shock**. Leads `ip_h12` $= ip_{t+12}$, lags `ip_l1` $= ip_{t-1}$.

# %%
raw = pd.read_csv(HERE / "data" / "pt_monthly.csv", parse_dates=["date"], index_col="date")
raw = raw.loc[START:END]
df = pd.DataFrame({"e1y": raw.euribor1y,                  # 1-year Euribor, %
                   "ip": 100 * np.log(raw.ip),            # industrial production, 100 x log
                   "p": 100 * np.log(raw.hicp_sa),        # HICP, 100 x log
                   "u": raw.unemp})                       # unemployment rate, %
for s in ["mp_median", "cbi_median", "mp_pm", "cbi_pm"]:
    df[s] = raw[s] / raw[s].std()                         # one-standard-deviation shocks
OUTCOMES = {"e1y": "1-year Euribor (pp)", "ip": "Industrial production (%)",
            "p": "HICP (%)", "u": "Unemployment (pp)"}
lags_ = {f"{v}_l{l}": df[v].shift(l) for l in range(1, 13) for v in df.columns}   # x_{t-l}
leads_ = {f"{v}_h{h}": df[v].shift(-h) for h in hs for v in OUTCOMES}             # x_{t+h}
df = pd.concat([df, pd.DataFrame(lags_), pd.DataFrame(leads_)], axis=1)


def lag_terms(variables, p):
    """The controls as formula text: lag_terms(["ip", "u"], 2) = 'ip_l1 + u_l1 + ip_l2 + u_l2'."""
    return " + ".join(f"{v}_l{l}" for l in range(1, p + 1) for v in variables)


print(f"Sample {df.index[0]:%Y-%m} to {df.index[-1]:%Y-%m}, T = {len(df)}")
print("Shock standard deviations (pp):", raw[["mp_median", "cbi_median", "mp_pm", "cbi_pm"]].std().round(3).to_dict())

fig, ax = plt.subplots(1, 3, figsize=(11, 3))
ax[0].plot(df.index, df.ip, color=BLUE)
ax[0].set_title("Industrial production (100 x log)")
ax[1].plot(df.index, df.e1y, color=BLUE, label="1-year Euribor")
ax[1].plot(df.index, df.u, color=RED, label="unemployment")
ax[1].set_title("Rates (%)")
ax[1].legend(frameon=False, fontsize=8)
ax[2].bar(df.index, raw.mp_median, width=25, color=GREEN, label="MP")
ax[2].bar(df.index, raw.cbi_median, width=25, color=GREY, alpha=0.7, label="CBI")
ax[2].set_title("Jarocinski-Karadi shocks, median (pp)")
ax[2].legend(frameon=False, fontsize=8)
show(fig, "pt_data")

# %% [markdown]
# ## 2. Is the shock a shock?
# An observed shock must be unpredictable from the past. Regress it on $P$ lags of everything and test that all coefficients are zero (robust F-test).

# %%
controls = lag_terms(["e1y", "ip", "p", "u", "mp_median", "cbi_median"], P)
for s in ["mp_median", "cbi_median"]:
    res = smf.ols(f"{s} ~ {controls}", data=df).fit(cov_type="HC1")
    print(f"{s}: R2 on {P} lags of everything = {res.rsquared:.2f}, "
          f"F = {res.fvalue:.2f}, p-value = {res.f_pvalue:.2f}")

# %% [markdown]
# ## 3. LP on the MP shock
# $$x_{t+h} = a_h + b_h\, s_t + \sum_{l=1}^{P} c_{h,l}' w_{t-l} + e_{t+h}, \qquad w = (\text{Euribor, IP, HICP, unemployment, MP, CBI})$$
# The lags make HC standard errors valid (lag augmentation, slide L5). First one outcome with the loop written out, then the same loop as a function for all four.

# %%
rows = []
for h in hs:
    res = smf.ols(f"ip_h{h} ~ mp_median + {controls}", data=df).fit(cov_type="HC1")
    rows.append({"h": h, "b": res.params["mp_median"], "se": res.bse["mp_median"]})
print(pd.DataFrame(rows).iloc[[0, 6, 12, 24, 36]].round(2))


# ... then the same loop as a function, for every outcome and shock:
def run_lp(dep, shock, rhs):
    """OLS of {dep}_h{h} on the formula rhs for h = 0..H. Returns an (H+1) x 2
    array: the coefficient on the shock and its standard error."""
    out = np.zeros((H + 1, 2))
    for h in hs:
        res = smf.ols(f"{dep}_h{h} ~ {rhs}", data=df).fit(cov_type="HC1")
        out[h] = res.params[shock], res.bse[shock]
    return out


def lp_panel(shock, controls, color, label, axes=None, ls="-"):
    """LPs of the four outcomes on `shock`, drawn on a 2 x 2 grid."""
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    irf = {}
    for ax, (v, title) in zip(axes.flat, OUTCOMES.items()):
        irf[v] = run_lp(v, shock, f"{shock} + {controls}")
        band(ax, irf[v], label, color, ls)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set(title=title, xlabel="months after the shock")
    axes[0, 0].legend(frameon=False, fontsize=8)
    return irf, axes


irf_mp, axes = lp_panel("mp_median", controls, BLUE, "MP: rates up, stocks down")
axes[0, 0].figure.suptitle("Portugal: one-sd ECB monetary policy (MP) shock", fontsize=11)
show(axes[0, 0].figure, "pt_mp")
for v, r in irf_mp.items():
    print(f"MP shock, {OUTCOMES[v]:28s} h = 0/12/24/36:", r[[0, 12, 24, 36], 0].round(2))
# Discuss: output and prices fall, as they should. But the Euribor *falls* after
# a few months. Why might that be? (Hint: what does the ECB do when output and
# inflation fall? And how small is a one-sd shock, in basis points?)

# %% [markdown]
# ## 4. LP on the CBI shock
# Same LP, information shock. If the CBI shock is news that the economy is stronger, output and prices should move the other way from an MP tightening.

# %%
irf_cbi, axes = lp_panel("cbi_median", controls, RED, "CBI: rates up, stocks up")
axes[0, 0].figure.suptitle("Portugal: one-sd ECB information (CBI) shock", fontsize=11)
show(axes[0, 0].figure, "pt_cbi")

_, axes = lp_panel("mp_median", controls, BLUE, "MP")
lp_panel("cbi_median", controls, RED, "CBI", axes=axes, ls="--")
axes[0, 0].figure.suptitle("Portugal: pure policy (MP) versus information (CBI) shocks", fontsize=11)
show(axes[0, 0].figure, "pt_mp_cbi")

# %% [markdown]
# ## 5. Poor man's sign restrictions
# Same LPs with the poor man's split (`mp_pm`, `cbi_pm`); the controls use the lags of the poor man's shocks.

# %%
controls_pm = lag_terms(["e1y", "ip", "p", "u", "mp_pm", "cbi_pm"], P)
_, axes = lp_panel("mp_pm", controls_pm, BLUE, "MP, poor man's sign")
lp_panel("cbi_pm", controls_pm, RED, "CBI, poor man's sign", axes=axes, ls="--")
axes[0, 0].figure.suptitle("Portugal: poor man's sign restrictions", fontsize=11)
show(axes[0, 0].figure, "pt_poor_man")
# Compare with cell 4. Which split gives the cleaner picture, and why might the
# median rotation and the poor man's split differ? (Hint: what does the poor
# man's split do with an announcement where rates and stocks barely move?)

# %% [markdown]
# ## 6. Your turn
# Change one thing at a time and rerun cells 3–5:
#
# - a) Extend the sample to 2025 (`END`). What does 2020–2022 do to the bands?
# - b) Use 12 lags (`P`), as in the lecture. What happens to the bands, and why?
# - c) Replace headline HICP by core inflation (`core_sa` in the data). Is the price response clearer?
# - d) Newey–West instead of lag augmentation: `.fit(cov_type="HAC", cov_kwds={"maxlags": h + 4})`. Do the bands change much?
# - e) LP-IV: instrument the 1-year Euribor with the MP shock, as in L3 of `week1_simple.py` (linearmodels, `[e1y ~ mp_median]`). What is the first-stage F, and what does that do to the bands?
