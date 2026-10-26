"""Fetch the data for the Portuguese version of the Week 1 running example.

Question: what happens to Portuguese output and prices after the ECB raises rates?

Writes, next to this file:

* ``pt_monthly.csv``: one row per month, 1999m1 onwards. Columns:
      date        first day of the month
      ip          industrial production, Portugal, B-D (industry excl. construction),
                  volume index, calendar and seasonally adjusted (Eurostat sts_inpr_m, SCA)
      hicp        HICP all items, Portugal, index, NOT seasonally adjusted
                  (Eurostat prc_hicp_midx; Eurostat publishes no SA HICP for Portugal)
      hicp_sa     hicp seasonally adjusted here: the average log change of each
                  calendar month is removed (see seasonal_adjust below)
      core_sa     HICP excluding energy, food, alcohol and tobacco, adjusted the same way
      unemp       unemployment rate, % of labour force, seasonally adjusted (Eurostat une_rt_m, SA)
      euribor1y   1-year Euribor, monthly average, % (ECB FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA)
      euribor3m   3-month Euribor, monthly average, % (ECB FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA)
      dfr         ECB deposit facility rate, end of month, % (ECB FM.D.U2.EUR.4F.KR.DFR.LEV)
      mp_median, mp_pm, cbi_median, cbi_pm, pc1
                  Jarocinski and Karadi (2020) euro area monetary policy (MP) and central
                  bank information (CBI) shocks, monthly sums, downloaded from
                  https://github.com/marekjarocinski/jkshocks_update_ecb

* ``pt_gdp_quarterly.csv``: real GDP, Portugal, chain-linked volumes (2020, EUR million),
  calendar and seasonally adjusted (Eurostat namq_10_gdp, B1GQ, CLV20_MEUR, SCA).
  Monthly GDP does not exist, so the monthly example uses industrial production.

Usage (needs internet; about a minute):

    cd data
    uv run --with eurostat,requests fetch_data_PT.py
"""
from io import StringIO
from pathlib import Path
import re

import eurostat
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
START = "1999-01"
GEO = "PT"


# --------------------------------------------------------------------------- ECB
def ecb_series(key: str) -> pd.Series:
    """One series from the ECB Data Portal (SDMX REST API), indexed by period."""
    flow, rest = key.split(".", 1)
    url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{rest}"
    r = requests.get(url, params={"format": "csvdata", "detail": "dataonly", "startPeriod": START},
                     timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    return pd.Series(df["OBS_VALUE"].values, index=df["TIME_PERIOD"], name=key).astype(float)


def ecb_monthly() -> pd.DataFrame:
    e1y = ecb_series("FM.M.U2.EUR.RT.MM.EURIBOR1YD_.HSTA")
    e3m = ecb_series("FM.M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA")
    out = pd.DataFrame({"euribor1y": e1y, "euribor3m": e3m})
    out.index = pd.PeriodIndex(out.index, freq="M")
    dfr = ecb_series("FM.D.U2.EUR.4F.KR.DFR.LEV")      # daily, only on change dates -> fill forward
    dfr.index = pd.to_datetime(dfr.index)
    dfr = dfr.resample("D").ffill().resample("MS").last()
    dfr.index = dfr.index.to_period("M")
    out["dfr"] = dfr
    return out


# --------------------------------------------------------------------------- Eurostat
def eurostat_monthly(code: str, filters: dict, prefer: dict | None = None) -> pd.Series:
    """One monthly Eurostat series for Portugal. `filters` must pin down every
    dimension except those in `prefer`, where the first available value is taken
    (e.g. the index base year changes over time: I25, I21, I15, ...)."""
    raw = eurostat.get_data_df(code, filter_pars={**filters, "geo": GEO})
    raw = raw.rename(columns={c: "geo" for c in raw.columns if str(c).startswith("geo")})
    for dim, values in (prefer or {}).items():
        have = [v for v in values if v in set(raw[dim])]
        if not have:
            raise ValueError(f"{code}: none of {values} in {dim}; available: {sorted(set(raw[dim]))}")
        raw = raw[raw[dim] == have[0]]
        print(f"  {code}: using {dim} = {have[0]}")
    if len(raw) != 1:
        raise ValueError(f"{code}: expected one series, got {len(raw)} rows:\n{raw.iloc[:, :8]}")
    cols = [c for c in raw.columns if re.fullmatch(r"\d{4}-\d{2}", str(c))]
    s = pd.to_numeric(raw[cols].iloc[0], errors="coerce")
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s[s.index >= pd.Period(START, "M")].dropna()


def seasonal_adjust(index: pd.Series) -> pd.Series:
    """Remove the average log change of each calendar month (a simple, transparent
    seasonal adjustment: monthly-dummy regression on the log change, then re-cumulate).
    Fine for teaching; for research use X-13 or the SA series where one is published."""
    dl = np.log(index).diff()
    seas = dl.groupby(dl.index.month).transform("mean") - dl.mean()
    adj = np.log(index.iloc[0]) + (dl - seas).fillna(0).cumsum()
    return np.exp(adj)


def eurostat_pt() -> pd.DataFrame:
    hicp_units = {"unit": ["I25", "I15", "I05"]}
    ip = eurostat_monthly("sts_inpr_m", {"freq": "M", "indic_bt": "PRD", "nace_r2": "B-D", "s_adj": "SCA"},
                          prefer={"unit": ["I21", "I15"]})
    hicp = eurostat_monthly("prc_hicp_midx", {"freq": "M", "coicop": "CP00"}, prefer=hicp_units)
    core = eurostat_monthly("prc_hicp_midx", {"freq": "M", "coicop": "TOT_X_NRG_FOOD"}, prefer=hicp_units)
    unemp = eurostat_monthly("une_rt_m", {"freq": "M", "s_adj": "SA", "age": "TOTAL", "sex": "T",
                                          "unit": "PC_ACT"})
    return pd.DataFrame({"ip": ip, "hicp": hicp, "hicp_sa": seasonal_adjust(hicp),
                         "core_sa": seasonal_adjust(core), "unemp": unemp})


def eurostat_gdp_q() -> pd.DataFrame:
    raw = eurostat.get_data_df("namq_10_gdp", filter_pars={"freq": "Q", "unit": "CLV20_MEUR", "s_adj": "SCA",
                                                           "na_item": "B1GQ", "geo": GEO})
    cols = [c for c in raw.columns if re.fullmatch(r"\d{4}-Q[1-4]", str(c))]
    s = pd.to_numeric(raw[cols].iloc[0], errors="coerce").dropna()
    q = pd.PeriodIndex(s.index.str.replace("-", ""), freq="Q")
    return pd.DataFrame({"quarter": q.astype(str), "date": q.to_timestamp(), "gdp": s.values})


# --------------------------------------------------------------------------- JK shocks
JK_URL = ("https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_ecb/main/"
          "shocks_ecb_mpd_me_m.csv")


def jk_shocks() -> pd.DataFrame:
    jk = pd.read_csv(JK_URL)
    jk.index = pd.PeriodIndex(pd.to_datetime(dict(year=jk.year, month=jk.month, day=1)), freq="M")
    return jk.rename(columns=str.lower)[["mp_median", "mp_pm", "cbi_median", "cbi_pm", "pc1_hf"]] \
             .rename(columns={"pc1_hf": "pc1"})


# --------------------------------------------------------------------------- main
def main() -> None:
    print("ECB ...")
    ecb = ecb_monthly()
    print("Eurostat ...")
    pt = eurostat_pt()
    jk = jk_shocks()

    m = pt.join(ecb, how="outer").join(jk, how="left")
    m = m[m.index >= pd.Period(START, "M")]
    m.insert(0, "date", m.index.to_timestamp())
    m.to_csv(HERE / "pt_monthly.csv", index=False, float_format="%.6g")

    q = eurostat_gdp_q()
    q.to_csv(HERE / "pt_gdp_quarterly.csv", index=False)

    full = m.dropna(subset=["ip", "hicp_sa", "unemp", "euribor1y", "mp_median"])
    print(f"pt_monthly.csv: {len(m)} months, {m.date.min():%Y-%m} to {m.date.max():%Y-%m}; "
          f"all of ip, hicp_sa, unemp, euribor1y, mp_median available "
          f"{full.date.min():%Y-%m} to {full.date.max():%Y-%m} (T = {len(full)})")
    print(m.drop(columns="date").describe().T[["count", "mean", "std", "min", "max"]].round(2))
    print(f"pt_gdp_quarterly.csv: {len(q)} quarters, {q.quarter.iloc[0]} to {q.quarter.iloc[-1]}")


if __name__ == "__main__":
    main()
