import argparse
import os
import random
import re
import time
from io import StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm.auto import tqdm

OUTPUT_DIR = "output_sp500"
RATIOS = ["ROE", "PBR", "PER", "ROA", "EPS"]
ANNUAL_START = pd.Timestamp("2022-01-01")
QUARTERLY_START = pd.Timestamp("2024-10-01")


def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    sp500_table = tables[0]

    tickers = sp500_table["Symbol"].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return sorted(set(tickers))


def _to_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = pd.to_datetime(out.columns, errors="coerce")
    out = out.loc[:, out.columns.notna()]
    return out


def _pick_row(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    for name in candidates:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce")
    return pd.Series(dtype="float64")


def _get_close_on_or_before(hist_close: pd.Series, target_date: pd.Timestamp) -> float:
    if hist_close is None or hist_close.empty:
        return np.nan
    s = hist_close.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[s.index <= target_date]
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def _get_shares_on_or_before(shares_series: pd.Series, target_date: pd.Timestamp) -> float:
    if shares_series is None or shares_series.empty:
        return np.nan
    s = shares_series.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[s.index <= target_date]
    if s.empty:
        return np.nan
    return float(s.iloc[-1])


def extract_ratios_for_ticker(
    ticker: str,
    annual_start: pd.Timestamp = ANNUAL_START,
    quarterly_start: pd.Timestamp = QUARTERLY_START,
) -> Tuple[List[Dict], List[Dict]]:
    tk = yf.Ticker(ticker)

    fin_a = _to_timestamp_index(tk.financials)
    bal_a = _to_timestamp_index(tk.balance_sheet)

    fin_q = _to_timestamp_index(tk.quarterly_financials)
    bal_q = _to_timestamp_index(tk.quarterly_balance_sheet)

    hist = tk.history(period="10y", interval="1d", auto_adjust=False)
    close = hist["Close"] if not hist.empty and "Close" in hist.columns else pd.Series(dtype="float64")

    shares_full = tk.get_shares_full(start="2010-01-01")
    if shares_full is None:
        shares_full = pd.Series(dtype="float64")

    net_income_a = _pick_row(fin_a, ["Net Income"])
    equity_a = _pick_row(bal_a, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
    assets_a = _pick_row(bal_a, ["Total Assets"])
    eps_a = _pick_row(fin_a, ["Diluted EPS", "Basic EPS"])

    net_income_q = _pick_row(fin_q, ["Net Income"])
    equity_q = _pick_row(bal_q, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])
    assets_q = _pick_row(bal_q, ["Total Assets"])
    eps_q = _pick_row(fin_q, ["Diluted EPS", "Basic EPS"])

    annual_dates = sorted(list(set(net_income_a.index) | set(equity_a.index) | set(assets_a.index) | set(eps_a.index)))
    annual_dates = [pd.Timestamp(dt) for dt in annual_dates if pd.Timestamp(dt) >= annual_start]

    quarterly_dates = sorted(list(set(net_income_q.index) | set(equity_q.index) | set(assets_q.index) | set(eps_q.index)))
    quarterly_dates = [pd.Timestamp(dt) for dt in quarterly_dates if pd.Timestamp(dt) >= quarterly_start]

    annual_rows: List[Dict] = []
    for dt in annual_dates:
        ni = net_income_a.get(dt, np.nan)
        eq = equity_a.get(dt, np.nan)
        ta = assets_a.get(dt, np.nan)
        eps = eps_a.get(dt, np.nan)

        price = _get_close_on_or_before(close, pd.Timestamp(dt))
        shares = _get_shares_on_or_before(shares_full, pd.Timestamp(dt))

        if pd.isna(eps) and pd.notna(ni) and pd.notna(shares) and shares != 0:
            eps = ni / shares

        roe = ni / eq if pd.notna(ni) and pd.notna(eq) and eq != 0 else np.nan
        roa = ni / ta if pd.notna(ni) and pd.notna(ta) and ta != 0 else np.nan

        pbr = np.nan
        if pd.notna(price) and pd.notna(eq) and pd.notna(shares) and shares != 0 and eq != 0:
            market_cap = price * shares
            pbr = market_cap / eq

        per = price / eps if pd.notna(price) and pd.notna(eps) and eps != 0 else np.nan

        annual_rows.append(
            {
                "Ticker": ticker,
                "Date": pd.Timestamp(dt),
                "ROE": roe,
                "PBR": pbr,
                "PER": per,
                "ROA": roa,
                "EPS": eps,
            }
        )

    quarterly_rows: List[Dict] = []
    for dt in quarterly_dates:
        ni = net_income_q.get(dt, np.nan)
        eq = equity_q.get(dt, np.nan)
        ta = assets_q.get(dt, np.nan)
        eps = eps_q.get(dt, np.nan)

        price = _get_close_on_or_before(close, pd.Timestamp(dt))
        shares = _get_shares_on_or_before(shares_full, pd.Timestamp(dt))

        if pd.isna(eps) and pd.notna(ni) and pd.notna(shares) and shares != 0:
            eps = ni / shares

        roe = ni / eq if pd.notna(ni) and pd.notna(eq) and eq != 0 else np.nan
        roa = ni / ta if pd.notna(ni) and pd.notna(ta) and ta != 0 else np.nan

        pbr = np.nan
        if pd.notna(price) and pd.notna(eq) and pd.notna(shares) and shares != 0 and eq != 0:
            market_cap = price * shares
            pbr = market_cap / eq

        per = price / (eps * 4) if pd.notna(price) and pd.notna(eps) and eps != 0 else np.nan

        quarterly_rows.append(
            {
                "Ticker": ticker,
                "Date": pd.Timestamp(dt),
                "ROE": roe,
                "PBR": pbr,
                "PER": per,
                "ROA": roa,
                "EPS": eps,
            }
        )

    return annual_rows, quarterly_rows


def build_ratio_pivots(df: pd.DataFrame, is_quarterly: bool) -> Dict[str, pd.DataFrame]:
    pivots: Dict[str, pd.DataFrame] = {}
    if df.empty:
        for r in RATIOS:
            pivots[r] = pd.DataFrame()
        return pivots

    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])

    if is_quarterly:
        temp = temp[temp["Date"] >= QUARTERLY_START].copy()
        temp["Label"] = temp["Date"].dt.year.astype(str) + "Q" + temp["Date"].dt.quarter.astype(str)
    else:
        temp = temp[temp["Date"] >= ANNUAL_START].copy()
        temp["Label"] = temp["Date"].dt.strftime("%Y")

    label_order = (
        temp.groupby("Label", as_index=True)["Date"]
        .max()
        .sort_values()
        .index
        .tolist()
    )

    for ratio in RATIOS:
        p = temp.pivot_table(index="Ticker", columns="Label", values=ratio, aggfunc="first")
        p = p.reindex(columns=label_order)
        p.index.name = "Ticker"
        pivots[ratio] = p

    return pivots


def _period_sort_key(label: str):
    s = str(label).strip()
    m_q = re.fullmatch(r"(\d{4})Q([1-4])", s)
    if m_q:
        return (int(m_q.group(1)), int(m_q.group(2)), 1)
    m_y = re.fullmatch(r"(\d{4})", s)
    if m_y:
        return (int(m_y.group(1)), 0, 0)
    return (10**9, 0, s)


def _normalize_us_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = out.index.map(lambda x: str(x).strip().upper())
    out.index.name = "Ticker"
    return out


def merge_preserve_existing(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    current_tickers: List[str],
) -> pd.DataFrame:
    existing = _normalize_us_index(existing if existing is not None else pd.DataFrame())
    new = _normalize_us_index(new if new is not None else pd.DataFrame())

    # 현재 구성종목만 유지: 편출 종목 제거, 편입 종목은 NaN 행으로 추가 가능
    current_index = pd.Index([str(t).strip().upper() for t in current_tickers], name="Ticker")

    if existing.empty and new.empty:
        return pd.DataFrame(index=current_index)
    if existing.empty:
        return new.reindex(index=current_index).sort_index()
    if new.empty:
        return existing.reindex(index=current_index).sort_index()

    all_index = current_index
    all_cols = sorted(set(existing.columns).union(set(new.columns)), key=_period_sort_key)

    ex = existing.reindex(index=all_index, columns=all_cols)
    nw = new.reindex(index=all_index, columns=all_cols)

    merged = ex.where(ex.notna(), nw)
    merged.index.name = "Ticker"
    return merged.sort_index()


def _read_sheet_or_empty(path: str, sheet_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet_name, index_col=0)
    except Exception:
        return pd.DataFrame()


def run_pipeline(
    sleep_min: float = 0.3,
    sleep_max: float = 0.5,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = get_sp500_tickers()
    current_tickers = [str(t).strip().upper() for t in tickers]

    print(f"Total tickers: {len(tickers)}")

    annual_records: List[Dict] = []
    quarterly_records: List[Dict] = []
    failed_tickers: List[Tuple[str, str]] = []

    for ticker in tqdm(tickers, desc="Fetching US Yahoo Finance data"):
        try:
            a_rows, q_rows = extract_ratios_for_ticker(
                ticker,
                annual_start=ANNUAL_START,
                quarterly_start=QUARTERLY_START,
            )
            annual_records.extend(a_rows)
            quarterly_records.extend(q_rows)
        except Exception as e:
            failed_tickers.append((ticker, str(e)))
            print(f"[WARN] skip {ticker}: {e}")
        finally:
            time.sleep(random.uniform(sleep_min, sleep_max))

    annual_df = pd.DataFrame(annual_records)
    quarterly_df = pd.DataFrame(quarterly_records)

    print("Annual rows:", annual_df.shape)
    print("Quarterly rows:", quarterly_df.shape)
    print("Failed tickers:", len(failed_tickers))

    annual_pivots = build_ratio_pivots(annual_df, is_quarterly=False)
    quarterly_pivots = build_ratio_pivots(quarterly_df, is_quarterly=True)

    saved_files = []
    for ratio in RATIOS:
        out_path = os.path.join(OUTPUT_DIR, f"US_{ratio}.xlsx")

        existing_annual = _read_sheet_or_empty(out_path, "Annual")
        existing_quarterly = _read_sheet_or_empty(out_path, "Quarterly")

        merged_annual = merge_preserve_existing(existing_annual, annual_pivots[ratio], current_tickers)
        merged_quarterly = merge_preserve_existing(existing_quarterly, quarterly_pivots[ratio], current_tickers)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            merged_annual.to_excel(writer, sheet_name="Annual", index=True)
            merged_quarterly.to_excel(writer, sheet_name="Quarterly", index=True)

        saved_files.append(out_path)
        print(f"[UPSERT] {ratio}: Annual {merged_annual.shape}, Quarterly {merged_quarterly.shape}")

    print("Saved files:")
    for f in saved_files:
        print("-", f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="US S&P500 financial ratio pipeline")
    parser.add_argument("--sleep-min", type=float, default=0.3)
    parser.add_argument("--sleep-max", type=float, default=0.5)
    args = parser.parse_args()

    run_pipeline(sleep_min=args.sleep_min, sleep_max=args.sleep_max)
