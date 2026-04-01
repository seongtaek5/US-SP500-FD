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
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

OUTPUT_DIR = "output_kospi200"
RATIOS = ["ROE", "PBR", "PER", "ROA", "EPS"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_kospi200_from_naver() -> pd.DataFrame:
    rows: List[Tuple[str, str]] = []
    seen = set()

    for page in range(1, 21):
        url = f"https://finance.naver.com/sise/entryJongmok.naver?&page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.select("td.ctg a")
        if not anchors:
            break

        for a in anchors:
            href = a.get("href", "")
            m = re.search(r"code=(\d{6})", href)
            if not m:
                continue
            code = m.group(1)
            if code in seen:
                continue
            seen.add(code)
            name = a.get_text(strip=True).replace(" ", "")
            rows.append((code, name))

    df = pd.DataFrame(rows, columns=["Code", "Name"])
    df = df.drop_duplicates("Code").sort_values("Code").reset_index(drop=True)

    if len(df) < 180:
        cache_path = os.path.join("data", "kospi200_codes_cache.csv")
        if os.path.exists(cache_path):
            cache_df = pd.read_csv(cache_path, dtype={"Code": str})
            cache_df["Code"] = cache_df["Code"].astype(str).str.zfill(6)
            cache_df = cache_df[["Code"]].drop_duplicates("Code")
            df = cache_df.merge(df, on="Code", how="left").sort_values("Code").reset_index(drop=True)

    return df


def get_naver_company_name(code: str) -> str:
    url = f"https://finance.naver.com/item/main.naver?code={code}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.text.strip() if soup.title else ""
    return title.split(":")[0].strip() if ":" in title else title


def _to_number(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in ["", "-", "N/A", "nan"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _clean_period_label(s: str) -> str:
    return str(s).replace("(E)", "").strip()


def _period_to_timestamp(period: str) -> pd.Timestamp:
    return pd.to_datetime(
        period.replace("/", ".") + ".01",
        format="%Y.%m.%d",
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)


def _get_naver_close_series(code: str) -> pd.Series:
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count=3000&requestType=0"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    pairs = re.findall(r"data=\"(\d{8}\|[^\"]+)\"", resp.text)
    if not pairs:
        return pd.Series(dtype="float64")

    data = []
    for p in pairs:
        parts = p.split("|")
        if len(parts) < 5:
            continue
        dt = pd.to_datetime(parts[0], format="%Y%m%d", errors="coerce")
        close = _to_number(parts[4])
        if pd.notna(dt) and pd.notna(close):
            data.append((dt, close))

    if not data:
        return pd.Series(dtype="float64")

    return pd.Series({dt: close for dt, close in data}).sort_index()


def _last_close_on_or_before(close_s: pd.Series, dt: pd.Timestamp) -> float:
    if close_s is None or close_s.empty:
        return np.nan
    sub = close_s[close_s.index <= pd.Timestamp(dt)]
    if sub.empty:
        return np.nan
    return float(sub.iloc[-1])


def _extract_financial_table(code: str) -> pd.DataFrame:
    url = (
        "https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?"
        f"pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"
    )
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))

    target = None
    for t in tables:
        if not isinstance(t.columns, pd.MultiIndex):
            continue
        lv0 = t.columns.get_level_values(0).astype(str)
        first_col = t.iloc[:, 0].astype(str).tolist() if not t.empty else []
        if (
            any("Annual" in c for c in lv0)
            and any("Net Quarter" in c for c in lv0)
            and any("ROE" in r for r in first_col)
            and any("EPS" in r for r in first_col)
        ):
            target = t
            break

    if target is None:
        raise RuntimeError(f"Financial summary table not found: {code}")

    return target


def extract_ratios_for_code(code: str, name: str) -> Tuple[List[Dict], List[Dict]]:
    t = _extract_financial_table(code)
    close_s = _get_naver_close_series(code)

    row_label_col = t.columns[0]
    labels = t[row_label_col].astype(str).str.strip()

    idx_map = {}
    for ratio, patt in {
        "ROE": r"^ROE",
        "PBR": r"^PBR",
        "PER": r"^PER",
        "EPS": r"^EPS",
        "ROA": r"^ROA",
        "NI": r"^당기순이익",
        "ASSET": r"^자산총계",
    }.items():
        hit = labels[labels.str.contains(patt, regex=True, na=False)]
        idx_map[ratio] = hit.index[0] if len(hit) > 0 else None

    annual_records: List[Dict] = []
    quarterly_records: List[Dict] = []

    for col in t.columns[1:]:
        sec = str(col[0]).strip()
        period_raw = str(col[1]).strip()

        if "(E)" in period_raw:
            continue

        period = _clean_period_label(period_raw)
        dt = _period_to_timestamp(period)
        if pd.isna(dt):
            continue

        def getv(key):
            idx = idx_map.get(key)
            if idx is None:
                return np.nan
            return _to_number(t.loc[idx, col])

        roe = getv("ROE")
        pbr = getv("PBR")
        per = getv("PER")
        eps = getv("EPS")
        roa = getv("ROA")

        if pd.isna(roa):
            ni = getv("NI")
            asset = getv("ASSET")
            if pd.notna(ni) and pd.notna(asset) and asset != 0:
                roa = (ni / asset) * 100.0

        if pd.isna(per) and pd.notna(eps) and eps != 0:
            close = _last_close_on_or_before(close_s, dt)
            if pd.notna(close):
                if "Net Quarter" in sec:
                    per = close / (eps * 4.0)
                elif "Annual" in sec:
                    per = close / eps

        rec = {
            "Code": code,
            "Name": name,
            "Date": pd.Timestamp(dt),
            "ROE": roe,
            "PBR": pbr,
            "PER": per,
            "ROA": roa,
            "EPS": eps,
        }

        if "Annual" in sec:
            annual_records.append(rec)
        elif "Net Quarter" in sec:
            quarterly_records.append(rec)

    return annual_records, quarterly_records


def build_pivots(df: pd.DataFrame, is_quarterly: bool) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df.empty:
        for r in RATIOS:
            out[r] = pd.DataFrame()
        return out

    temp = df.copy()
    temp["Date"] = pd.to_datetime(temp["Date"])
    temp["Ticker"] = temp["Code"]

    if is_quarterly:
        temp["Label"] = temp["Date"].dt.year.astype(str) + "Q" + temp["Date"].dt.quarter.astype(str)
    else:
        temp["Label"] = temp["Date"].dt.strftime("%Y")

    label_order = (
        temp[["Label", "Date"]]
        .drop_duplicates()
        .sort_values("Date")["Label"]
        .tolist()
    )

    for ratio in RATIOS:
        p = temp.pivot_table(index="Ticker", columns="Label", values=ratio, aggfunc="first")
        p = p.reindex(columns=label_order)
        p.index.name = "Ticker"
        out[ratio] = p

    return out


def _period_sort_key(label: str):
    s = str(label).strip()
    m_q = re.fullmatch(r"(\d{4})Q([1-4])", s)
    if m_q:
        return (int(m_q.group(1)), int(m_q.group(2)), 1)
    m_y = re.fullmatch(r"(\d{4})", s)
    if m_y:
        return (int(m_y.group(1)), 0, 0)
    return (10**9, 0, s)


def _normalize_kr_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out.index = out.index.map(lambda x: str(x).strip())
    out.index = out.index.map(lambda x: x.zfill(6) if x.isdigit() else x)
    out.index.name = "Ticker"
    return out


def merge_preserve_existing(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    current_tickers: List[str],
) -> pd.DataFrame:
    existing = _normalize_kr_index(existing if existing is not None else pd.DataFrame())
    new = _normalize_kr_index(new if new is not None else pd.DataFrame())

    # 현재 구성종목만 유지: 편출 종목 제거, 편입 종목은 NaN 행으로 추가 가능
    current_index = pd.Index([str(t).zfill(6) for t in current_tickers], name="Ticker")

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
    sleep_min: float = 0.25,
    sleep_max: float = 0.45,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    kospi200_df = get_kospi200_from_naver()
    if "Name" not in kospi200_df.columns:
        kospi200_df["Name"] = ""

    missing = kospi200_df["Name"].astype(str).str.strip().eq("")
    for i in kospi200_df[missing].index:
        code = kospi200_df.at[i, "Code"]
        try:
            kospi200_df.at[i, "Name"] = get_naver_company_name(code)
        except Exception:
            pass
        time.sleep(random.uniform(0.1, 0.2))

    kospi200_df = kospi200_df[["Code", "Name"]].drop_duplicates("Code").reset_index(drop=True)
    current_codes = kospi200_df["Code"].astype(str).str.zfill(6).tolist()
    print("Total constituents parsed:", len(kospi200_df))

    annual_records: List[Dict] = []
    quarterly_records: List[Dict] = []
    failed: List[Tuple[str, str, str]] = []

    for _, row in tqdm(
        kospi200_df.iterrows(),
        total=len(kospi200_df),
        desc="Fetching KR financial summary",
    ):
        code = row["Code"]
        name = row["Name"]
        try:
            a, q = extract_ratios_for_code(code, name)
            annual_records.extend(a)
            quarterly_records.extend(q)
        except Exception as e:
            failed.append((code, name, str(e)))
            print(f"[WARN] skip {code} {name}: {e}")
        finally:
            time.sleep(random.uniform(sleep_min, sleep_max))

    annual_df = pd.DataFrame(annual_records)
    quarterly_df = pd.DataFrame(quarterly_records)
    print("Annual rows:", annual_df.shape)
    print("Quarterly rows:", quarterly_df.shape)
    print("Failed:", len(failed))

    annual_pivots = build_pivots(annual_df, is_quarterly=False)
    quarterly_pivots = build_pivots(quarterly_df, is_quarterly=True)

    saved_files = []
    for r in RATIOS:
        out_path = os.path.join(OUTPUT_DIR, f"KR_{r}.xlsx")

        existing_annual = _read_sheet_or_empty(out_path, "Annual")
        existing_quarterly = _read_sheet_or_empty(out_path, "Quarterly")

        merged_annual = merge_preserve_existing(existing_annual, annual_pivots[r], current_codes)
        merged_quarterly = merge_preserve_existing(existing_quarterly, quarterly_pivots[r], current_codes)

        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            merged_annual.to_excel(writer, sheet_name="Annual", index=True)
            merged_quarterly.to_excel(writer, sheet_name="Quarterly", index=True)

        saved_files.append(out_path)
        print(f"[UPSERT] {r}: Annual {merged_annual.shape}, Quarterly {merged_quarterly.shape}")

    print("Saved files:")
    for p in saved_files:
        print("-", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KR KOSPI200 financial ratio pipeline")
    parser.add_argument("--sleep-min", type=float, default=0.25)
    parser.add_argument("--sleep-max", type=float, default=0.45)
    args = parser.parse_args()

    run_pipeline(sleep_min=args.sleep_min, sleep_max=args.sleep_max)
