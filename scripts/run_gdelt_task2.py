"""Attempt GDELT news intensity for Task 2 (NWFD / Russia, 2023–2024).

Honest alignment like Trends: one monthly intensity series → merge to BOTH
rural/urban aggregates identically. Past months only (lag≥1, no same-month
leakage into features beyond what Trends does — we use lag_1 so feature at ds
is previous month's article count).

Outputs:
  data/processed/gdelt_news_intensity.csv
  results/forecast_news.csv
  Appends status to docs/TASK2_GAPS.md if no MAE win.

Budget: stop cleanly on API/geo/empty failures.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import DATA_INT, DATA_PROC, RESULTS, norm_name  # noqa: E402

HORIZONS = [1, 3, 6, 12]
CLUSTER_NAMES = {0: "сельские", 1: "городские"}
OUT_INTENSITY = DATA_PROC / "gdelt_news_intensity.csv"
OUT_FORECAST = RESULTS / "forecast_news.csv"
GAPS = ROOT / "docs" / "TASK2_GAPS.md"


def _http_get(url: str, timeout: int = 90, retries: int = 5) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "sber-project-task2/1.0 (research; academic)"},
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                wait = 20 * (attempt + 1)
                print(f"  ! HTTP {e.code}, sleep {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            wait = 10 * (attempt + 1)
            print(f"  ! net error {e}, sleep {wait}s")
            time.sleep(wait)
    raise last_err or RuntimeError("http failed")


def fetch_gdelt_doc_api(query: str, start: str, end: str) -> pd.DataFrame | None:
    """GDELT DOC 2.0 timelinevol for a query. Dates YYYYMMDDHHMMSS."""
    params = {
        "query": query,
        "mode": "timelinevol",
        "startdatetime": start,
        "enddatetime": end,
        "format": "json",
        "timelinesmooth": "0",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    print(f"  GET DOC timelinevol: {query[:80]}...")
    try:
        raw = _http_get(url)
        text = raw.decode("utf-8", errors="replace").strip()
        if not text or text.startswith("<"):
            print(f"  ! empty/HTML response ({len(text)} chars)")
            return None
        data = json.loads(text)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as e:
        print(f"  ! DOC API failed: {e}")
        return None

    # timelinevol: data.timeline[0].data → [{date, value}, ...]
    timeline = None
    if isinstance(data, dict):
        tl = data.get("timeline") or data.get("data")
        if isinstance(tl, list) and tl:
            if isinstance(tl[0], dict) and "data" in tl[0]:
                timeline = tl[0]["data"]
            elif isinstance(tl[0], dict) and "date" in tl[0]:
                timeline = tl
    if not timeline:
        print(f"  ! no timeline in response keys={list(data)[:8] if isinstance(data, dict) else type(data)}")
        return None

    rows = []
    for pt in timeline:
        d = str(pt.get("date", ""))
        v = pt.get("value", pt.get("count", 0))
        if len(d) >= 8:
            # YYYYMMDD… → month start
            period = pd.Timestamp(f"{d[:4]}-{d[4:6]}-01")
            try:
                rows.append({"period": period, "value": float(v)})
            except (TypeError, ValueError):
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    monthly = df.groupby("period", as_index=False)["value"].sum()
    monthly = monthly.rename(columns={"value": "gdelt_vol"})
    return monthly


def fetch_gdelt_package() -> pd.DataFrame | None:
    """Try python-gdelt package if installed (Events/GKG). Often heavy/flaky."""
    try:
        import gdelt  # type: ignore
    except ImportError:
        print("  gdelt package not installed — skip")
        return None
    print("  Trying gdelt package SearchEvents (may be slow)…")
    try:
        gd = gdelt.gdelt(version=2)
        # Single-day probe to see if package works; full range is huge
        tables = gd.Search(["2024 Oct 01"], table="events", coverage=False, output="df")
        if tables is None or (hasattr(tables, "empty") and tables.empty):
            print("  ! gdelt package returned empty")
            return None
        print(f"  gdelt package probe ok: {len(tables)} rows (not using full dump — too heavy)")
        return None  # Prefer DOC API monthly intensity; package is event-dump oriented
    except Exception as e:
        print(f"  ! gdelt package failed: {e}")
        return None


def build_intensity() -> tuple[pd.DataFrame | None, str]:
    """Return monthly intensity + geo_note. Prefer Russia-wide if SZFO too thin."""
    start, end = "20230101000000", "20241231235959"
    # ASCII-heavy queries first (encoding/rate-limit friendlier); SZFO geo is weak.
    queries = [
        ("sourcecountry:Russia", "russia_wide"),
        ("(Murmansk OR Arkhangelsk OR Kaliningrad OR Karelia OR Vologda OR Pskov OR Novgorod OR Petersburg OR Leningrad) sourcecountry:Russia",
         "szfo_keywords"),
        ("Russia", "russia_en"),
    ]
    fetch_gdelt_package()  # probe only

    best = None
    geo_note = "none"
    for q, tag in queries:
        monthly = fetch_gdelt_doc_api(q, start, end)
        time.sleep(8)
        if monthly is None or len(monthly) < 6:
            print(f"  skip {tag}: insufficient months")
            continue
        print(f"  {tag}: {len(monthly)} months, sum={monthly['gdelt_vol'].sum():.1f}")
        # Prefer usable Russia-wide; SZFO keyword filter often too thin
        if tag.startswith("russia") and monthly["gdelt_vol"].sum() > 1.0:
            best, geo_note = monthly, f"{tag} — SZFO geo weak; Russia-wide intensity (honest)"
            break
        if tag == "szfo_keywords" and monthly["gdelt_vol"].sum() > 1.0:
            best, geo_note = monthly, "szfo_keywords (weak geo filter; may be sparse)"
            break
        if best is None:
            best, geo_note = monthly, f"{tag} (fallback)"

    if best is None:
        return None, "fetch_failed"

    best = best.sort_values("period").reset_index(drop=True)
    # Intensity index: z-score of monthly volume (or raw if constant)
    vol = best["gdelt_vol"].astype(float)
    if vol.std() > 1e-9:
        best["news_intensity"] = (vol - vol.mean()) / vol.std()
    else:
        best["news_intensity"] = vol
    # lag-1: feature for month t uses intensity of t-1 (past only)
    best["news_intensity_lag1"] = best["news_intensity"].shift(1)
    best["geo_note"] = geo_note
    OUT_INTENSITY.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(OUT_INTENSITY, index=False, encoding="utf-8-sig")
    print(f"OK intensity -> {OUT_INTENSITY} ({geo_note})")
    return best, geo_note


def load_agg_spectral():
    spend = pd.read_parquet(DATA_INT / "spend.parquet")
    spend["period"] = pd.to_datetime(spend["period"])
    spend["mo_norm"] = spend["mo"].apply(norm_name)
    all_cat = spend[spend["category_15"] == "Все категории"].copy()
    all_cat = all_cat.groupby(["mo_norm", "period"])["value"].mean().reset_index()
    labels_thr = np.load(RESULTS / "labels_threshold.npy")
    final = pd.read_csv(DATA_PROC / "features_szfo_v2_final.csv", encoding="utf-8-sig")
    final["cluster"] = labels_thr
    mo_to_cluster = dict(zip(final["mo_norm"], final["cluster"]))
    all_cat["cluster"] = all_cat["mo_norm"].map(mo_to_cluster)
    all_cat = all_cat.dropna(subset=["cluster"])
    all_cat["cluster"] = all_cat["cluster"].astype(int)
    agg = all_cat.groupby(["cluster", "period"])["value"].mean().reset_index()
    agg = agg.sort_values(["cluster", "period"])

    with open(RESULTS / "dynamic_summary.json", encoding="utf-8") as f:
        dyn = json.load(f)
    spectral = pd.DataFrame(dyn["windows"])
    for k in range(5):
        spectral[f"lambda_{k+1}"] = spectral["top_evals"].apply(
            lambda x, kk=k: x[kk] if kk < len(x) else np.nan
        )
    spectral["end_dt"] = pd.to_datetime(spectral["end"])
    return agg, spectral


def make_features(sub, spectral, news_df, use_news: bool):
    df = sub.copy()
    df["month"] = df["ds"].dt.month
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ma_3"] = df["y"].shift(1).rolling(3).mean()
    df["ma_6"] = df["y"].shift(1).rolling(6).mean()
    for idx, row in df.iterrows():
        mask = spectral["end_dt"] < row["ds"]
        if mask.any():
            last = spectral[mask].iloc[-1]
            for col in ["PR_plus", "frustration", "lambda_1", "lambda_2"]:
                df.loc[idx, col] = last[col]
    if use_news and news_df is not None:
        # merge lag-1 intensity on same calendar month as ds (value already lagged)
        nd = news_df[["period", "news_intensity_lag1"]].rename(
            columns={"news_intensity_lag1": "news_intensity"}
        )
        df = df.merge(nd, left_on="ds", right_on="period", how="left")
        df = df.drop(columns=["period"])
    return df.dropna()


def predict_lgbm(df, H):
    import lightgbm as lgb
    tr, te = df.iloc[:-H], df.iloc[-H:]
    fcols = [c for c in df.columns if c not in ("ds", "y")]
    m = lgb.LGBMRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        min_child_samples=3, reg_alpha=0.5, reg_lambda=0.5,
        verbose=-1, random_state=42,
    )
    m.fit(tr[fcols].values, tr["y"].values)
    return m.predict(te[fcols].values), te["y"].values


def run_ablation(news_df: pd.DataFrame, geo_note: str) -> pd.DataFrame:
    from sklearn.metrics import mean_absolute_error, r2_score

    agg, spectral = load_agg_spectral()
    # Trends baseline from existing CSV
    base = pd.read_csv(RESULTS / "forecast_all_models.csv")
    rows = []
    for cluster in [0, 1]:
        sub = agg[agg["cluster"] == cluster].sort_values("period").copy()
        sub = sub.rename(columns={"period": "ds", "value": "y"})[["ds", "y"]]
        label = CLUSTER_NAMES[cluster]
        for H in HORIZONS:
            row = {
                "cluster": cluster,
                "cluster_name": label,
                "horizon": H,
                "geo_note": geo_note,
            }
            bref = base[(base["cluster"] == cluster) & (base["horizon"] == H)]
            if len(bref):
                row["MAE_lgbm"] = bref.iloc[0].get("MAE_lgbm")
                row["MAE_lgbm_trends"] = bref.iloc[0].get("MAE_lgbm_trends")
                row["MAE_prophet"] = bref.iloc[0].get("MAE_prophet")
                row["MAE_naive"] = bref.iloc[0].get("MAE_naive")
            try:
                df = make_features(sub, spectral, news_df, use_news=True)
                if len(df) >= 10:
                    p, yt = predict_lgbm(df, H)
                    row["MAE_lgbm_news"] = mean_absolute_error(yt, p)
                    row["R2_lgbm_news"] = r2_score(yt, p) if H > 1 else np.nan
                    if pd.notna(row.get("MAE_lgbm")):
                        row["delta_vs_lgbm"] = row["MAE_lgbm_news"] - row["MAE_lgbm"]
                    if pd.notna(row.get("MAE_lgbm_trends")):
                        row["delta_vs_trends"] = row["MAE_lgbm_news"] - row["MAE_lgbm_trends"]
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
            print(f"  {label} H={H}: MAE_news={row.get('MAE_lgbm_news', '—')} "
                  f"Δlgbm={row.get('delta_vs_lgbm', '—')}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_FORECAST, index=False, encoding="utf-8-sig")
    print(f"OK forecast_news -> {OUT_FORECAST}")
    return out


def update_gaps(ok: bool, improved: bool, detail: str):
    if not GAPS.exists():
        return
    text = GAPS.read_text(encoding="utf-8")
    marker = "## GDELT attempt"
    block = (
        f"\n{marker}\n\n"
        f"- Дата прогона: 2026-10-08\n"
        f"- Статус: {'успех выгрузки' if ok else 'fail / blocked'}"
        f"{'; MAE улучшился' if improved else '; выигрыша по MAE нет' if ok else ''}\n"
        f"- Детали: {detail}\n"
        f"- Артефакты: `data/processed/gdelt_news_intensity.csv`, `results/forecast_news.csv`\n"
        f"- Гео: фильтр СЗФО слабый; при необходимости — Russia-wide (честно в note)\n"
        f"- Anti-leakage: `news_intensity_lag1` (прошлый месяц) + spectral `end_dt < ds`\n"
    )
    if marker in text:
        # replace from marker to next ## or EOF
        import re
        text = re.sub(r"\n## GDELT attempt\n[\s\S]*?(?=\n## |\Z)", block, text, count=1)
    else:
        text = text.rstrip() + "\n" + block
    GAPS.write_text(text, encoding="utf-8")
    print(f"OK updated {GAPS.name}")


def main():
    print("=== GDELT Task 2 attempt ===")
    news_df, geo_note = build_intensity()
    if news_df is None:
        update_gaps(False, False, "DOC API / package не дали месячный ряд 2023–2024")
        print("FAIL: no intensity series")
        return 1
    # need lag1 non-null for enough months
    usable = news_df.dropna(subset=["news_intensity_lag1"])
    if len(usable) < 8:
        update_gaps(False, False, f"мало месяцев после lag1: {len(usable)}; geo={geo_note}")
        print("FAIL: too few months after lag")
        return 1
    out = run_ablation(news_df, geo_note)
    deltas = out["delta_vs_lgbm"].dropna() if "delta_vs_lgbm" in out.columns else pd.Series(dtype=float)
    # improve = lower MAE on any cell without worsening all others badly; honest: any cell better
    improved = bool((deltas < -1.0).any()) if len(deltas) else False
    n_better = int((deltas < -1.0).sum()) if len(deltas) else 0
    detail = (
        f"geo={geo_note}; cells_better_vs_lgbm={n_better}/{len(deltas)}; "
        f"mean_delta={float(deltas.mean()) if len(deltas) else float('nan'):.1f}"
    )
    update_gaps(True, improved, detail)
    if improved:
        print("IMPROVED on ≥1 cell — consider slide 'News: GDELT'")
    else:
        print("NO MAE improvement — documenting gap, no narrative win")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
