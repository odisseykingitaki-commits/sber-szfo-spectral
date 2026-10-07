"""Fetch GDELT events for Russia 2023-2024, aggregate monthly intensity."""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import gdelt
import numpy as np
import pandas as pd

OUT = Path("data/processed/gdelt_news_monthly.csv")
META = Path("results/gdelt_fetch_meta.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)
META.parent.mkdir(parents=True, exist_ok=True)

# NWFD / SZFO name fragments (English GDELT FullName)
NWFD_NEEDLES = [
    "petersburg", "sankt-peterburg", "leningrad", "murmansk", "arkhangel",
    "vologda", "kaliningrad", "karelia", "novgorod", "pskov", "nenets",
    "komi", "severodvinsk", "cherepovets", "petrozavodsk", "syktyvkar",
]

# Sample days per month to limit download volume / rate
SAMPLE_DAYS = [1, 8, 15, 22]


def is_nwfd(name: str) -> bool:
    if not isinstance(name, str):
        return False
    low = name.lower()
    return any(n in low for n in NWFD_NEEDLES)


def month_dates(year: int, month: int):
    dates = []
    for d in SAMPLE_DAYS:
        try:
            dates.append(f"{year:04d}-{month:02d}-{d:02d}")
        except Exception:
            pass
    return dates


def fetch_day(gd, date_str: str, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            df = gd.Search([date_str], table="events", coverage=False, translation=False)
            if isinstance(df, pd.DataFrame):
                return df
            return pd.DataFrame()
        except Exception as e:
            last_err = e
            time.sleep(2 + attempt * 2)
    print(f"  FAIL {date_str}: {last_err}")
    return None


def main():
    gd = gdelt.gdelt(version=2)
    rows = []
    errors = []
    n_ok = 0
    n_fail = 0

    months = [(y, m) for y in (2023, 2024) for m in range(1, 13)]
    print(f"Months: {len(months)}, sample days/month: {SAMPLE_DAYS}")

    for yi, (year, month) in enumerate(months):
        day_frames = []
        for date_str in month_dates(year, month):
            print(f"[{yi+1}/{len(months)}] {date_str} ...", flush=True)
            df = fetch_day(gd, date_str)
            time.sleep(0.4)  # be polite to CDN
            if df is None:
                n_fail += 1
                errors.append(date_str)
                continue
            n_ok += 1
            if len(df) == 0:
                continue
            # Russia-related: action geo OR actor country
            mask = pd.Series(False, index=df.index)
            if "ActionGeo_CountryCode" in df.columns:
                mask = mask | (df["ActionGeo_CountryCode"] == "RS")
            if "Actor1CountryCode" in df.columns:
                mask = mask | (df["Actor1CountryCode"] == "RUS")
            if "Actor2CountryCode" in df.columns:
                mask = mask | (df["Actor2CountryCode"] == "RUS")
            sub = df.loc[mask].copy()
            if len(sub) == 0:
                continue
            sub["_nwfd"] = sub.get("ActionGeo_FullName", pd.Series("", index=sub.index)).map(is_nwfd)
            day_frames.append(sub)

        ym = f"{year:04d}-{month:02d}"
        if not day_frames:
            rows.append({
                "year_month": ym,
                "n_sample_days_ok": 0,
                "event_count_russia": 0,
                "event_count_nwfd": 0,
                "mentions_sum": 0.0,
                "avg_tone": np.nan,
                "avg_goldstein": np.nan,
                "tone_volume": 0.0,  # count * |tone| proxy
                "geo_scope_note": "no_data",
            })
            continue

        cat = pd.concat(day_frames, ignore_index=True)
        tone = cat["AvgTone"] if "AvgTone" in cat.columns else pd.Series(dtype=float)
        gold = cat["GoldsteinScale"] if "GoldsteinScale" in cat.columns else pd.Series(dtype=float)
        mentions = cat["NumMentions"] if "NumMentions" in cat.columns else pd.Series(1.0, index=cat.index)
        nwfd_n = int(cat["_nwfd"].sum())
        rows.append({
            "year_month": ym,
            "n_sample_days_ok": len(day_frames),
            "event_count_russia": int(len(cat)),
            "event_count_nwfd": nwfd_n,
            "mentions_sum": float(mentions.sum()),
            "avg_tone": float(tone.mean()) if len(tone) else np.nan,
            "avg_goldstein": float(gold.mean()) if len(gold) else np.nan,
            "tone_volume": float((mentions.abs() * tone.abs()).sum()) if len(tone) else float(mentions.sum()),
            "geo_scope_note": "russia_wide_primary_nwfd_sparse" if nwfd_n < 20 else "russia_and_nwfd",
        })
        print(f"  -> {ym}: RU={len(cat)} NWFD={nwfd_n} tone={rows[-1]['avg_tone']:.2f}")

    out = pd.DataFrame(rows)
    # Intensity index: z-score of event_count + tone_volume (lag-ready monthly)
    for col in ["event_count_russia", "tone_volume", "mentions_sum"]:
        s = out[col].astype(float)
        out[f"{col}_z"] = (s - s.mean()) / (s.std() + 1e-9)
    out["news_intensity"] = (
        out["event_count_russia_z"].fillna(0)
        + 0.5 * out["tone_volume_z"].fillna(0)
    )
    # Prefer Russia-wide; expose NWFD count for transparency
    out["geo_scope"] = "russia_wide"
    if out["event_count_nwfd"].sum() >= 50:
        # secondary NWFD intensity (often sparse)
        s = out["event_count_nwfd"].astype(float)
        out["news_intensity_nwfd"] = (s - s.mean()) / (s.std() + 1e-9)
    else:
        out["news_intensity_nwfd"] = np.nan

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    meta = [
        f"ok_days={n_ok}",
        f"fail_days={n_fail}",
        f"fail_list={errors[:20]}",
        f"out={OUT}",
        f"geo_scope=russia_wide (NWFD filter via FullName; total_nwfd_events={int(out['event_count_nwfd'].sum())})",
        f"sample_days={SAMPLE_DAYS}",
        f"period=2023-01..2024-12",
        f"source=gdelt python package v2 events (daily files, sampled days)",
    ]
    META.write_text("\n".join(meta), encoding="utf-8")
    print("\n".join(meta))
    print(out.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        META.write_text("FETCH_FAILED\n" + traceback.format_exc(), encoding="utf-8")
        raise
