"""LGBM ablation: spectral + GDELT news (lag-1) vs baselines from forecast_all_models.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils import DATA_INT, DATA_PROC, RESULTS, norm_name  # noqa: E402

HORIZONS = [1, 3, 6, 12]
CLUSTER_NAMES = {0: "сельские", 1: "городские"}
NOTE_PATH = RESULTS / "forecast_news_notes.txt"


def load_agg_spectral_trends():
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
            lambda x, k=k: x[k] if k < len(x) else np.nan
        )
    spectral["end_dt"] = pd.to_datetime(spectral["end"])

    trends_pivot, trends_cols = None, []
    try:
        trends = pd.read_csv(DATA_PROC / "google_trends_szfo.csv", encoding="utf-8-sig")
        trends["period"] = pd.to_datetime(trends["year_month"] + "-01")
        queries_keep = [
            q
            for q in trends["query"].unique()
            if trends[trends["query"] == q]["interest"].sum() > 100
        ]
        trends = trends[trends["query"].isin(queries_keep)]
        trends_agg = trends.groupby(["period", "query"])["interest"].mean().reset_index()
        trends_pivot = trends_agg.pivot(
            index="period", columns="query", values="interest"
        ).reset_index()
        trends_pivot.columns.name = None
        trends_cols = [c for c in trends_pivot.columns if c != "period"]
    except Exception:
        pass

    return agg, spectral, trends_pivot, trends_cols


def load_news():
    path = DATA_PROC / "gdelt_news_monthly.csv"
    news = pd.read_csv(path, encoding="utf-8-sig")
    news["period"] = pd.to_datetime(news["year_month"] + "-01")
    # Conservative anti-leakage: feature for month t uses news of t-1 only
    news = news.sort_values("period")
    feat_cols = [
        "news_intensity",
        "event_count_russia",
        "avg_tone",
        "tone_volume",
        "avg_goldstein",
    ]
    keep = ["period"] + [c for c in feat_cols if c in news.columns]
    news = news[keep].copy()
    for c in keep:
        if c != "period":
            news[f"{c}_lag1"] = news[c].shift(1)
    lag_cols = [c for c in news.columns if c.endswith("_lag1")]
    news_lag = news[["period"] + lag_cols].copy()
    return news_lag, lag_cols


def make_features(sub, spectral, trends_pivot, trends_cols, news_lag, news_cols,
                  use_trends, use_spectral, use_news):
    df = sub.copy()
    df["month"] = df["ds"].dt.month
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["ma_3"] = df["y"].shift(1).rolling(3).mean()
    df["ma_6"] = df["y"].shift(1).rolling(6).mean()

    if use_spectral:
        for idx, row in df.iterrows():
            mask = spectral["end_dt"] < row["ds"]
            if mask.any():
                last = spectral[mask].iloc[-1]
                for col in ["PR_plus", "frustration", "lambda_1", "lambda_2"]:
                    df.loc[idx, col] = last[col]

    if use_trends and trends_pivot is not None:
        df = df.merge(
            trends_pivot[["period"] + trends_cols],
            left_on="ds",
            right_on="period",
            how="left",
        )
        df = df.drop(columns=["period"])

    if use_news and news_lag is not None:
        # merge on calendar month: news_*_lag1 at period=t is news from t-1
        df = df.merge(news_lag, left_on="ds", right_on="period", how="left")
        df = df.drop(columns=["period"])

    return df.dropna()


def predict_lgbm(df, H):
    import lightgbm as lgb

    tr, te = df.iloc[:-H], df.iloc[-H:]
    fcols = [c for c in df.columns if c not in ("ds", "y")]
    m = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        min_child_samples=3,
        reg_alpha=0.5,
        reg_lambda=0.5,
        verbose=-1,
        random_state=42,
    )
    m.fit(tr[fcols].values, tr["y"].values)
    return m.predict(te[fcols].values), te["y"].values


def main():
    notes = []
    news_path = DATA_PROC / "gdelt_news_monthly.csv"
    if not news_path.exists():
        notes.append("GDELT monthly file missing; cannot train news models.")
        base = pd.read_csv(RESULTS / "forecast_all_models.csv", encoding="utf-8-sig")
        out = base[["cluster", "cluster_name", "horizon", "MAE_lgbm", "MAE_lgbm_trends"]].copy()
        out["MAE_lgbm_news"] = np.nan
        out["MAE_lgbm_news_trends"] = np.nan
        out["delta_news_vs_lgbm"] = np.nan
        out["delta_news_vs_trends"] = np.nan
        out["improved_vs_lgbm"] = False
        out["geo_scope"] = "n/a"
        out.to_csv(RESULTS / "forecast_news.csv", index=False, encoding="utf-8-sig")
        NOTE_PATH.write_text("\n".join(notes), encoding="utf-8")
        print("FAIL: no news file")
        return

    agg, spectral, trends_pivot, trends_cols = load_agg_spectral_trends()
    news_lag, news_cols = load_news()
    notes.append(
        "GDELT fetch OK via gdelt package (events v2), sampled days [1,8,15,22]/month."
    )
    notes.append(
        "Geo scope: RUSSIA-WIDE (ActionGeo RS / Actor RUS). NWFD FullName filter sparse "
        f"(see gdelt_news_monthly.csv event_count_nwfd); not used as primary feature."
    )
    notes.append(
        "Anti-leakage: news features are lag-1 (month t uses intensity of month t-1 only)."
    )

    base = pd.read_csv(RESULTS / "forecast_all_models.csv", encoding="utf-8-sig")
    results = []

    for cluster in [0, 1]:
        sub = agg[agg["cluster"] == cluster].sort_values("period").copy()
        sub = sub.rename(columns={"period": "ds", "value": "y"})[["ds", "y"]]
        label = CLUSTER_NAMES[cluster]
        print(f"=== {label} n={len(sub)} ===")

        for H in HORIZONS:
            row = {
                "cluster": cluster,
                "cluster_name": label,
                "horizon": H,
                "geo_scope": "russia_wide",
            }
            b = base[(base["cluster"] == cluster) & (base["horizon"] == H)]
            if len(b):
                row["MAE_lgbm"] = float(b["MAE_lgbm"].iloc[0]) if pd.notna(b["MAE_lgbm"].iloc[0]) else np.nan
                row["MAE_lgbm_trends"] = (
                    float(b["MAE_lgbm_trends"].iloc[0])
                    if pd.notna(b["MAE_lgbm_trends"].iloc[0])
                    else np.nan
                )
            else:
                row["MAE_lgbm"] = np.nan
                row["MAE_lgbm_trends"] = np.nan

            # spectral + news
            try:
                df = make_features(
                    sub, spectral, trends_pivot, trends_cols, news_lag, news_cols,
                    use_trends=False, use_spectral=True, use_news=True,
                )
                if len(df) > H and len(df) >= 10:
                    p, yt = predict_lgbm(df, H)
                    row["MAE_lgbm_news"] = mean_absolute_error(yt, p)
                    row["R2_lgbm_news"] = r2_score(yt, p) if H > 1 else np.nan
                else:
                    row["MAE_lgbm_news"] = np.nan
                    notes.append(f"cluster={cluster} H={H}: too few rows after dropna ({len(df)})")
            except Exception as e:
                row["MAE_lgbm_news"] = np.nan
                notes.append(f"cluster={cluster} H={H} news err: {e}")

            # spectral + news + trends
            try:
                df = make_features(
                    sub, spectral, trends_pivot, trends_cols, news_lag, news_cols,
                    use_trends=True, use_spectral=True, use_news=True,
                )
                if len(df) > H and len(df) >= 10:
                    p, yt = predict_lgbm(df, H)
                    row["MAE_lgbm_news_trends"] = mean_absolute_error(yt, p)
                    row["R2_lgbm_news_trends"] = r2_score(yt, p) if H > 1 else np.nan
                else:
                    row["MAE_lgbm_news_trends"] = np.nan
            except Exception as e:
                row["MAE_lgbm_news_trends"] = np.nan
                notes.append(f"cluster={cluster} H={H} news+trends err: {e}")

            if pd.notna(row.get("MAE_lgbm_news")) and pd.notna(row.get("MAE_lgbm")):
                row["delta_news_vs_lgbm"] = row["MAE_lgbm_news"] - row["MAE_lgbm"]
                row["improved_vs_lgbm"] = row["delta_news_vs_lgbm"] < -1e-6
            else:
                row["delta_news_vs_lgbm"] = np.nan
                row["improved_vs_lgbm"] = False

            if pd.notna(row.get("MAE_lgbm_news")) and pd.notna(row.get("MAE_lgbm_trends")):
                row["delta_news_vs_trends"] = row["MAE_lgbm_news"] - row["MAE_lgbm_trends"]
            else:
                row["delta_news_vs_trends"] = np.nan

            if pd.notna(row.get("MAE_lgbm_news_trends")) and pd.notna(row.get("MAE_lgbm")):
                row["delta_news_trends_vs_lgbm"] = row["MAE_lgbm_news_trends"] - row["MAE_lgbm"]
            else:
                row["delta_news_trends_vs_lgbm"] = np.nan

            results.append(row)
            print(
                f"  H={H}: lgbm={row.get('MAE_lgbm')} trends={row.get('MAE_lgbm_trends')} "
                f"news={row.get('MAE_lgbm_news')} news+tr={row.get('MAE_lgbm_news_trends')} "
                f"d_news={row.get('delta_news_vs_lgbm')}"
            )

    res_df = pd.DataFrame(results)
    # Any meaningful improvement? (>1% relative MAE drop on any cell with finite baseline)
    improved_cells = []
    for _, r in res_df.iterrows():
        if pd.notna(r.get("MAE_lgbm_news")) and pd.notna(r.get("MAE_lgbm")) and r["MAE_lgbm"] > 0:
            rel = (r["MAE_lgbm"] - r["MAE_lgbm_news"]) / r["MAE_lgbm"]
            if rel > 0.01:
                improved_cells.append(
                    f"cluster={r['cluster']} H={r['horizon']} rel_improve={rel:.3f}"
                )
        if pd.notna(r.get("MAE_lgbm_news_trends")) and pd.notna(r.get("MAE_lgbm_trends")) and r["MAE_lgbm_trends"] > 0:
            rel = (r["MAE_lgbm_trends"] - r["MAE_lgbm_news_trends"]) / r["MAE_lgbm_trends"]
            if rel > 0.01:
                improved_cells.append(
                    f"cluster={r['cluster']} H={r['horizon']} news+trends vs trends rel={rel:.3f}"
                )

    if improved_cells:
        notes.append("Meaningful improvements (>1% rel MAE): " + "; ".join(improved_cells))
        notes.append("SLIDE_CANDIDATE: News GDELT — partial wins listed above.")
    else:
        notes.append(
            "NO meaningful win: GDELT russia-wide lag-1 news did not reduce MAE by >1% "
            "vs LGBM spectral (or news+trends vs trends) on evaluated cells."
        )

    res_df.to_csv(RESULTS / "forecast_news.csv", index=False, encoding="utf-8-sig")
    NOTE_PATH.write_text("\n".join(notes), encoding="utf-8")
    print("\n".join(notes))
    print("Saved", RESULTS / "forecast_news.csv")


if __name__ == "__main__":
    main()
