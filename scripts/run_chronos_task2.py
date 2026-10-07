"""Zero-shot Chronos on Task 2 cluster aggregates. Budget: ~1h.

Writes results/forecast_foundation.csv with MAE_chronos vs holdout protocol
matching src/12_timeseries.py (last H months).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import DATA_INT, DATA_PROC, RESULTS, norm_name  # noqa: E402

HORIZONS = [1, 3, 6, 12]
CLUSTER_NAMES = {0: "сельские", 1: "городские"}
# Prefer bolt tiny for speed on short series; fall back to t5-tiny
MODEL_CANDIDATES = [
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-t5-tiny",
]


def load_agg() -> pd.DataFrame:
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
    return agg.sort_values(["cluster", "period"])


def load_pipeline():
    from chronos import BaseChronosPipeline
    import torch

    last_err = None
    for mid in MODEL_CANDIDATES:
        try:
            print(f"Loading {mid} ...")
            pipe = BaseChronosPipeline.from_pretrained(
                mid,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            print(f"OK model={mid}")
            return pipe, mid
        except Exception as e:
            print(f"FAIL {mid}: {e}")
            last_err = e
    raise RuntimeError(f"Could not load Chronos: {last_err}")


def forecast_mae(pipe, y_train: np.ndarray, y_true: np.ndarray, H: int) -> float:
    import torch
    from sklearn.metrics import mean_absolute_error

    context = torch.tensor(y_train, dtype=torch.float32)
    # predict returns samples or quantiles depending on pipeline type
    out = pipe.predict(context, prediction_length=H)
    # ChronosPipeline: (num_samples, H); ChronosBolt: often (1, H) or quantiles
    arr = np.asarray(out, dtype=float)
    if arr.ndim == 3:
        # (batch, samples/quantiles, H)
        pred = arr[0].mean(axis=0)
    elif arr.ndim == 2:
        # (samples, H) or (1, H)
        pred = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
    else:
        pred = arr.reshape(-1)[:H]
    pred = pred[:H]
    return float(mean_absolute_error(y_true, pred))


def main():
    t0 = time.time()
    agg = load_agg()
    print(f"agg shape={agg.shape}")
    pipe, model_id = load_pipeline()

    # baseline columns from canon CSV for comparison
    canon = pd.read_csv(RESULTS / "forecast_all_models.csv")

    rows = []
    for cluster in [0, 1]:
        sub = agg[agg["cluster"] == cluster].sort_values("period")
        y = sub["value"].values.astype(float)
        label = CLUSTER_NAMES[cluster]
        print(f"\n=== {label} n={len(y)} ===")
        for H in HORIZONS:
            y_train, y_true = y[:-H], y[-H:]
            try:
                mae = forecast_mae(pipe, y_train, y_true, H)
                print(f"  H={H}: MAE_chronos={mae:.4f}")
                err = None
            except Exception as e:
                mae = np.nan
                err = str(e)
                print(f"  H={H}: FAIL {e}")
            base = canon[(canon["cluster"] == cluster) & (canon["horizon"] == H)]
            row = {
                "cluster": cluster,
                "cluster_name": label,
                "horizon": H,
                "model_id": model_id,
                "MAE_chronos": mae,
                "error": err,
            }
            if len(base):
                b = base.iloc[0]
                row["MAE_prophet"] = b.get("MAE_prophet")
                row["MAE_lgbm"] = b.get("MAE_lgbm")
                row["MAE_naive"] = b.get("MAE_naive")
            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = RESULTS / "forecast_foundation.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(out.to_string(index=False))
    print(f"Elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
