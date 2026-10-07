"""A) name_norm duplicates + sensitivity WITH/WITHOUT SPb (imputed income)."""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh

from utils import (norm_name, DATA_INT, DATA_PROC, RESULTS,
                   build_J, spectral_metrics, feature_cols)

C_REG = 0.2
N_BOOT = 20  # sample bootstrap for sensitivity (full 100 is in 09)


def threshold_cluster(X, X_bin, C_reg=C_REG):
    J = build_J(X_bin, C_reg=C_reg)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    v1 = evecs[:, 0]
    U1 = X @ v1
    labels = (U1 > np.median(U1)).astype(int)
    return labels, evals, J


def metrics_bundle(X, X_bin, label='full'):
    labels, evals, J = threshold_cluster(X, X_bin)
    p = X.shape[1]
    PR, frustr, n_pos, n_neg = spectral_metrics(evals, p)
    sw = float(silhouette_score(X, labels))
    sizes = np.bincount(labels).tolist()
    out = {
        'label': label,
        'N': int(X.shape[0]),
        'lambda_max': float(evals[0]),
        'PR_plus': float(PR),
        'Frustration': float(frustr),
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
        'SW_threshold': sw,
        'sizes': sizes,
    }
    print(f"\n=== {label} (N={out['N']}) ===")
    print(f"  λ_max={out['lambda_max']:+.3f}  PR₊={out['PR_plus']:.2f}  "
          f"Frustration={out['Frustration']:.2f}  SW={sw:+.3f}  sizes={sizes}")
    return out, labels


def main():
    print("=" * 70)
    print("A1) name_norm duplicates across regions in urov")
    print("=" * 70)
    urov = pd.read_parquet(DATA_INT / 'urov_parsed.parquet')
    urov['name_norm'] = urov['name_raw'].apply(norm_name)
    dups = urov.groupby('name_norm')['region'].nunique()
    dups = dups[dups > 1]
    print(f"name_norm with >1 region: {len(dups)}")
    print(dups.sort_values(ascending=False).head(20))

    # How many feature rows affected after merge?
    feat = pd.read_csv(DATA_PROC / 'features_szfo_v2_final.csv', encoding='utf-8-sig')
    collision_names = set(dups.index)
    affected = feat['mo_norm'].isin(collision_names).sum()
    print(f"\nFeature rows with multi-region name_norm: {affected} / {len(feat)}")
    if affected:
        print(feat.loc[feat['mo_norm'].isin(collision_names), 'mo_norm'].tolist())

    print("\n" + "=" * 70)
    print("A2) Identify proxy/imputed income rows ≈ Spb")
    print("=" * 70)
    feat = feat.copy()
    if 'income_source' in feat.columns:
        feat['income_matched'] = feat['income_source'] == 'rosstat'
    else:
        urov_wide = urov.pivot_table(
            index=['region', 'name_norm'],
            columns='year',
            values='income_thousand_rub',
            aggfunc='mean'
        ).reset_index()
        urov_wide['log_income_2024'] = (
            np.log1p(urov_wide[2024]) if 2024 in urov_wide.columns else np.nan
        )
        merged = feat[['mo_norm']].merge(
            urov_wide[['name_norm', 'region', 'log_income_2024']],
            left_on='mo_norm', right_on='name_norm', how='left'
        )
        match_any = merged.groupby('mo_norm')['log_income_2024'].apply(
            lambda s: s.notna().any()
        )
        feat['income_matched'] = feat['mo_norm'].map(match_any).fillna(False)

    n_unmatched = int((~feat['income_matched']).sum())
    print(f"Unmatched/proxy MOs: {n_unmatched}")
    unmatched_names = feat.loc[~feat['income_matched'], 'mo_norm'].tolist()
    print("Sample unmatched:", unmatched_names[:30])

    spb_regions = urov['region'].dropna().unique()
    spb_like = [r for r in spb_regions if 'петерб' in str(r).lower() or 'спб' in str(r).lower()]
    print(f"Urov regions looking like Spb: {spb_like}")

    fcols = feature_cols(feat)

    X_full = StandardScaler().fit_transform(feat[fcols].values)
    X_full_bin = (X_full > 0).astype(int)
    m_full, lab_full = metrics_bundle(X_full, X_full_bin, 'with_Spb_full')

    feat_no = feat[feat['income_matched']].copy()
    X_no = StandardScaler().fit_transform(feat_no[fcols].values)
    X_no_bin = (X_no > 0).astype(int)
    m_no, lab_no = metrics_bundle(X_no, X_no_bin, 'without_Spb_proxy')

    # Small bootstrap sample on both for SW/ARI stability feel
    print(f"\n=== Bootstrap sample (n={N_BOOT}) threshold ARI ===")
    from sklearn.utils import resample
    from sklearn.metrics import adjusted_rand_score

    def boot_ari(X, X_bin, labels_ref, n=N_BOOT):
        aris = []
        for i in range(n):
            idx = resample(np.arange(len(X)), n_samples=len(X), replace=True, random_state=i)
            lab_b, _, _ = threshold_cluster(X[idx], X_bin[idx])
            aris.append(adjusted_rand_score(labels_ref[idx], lab_b))
        return float(np.mean(aris)), float(np.std(aris))

    boot_full = boot_ari(X_full, X_full_bin, lab_full)
    boot_no = boot_ari(X_no, X_no_bin, lab_no)
    print(f"  with Spb:    {boot_full[0]:.3f} ± {boot_full[1]:.3f}")
    print(f"  without Spb: {boot_no[0]:.3f} ± {boot_no[1]:.3f}")

    result = {
        'n_name_norm_multi_region': int(len(dups)),
        'top_multi_region': dups.sort_values(ascending=False).head(20).to_dict(),
        'feature_rows_affected_by_collision': int(affected),
        'n_unmatched_imputed': int(n_unmatched),
        'unmatched_names': unmatched_names,
        'with_Spb': m_full,
        'without_Spb': m_no,
        'bootstrap_sample_n': N_BOOT,
        'bootstrap_ARI_with_Spb': {'mean': boot_full[0], 'std': boot_full[1]},
        'bootstrap_ARI_without_Spb': {'mean': boot_no[0], 'std': boot_no[1]},
        'delta': {
            'lambda_max': m_no['lambda_max'] - m_full['lambda_max'],
            'PR_plus': m_no['PR_plus'] - m_full['PR_plus'],
            'Frustration': m_no['Frustration'] - m_full['Frustration'],
            'SW_threshold': m_no['SW_threshold'] - m_full['SW_threshold'],
            'N': m_no['N'] - m_full['N'],
        },
    }
    out_path = RESULTS / 'sensitivity_no_spb.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {out_path}")


if __name__ == '__main__':
    main()
