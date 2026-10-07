"""Сеть МО + Louvain → data/processed/clusters_final_louvain.csv"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (pairwise_distances, adjusted_rand_score,
                             silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.cluster import KMeans
from scipy.linalg import eigh
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt

from utils import DATA_PROC, RESULTS, FIGURES, feature_cols

FEAT_PATH = DATA_PROC / 'features_szfo_v2_final.csv'
J_PATH = RESULTS / 'J_v2.npy'
OUT_CSV = DATA_PROC / 'clusters_final_louvain.csv'
OUT_A = RESULTS / 'adjacency_final.npy'
OUT_LAB = RESULTS / 'labels_louvain.npy'  # canonical Louvain labels
OUT_LAB_LEGACY = RESULTS / 'labels_final.npy'  # deprecated alias (= Louvain); НЕ «наш»
OUT_FIG = FIGURES / 'fig_network.png'

def main():
    df = pd.read_csv(FEAT_PATH, encoding='utf-8-sig')
    mo_names = df['mo_norm'].values
    feat = feature_cols(df)
    X = StandardScaler().fit_transform(df[feat].values)
    N, p = X.shape

    J = np.load(J_PATH)
    evals, evecs = eigh(J)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    V_k = evecs[:, :5]
    U = X @ V_k
    D = pairwise_distances(U)

    print("=== Подбор порога для связного графа ===")
    print(f"{'кв.':>6} {'eps':>8} {'рёбер':>8} {'компонент':>11}")
    print("-" * 40)
    best_eps = None
    best_q = None
    for q in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        eps = np.quantile(D[D > 0], q)
        A_tmp = (D < eps).astype(int)
        np.fill_diagonal(A_tmp, 0)
        G_tmp = nx.from_numpy_array(A_tmp)
        comps = list(nx.connected_components(G_tmp))
        print(f"{q:>6.2f} {eps:>8.3f} {A_tmp.sum()//2:>8} {len(comps):>11}")
        if len(comps) == 1 and best_eps is None:
            best_eps = eps
            best_q = q

    if best_eps is None:
        best_eps = np.quantile(D[D > 0], 0.40)
        best_q = 0.40
        print(f"⚠️ Связного графа нет, берём q=0.40")

    print(f"\n✅ eps={best_eps:.3f} (q={best_q})")

    A = (D < best_eps).astype(int)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")

    communities = louvain_communities(G, seed=42, resolution=0.5)
    communities = sorted(communities, key=len, reverse=True)
    print(f"Всего сообществ: {len(communities)}")
    for i, c in enumerate(communities):
        print(f"  {i}: n={len(c)}")

    MIN_SIZE = 10
    big = [c for c in communities if len(c) >= MIN_SIZE]
    small_nodes = set()
    for c in communities:
        if len(c) < MIN_SIZE:
            small_nodes.update(c)

    labels_louvain = np.full(N, -1)
    for i, com in enumerate(big):
        for node in com:
            labels_louvain[node] = i
    if small_nodes:
        n_big = len(big)
        for node in small_nodes:
            labels_louvain[node] = n_big
        n_final = n_big + 1
    else:
        n_final = len(big)

    print(f"\n✅ Louvain кластеров: {n_final}")
    sizes = np.bincount(labels_louvain)
    print(f"Размеры: {sizes.tolist()}")

    sw = silhouette_score(X, labels_louvain)
    ch = calinski_harabasz_score(X, labels_louvain)
    db = davies_bouldin_score(X, labels_louvain)
    print(f"SW={sw:+.3f}, CH={ch:.1f}, DB={db:.3f}")

    np.save(OUT_A, A)
    np.save(OUT_LAB, labels_louvain)
    # deprecated alias: старое имя labels_final.npy = Louvain (НЕ «наш»/threshold)
    np.save(OUT_LAB_LEGACY, labels_louvain)
    df['louvain'] = labels_louvain
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=1/np.sqrt(N))
    cmap = plt.cm.tab10
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=labels_louvain, cmap=cmap,
                           node_size=30, alpha=0.8, ax=ax)
    ax.set_title(f'Сеть МО СЗФО (eps={best_eps:.2f}, {G.number_of_edges()} рёбер)',
                 fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Сохранено: {OUT_CSV}, {OUT_A}, {OUT_LAB} "
          f"(+legacy {OUT_LAB_LEGACY.name}), {OUT_FIG}")

if __name__ == '__main__':
    main()
