"""Общие функции для всего пайплайна."""
import re
from pathlib import Path
import numpy as np

# ==== Пути ====
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_INT = ROOT / 'data' / 'intermediate'
DATA_PROC = ROOT / 'data' / 'processed'
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'
CONFIGS = ROOT / 'configs'

for d in [DATA_RAW, DATA_INT, DATA_PROC, FIGURES, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)


def load_config(name='config.yaml'):
    """Опциональная загрузка configs/*.yaml (PyYAML).

    Скрипты пайплайна по умолчанию используют константы в коде —
    YAML здесь источник истины для жюри / документации, не ломает прогоны.
    """
    path = CONFIGS / name
    try:
        import yaml
    except ImportError:
        return None
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_methods():
    """Shortcut: configs/methods.yaml или None."""
    return load_config('methods.yaml')

# Meta / label columns that must never enter PLM feature matrix
META_COLS = {
    'mo_norm', 'kmeans_cluster', 'income_source',
    'new_cluster', 'cluster', 'income_matched',
}


def feature_cols(df):
    """Numeric feature columns only (excludes meta/flags)."""
    return [
        c for c in df.columns
        if c not in META_COLS and np.issubdtype(df[c].dtype, np.number)
    ]

# ==== Нормализация названий МО ====
def norm_name(s):
    """Приводит названия МО к общему виду."""
    s = str(s).lower().strip()
    s = re.sub(r'внутригородская территория города федерального значения', '', s)
    s = re.sub(
        r'муниципальный район|городской округ|муниципальное образование|'
        r'муниципальный округ|городское поселение|сельское поселение|'
        r'район|округ|поселение|город|г\.',
        '', s
    )
    return re.sub(r'\s+', ' ', s).strip()

# ==== PLM ====
def build_J(X_bin, C_reg=0.2):
    """PLM: строит матрицу условных зависимостей J (p×p)."""
    from sklearn.linear_model import LogisticRegression
    p = X_bin.shape[1]
    J = np.zeros((p, p))
    for j in range(p):
        y = X_bin[:, j]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        Xo = np.delete(X_bin, j, axis=1)
        try:
            m = LogisticRegression(
                penalty='l2', C=C_reg, solver='lbfgs',
                max_iter=3000, tol=1e-6
            ).fit(Xo, y)
            J[j, np.arange(p) != j] = m.coef_[0]
        except Exception:
            pass
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return J

# ==== Спектральные метрики ====
def spectral_metrics(evals, p):
    """PR, Frustration из спектра."""
    pos = evals[evals > 0]
    neg = evals[evals < 0]
    PR_plus = (pos.sum())**2 / (pos**2).sum() if len(pos) else 0.0
    frustration = len(neg) / p
    return PR_plus, frustration, len(pos), len(neg)
