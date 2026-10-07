"""Запуск всего пайплайна по порядку.

11 (threshold = основной метод) идёт до 08 (ICVI),
чтобы labels_threshold.npy был доступен для «Наш».

После задачи 1 (01–11): задача 2 — прогноз (12) и сдвиги (13).
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    '01_parse_spend.py',
    '02_parse_mobility.py',
    '03_parse_urov.py',
    '04_features.py',
    '05_plm_spectral.py',
    '06_network_louvain.py',   # → labels_louvain.npy (+legacy labels_final.npy)
    '11_final_clusters.py',    # → labels_threshold.npy (основной; до ICVI)
    '07_dynamics.py',
    '08_icvi.py',              # «Наш» = threshold; Louvain — отдельно
    '09_bootstrap.py',         # threshold on U1
    '10_robustness.py',        # threshold on U1
    # --- Задача 2 ---
    '12_timeseries.py',        # прогноз расходов (+ Google Trends)
    '13_changepoints.py',      # структурные сдвиги (ruptures + CUSUM)
]

ROOT = Path(__file__).resolve().parent

for s in SCRIPTS:
    print(f"\n{'='*70}\n>>> {s}\n{'='*70}")
    r = subprocess.run([sys.executable, str(ROOT / 'src' / s)], cwd=str(ROOT / 'src'))
    if r.returncode != 0:
        print(f"\n❌ ОШИБКА в {s}. Остановка.")
        sys.exit(1)

print(f"\n{'='*70}")
print("✅ ПАЙПЛАЙН ЗАВЕРШЁН")
print(f"{'='*70}")
