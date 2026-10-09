"""Запуск пайплайна типологии по порядку.

11 (threshold = основной метод) идёт до 08 (ICVI),
чтобы labels_threshold.npy был доступен для строки «Наш».
"""
import os
import subprocess
import sys
from pathlib import Path

# Windows cp1251 console cannot print ✓/❌ — force UTF-8 for child scripts
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
