"""Парсинг мобильности → data/intermediate/mobility.parquet"""
import zipfile
import pandas as pd
from utils import DATA_RAW, DATA_INT

ZIP_PATH = DATA_RAW / 'indeks-mobilnosti_ru_1764063764975.csv.zip'
OUT_PATH = DATA_INT / 'mobility.parquet'

def read_zip_csv(zpath, **kwargs):
    with zipfile.ZipFile(zpath) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            try:
                return pd.read_csv(f, sep=None, engine='python', **kwargs)
            except Exception:
                f.seek(0)
                return pd.read_csv(f, sep=';', encoding='cp1251', **kwargs)

def main():
    print(f"Читаю: {ZIP_PATH.name}")
    mob = read_zip_csv(ZIP_PATH)
    print(f"Shape: {mob.shape}")
    print(f"МО: {mob['ref_area'].nunique()}")
    print(f"Периоды: {sorted(mob['period'].unique())}")

    mob.to_parquet(OUT_PATH)
    print(f"\n✓ Сохранено: {OUT_PATH}")

if __name__ == '__main__':
    main()
