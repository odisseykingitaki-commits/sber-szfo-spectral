"""Парсинг расходов СберИндекса → data/intermediate/spend.parquet"""
import zipfile
import pandas as pd
from utils import DATA_RAW, DATA_INT

ZIP_PATH = DATA_RAW / 'potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij_ru_1764079373653.csv.zip'
OUT_PATH = DATA_INT / 'spend.parquet'

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
    spend = read_zip_csv(ZIP_PATH)
    print(f"Shape: {spend.shape}")
    print(f"Колонки: {spend.columns.tolist()}")
    print(f"МО: {spend['mo'].nunique()}")
    print(f"Периоды: {spend['period'].min()} .. {spend['period'].max()}")

    spend.to_parquet(OUT_PATH)
    print(f"\n✓ Сохранено: {OUT_PATH}")

if __name__ == '__main__':
    main()
