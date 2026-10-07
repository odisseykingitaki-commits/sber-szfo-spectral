"""Парсинг Росстат urov_2010-2024.xlsx → data/intermediate/urov_parsed.parquet"""
import re
import pandas as pd
from utils import DATA_RAW, DATA_INT

XLSX_PATH = DATA_RAW / 'urov_2010-2024.xlsx'
OUT_PATH = DATA_INT / 'urov_parsed.parquet'

SZFO_KEYS = {
    'санкт-петербург': 'СПб',
    'ленинградская': 'ЛенОбл',
    'новгородская': 'НовОбл',
    'псковская': 'ПскОбл',
    'мурманская': 'МурОбл',
    'архангельская': 'АрхОбл',
    'вологодская': 'ВолОбл',
    'калининградская': 'КалОбл',
    'карелия': 'Карелия',
    'коми': 'Коми',
    'ненецкий': 'НАО',
}

def detect_szfo(text):
    if not isinstance(text, str):
        return None
    t = text.lower().replace('ё', 'е')
    for key, short in SZFO_KEYS.items():
        if key in t:
            return short
    return None

def looks_like_region(text):
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(kw in t for kw in [
        'край', 'область', 'республика', 'автономн',
        'санкт-петербург', 'москва', 'севастополь'
    ])

def is_header(text):
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(kw in t for kw in [
        'муниципальные округа', 'муниципальные районы',
        'городские округа', 'муниципальные образования',
        'содержание', 'основные понятия', 'внутригородские'
    ])

def parse_sheet(xl, year):
    df = xl.parse(str(year), header=None)
    rows = []
    current_region = None

    for i in range(len(df)):
        r = df.iloc[i].tolist()
        while len(r) < 4:
            r.append(None)

        # Строка-регион
        if pd.isna(r[0]) and isinstance(r[1], str) and pd.isna(r[3]):
            text = r[1].strip()
            if is_header(text):
                continue
            if looks_like_region(text):
                current_region = detect_szfo(text)
            continue

        # Строка МО
        if pd.notna(r[3]) and pd.notna(r[1]) and isinstance(r[1], str):
            if current_region is None:
                continue
            name = r[1].strip()
            if not name or is_header(name):
                continue
            try:
                value = float(r[3])
            except (ValueError, TypeError):
                continue
            oktmo = None
            if pd.notna(r[2]):
                o = str(r[2]).replace(' ', '').replace('\xa0', '').strip()
                if o.isdigit():
                    oktmo = o
            rows.append({
                'year': int(year),
                'region': current_region,
                'name_raw': name,
                'oktmo': oktmo,
                'income_thousand_rub': value,
            })
    return pd.DataFrame(rows)

def main():
    print(f"Читаю: {XLSX_PATH.name}")
    xl = pd.ExcelFile(XLSX_PATH)

    all_data = []
    for year in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        df_y = parse_sheet(xl, year)
        print(f"  {year}: {len(df_y)} строк СЗФО")
        all_data.append(df_y)

    urov = pd.concat(all_data, ignore_index=True)
    print(f"\nВсего: {len(urov)} строк")
    print(f"Уникальных МО: {urov.groupby(['region', 'name_raw']).ngroups}")

    urov.to_parquet(OUT_PATH)
    print(f"\n✓ Сохранено: {OUT_PATH}")

if __name__ == '__main__':
    main()
