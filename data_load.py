import pandas as pd
import numpy as np
from io import StringIO
from constants import DEFAULT_PATH

import streamlit as st

@st.cache_data
def load_and_process(path: str = DEFAULT_PATH):
    try:
        df = pd.read_csv(path, sep=';')
    except FileNotFoundError:
        raise
    except Exception as e:
        raise

    df.columns = [str(col).strip().replace('\xa0', '') for col in df.columns]

    if 'Time' in df.columns:
        df['Time'] = df['Time'].astype(str).str.replace('.', ':', regex=False)
    else:
        df['Time'] = '00:00:00'

    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    else:
        df['Date'] = ''

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Datetime']).reset_index(drop=True)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols) if empty_cols else df

    possible_num = [c for c in df.select_dtypes(include=['object']).columns if c not in ['Date', 'Time']]

    for col in possible_num:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '.')
            .replace({'': np.nan, 'nan': np.nan})
        )

        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(-200, np.nan)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    iter_count = 0
    while True:
        prev_na = df[num_cols].isna().sum().sum()

        for col in num_cols:
            df = df.set_index('Datetime')
            df[col] = df[col].interpolate(method='time', limit_direction='both')
            df = df.reset_index()

        for col in num_cols:
            mean_val = df[col].mean()
            if np.isfinite(mean_val):
                df[col] = df[col].fillna(mean_val)

        curr_na = df[num_cols].isna().sum().sum()
        iter_count += 1

        if curr_na == 0 or iter_count >= 5:
            break

    df = df.sort_values('Datetime').reset_index(drop=True)

    meta = {
        'original_columns': df.columns.tolist(),
        'numeric_columns': num_cols,
        'removed_empty_columns': empty_cols
    }

    return df, meta
