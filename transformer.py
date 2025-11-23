import numpy as np
import pandas as pd

def log_transform(df: pd.DataFrame, column: str):
    new_col = column + '_log'
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError('Колонка не найдена или не числовая')
    df[new_col] = np.log1p(df[column].clip(lower=0))
    return new_col


def cap_outliers(df: pd.DataFrame, column: str, percentile: float = 95.0):
    new_col = column + f'_capped_{int(percentile)}'
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError('Колонка не найдена или не числовая')

    cap_value = df[column].quantile(percentile/100.0)
    df[new_col] = df[column].copy()

    capped_values = []
    for v in df[new_col]:
        if pd.isna(v):
            capped_values.append(np.nan)
        elif v > cap_value:
            capped_values.append(cap_value)
        else:
            capped_values.append(v)

    df[new_col] = capped_values
    return new_col, cap_value
