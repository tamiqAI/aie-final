import pandas as pd
from scipy.stats import ttest_ind

def perform_ttest(df: pd.DataFrame, column: str, h1:int, h2:int):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError('Колонка не найдена или не числовая')

    group1 = df[df['Datetime'].dt.hour == h1][column].dropna()
    group2 = df[df['Datetime'].dt.hour == h2][column].dropna()

    if len(group1) < 2 or len(group2) < 2:
        raise ValueError('Слишком мало наблюдений для проведения t-test')

    t, p = ttest_ind(group1, group2)
    return t, p
