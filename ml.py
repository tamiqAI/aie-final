import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def linear_forecast(df: pd.DataFrame, column: str, window:int=200):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError('Колонка не найдена или не числовая')

    ts = df[['Datetime', column]].dropna().sort_values('Datetime').reset_index(drop=True)
    if len(ts) < 10:
        raise ValueError('Недостаточно наблюдений для обучения')

    X = np.arange(len(ts))[-window:].reshape(-1,1)
    y = ts[column].values[-window:]

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(ts), len(ts)+1).reshape(-1,1)
    future_pred = model.predict(future_t)[0]

    ts['t'] = np.arange(len(ts))
    ts['forecast'] = model.predict(ts[['t']])

    return ts, future_pred
