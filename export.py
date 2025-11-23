import pandas as pd
import json

def save_head_json(df: pd.DataFrame, n:int=20):
    records = df.head(n).copy()

    for col in records.select_dtypes(include=['datetime']):
        records[col] = records[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    return json.dumps(records.to_dict(orient='records'), ensure_ascii=False, indent=2)
