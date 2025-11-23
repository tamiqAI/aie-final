import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set(style="whitegrid")

def make_boxplot(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f"Boxplot — {column}")
    return fig


def make_trend(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Datetime'], df[column], marker='.', linewidth=0.7)
    ax.set_title(f"Trend — {column}")
    plt.xticks(rotation=45)
    return fig


def make_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax)
    ax.set_title('Correlation matrix')
    return fig
