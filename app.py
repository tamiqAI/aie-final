import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

from constants import DEFAULT_PATH
from data_loader import load_and_process
from plots import make_boxplot, make_trend, make_heatmap
from transforms import log_transform, cap_outliers
from ml import linear_forecast
from stats_tests import perform_ttest
from export import save_head_json


def main():
    st.set_page_config(page_title='Air Quality Dashboard', layout='wide')
    st.title('Air Quality interactive analysis')

    st.sidebar.header('Data & Controls')
    uploaded = st.sidebar.file_uploader('Загрузить CSV', type=['csv'])
    use_default = st.sidebar.checkbox(f'Использовать {DEFAULT_PATH} из рабочей папки', value=True)

    df = None
    meta = {}
    try:
        if uploaded is not None:
            file_bytes = uploaded.getvalue().decode('utf-8')
            df, meta = load_and_process(StringIO(file_bytes))
        elif use_default:
            df, meta = load_and_process(DEFAULT_PATH)
        else:
            st.info('Загрузить файл или DEFAULT_PATH')
            return
    except FileNotFoundError:
        st.error(f'Файл {DEFAULT_PATH} не найден.')
        return
    except Exception as e:
        st.error(f'Ошибка при загрузке/обработке: {e}')
        return

    st.sidebar.subheader('Инфо по данным')
    st.sidebar.write(f"число строк: {len(df)}")
    st.sidebar.write(f"число числовых колонок: {len(meta['numeric_columns'])}")
    st.sidebar.write('Удалённые пустые колонки:')
    st.sidebar.write(meta['removed_empty_columns'])

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    st.sidebar.markdown('---')
    action = st.sidebar.selectbox('Выберите действие', [
        'EDA:Описательная статистика',
        'EDA:Пропуски и типы',
        'Визуализации:Корреляции',
        'Визуализации:Boxplot',
        'Визуализации:Trend',
        'Преобразования:Log transform',
        'Преобразования:Cap outliers',
        'ML:Linear forecast',
        'Гипотеза:T-test',
        'Экспорт:Сохранить JSON'
    ])

    history = []
    results = {}

    if action == 'EDA:Описательная статистика':
        st.header('Описательная статистика (после очистки)')
        st.dataframe(df.describe())
        history.append('describe')

    elif action == 'EDA:Пропуски и типы':
        st.header('Пропуски')
        st.dataframe(df.isna().sum())
        st.header('Типы признаков')
        st.dataframe(df.dtypes)
        history.append('missing_types')

    elif action == 'Визуализации:Корреляции':
        st.header('Корреляционная матрица')
        fig = make_heatmap(df)
        st.pyplot(fig)
        history.append('heatmap')

    elif action == 'Визуализации:Boxplot':
        st.header('Boxplot')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Выберите колонку', numeric_cols)
            fig = make_boxplot(df, col)
            st.pyplot(fig)
            history.append(('box', col))

    elif action == 'Визуализации:Trend':
        st.header('Trend plot')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Выберите колонку', numeric_cols)
            fig = make_trend(df, col)
            st.pyplot(fig)
            history.append(('trend', col))

    elif action == 'Преобразования:Log transform':
        st.header('Log-transform (log1p)')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Колонка для log', numeric_cols)
            if st.button('Применить log1p'):
                try:
                    new_col = log_transform(df, col)
                    st.success(f'Добавлен столбец {new_col}')
                    fig = make_boxplot(df, new_col)
                    st.pyplot(fig)
                    results['last_log'] = new_col
                    history.append(('log', col))
                except Exception as e:
                    st.error(f'Ошибка: {e}')

    elif action == 'Преобразования:Cap outliers':
        st.header('Cap outliers')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Колонка для capping', numeric_cols)
            pct = st.slider('Процентиль для capping', 80, 99, 95)
            if st.button('Применить capping'):
                try:
                    new_col, cap = cap_outliers(df, col, percentile=pct)
                    st.success(f'Добавлен столбец {new_col}, cap value={cap:.3f}')
                    fig = make_boxplot(df, new_col)
                    st.pyplot(fig)
                    results['last_cap'] = (new_col, cap)
                    history.append(('cap', col, pct))
                except Exception as e:
                    st.error(f'Ошибка: {e}')

    elif action == 'ML:Linear forecast':
        st.header('Линейный прогноз по времени')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Колонка для прогноза', numeric_cols)
            window = st.number_input('Window size (последние n точек)', min_value=10, max_value=2000, value=200)
            if st.button('Обучить и показать прогноз'):
                try:
                    ts, fut = linear_forecast(df, col, window=window)
                    fig = plt.figure(figsize=(10,4))
                    ax = fig.add_subplot(111)
                    ax.plot(ts['Datetime'], ts[col], label='Real')
                    ax.plot(ts['Datetime'], ts['forecast'], label='Forecast')
                    ax.legend()
                    ax.set_title(f'Linear forecast — {col} | next_pred={fut:.3f}')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    results['linear_forecast'] = {'col': col, 'next_pred': float(fut)}
                    history.append(('ml', col))
                except Exception as e:
                    st.error(f'Ошибка при обучении: {e}')

    elif action == 'Гипотеза:T-test':
        st.header('T-test по часам')
        if not numeric_cols:
            st.warning('Нет числовых колонок')
        else:
            col = st.selectbox('Колонка для t-test', numeric_cols)
            h1 = st.slider('Час 1', 0, 23, 8)
            h2 = st.slider('Час 2', 0, 23, 17)
            if st.button('Провести t-test'):
                try:
                    t, p = perform_ttest(df, col, h1, h2)
                    st.write(f't = {t:.4f}, p = {p:.4e}')
                    results['ttest'] = {'col': col, 'h1': h1, 'h2': h2, 't': float(t), 'p': float(p)}
                    history.append(('ttest', col, h1, h2))
                except Exception as e:
                    st.error(f'Ошибка: {e}')

    elif action == 'Экспорт:Сохранить JSON':
        st.header('Экспорт')
        n = st.number_input('Число строк для экспорта', min_value=1, max_value=500, value=20)
        j = save_head_json(df, n)
        st.text_area('JSON (head)', j, height=200)
        st.download_button('Скачать JSON', data=j, file_name='head.json', mime='application/json')
        history.append(('export', n))

    st.sidebar.markdown('---')
    st.sidebar.subheader('История действий (последние 10)')
    if history:
        for item in history[-10:]:
            st.sidebar.write(item)
    else:
        st.sidebar.write('нет действий')

    st.sidebar.markdown('---')
    st.sidebar.subheader('Результаты')
    st.sidebar.write(results)


if __name__ == '__main__':
    main()

