#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
volatility.py — анализатор волатильности инструментов

Скрипт позволяет загружать данные с Yahoo Finance или из CSV,
рассчитывать волатильность различными методами (Close-to-Close,
Parkinson, Garman–Klass, Rogers–Satchell, ATR), строить графики
и сохранять результаты.
"""

import argparse
import sys
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# yfinance импортируем только при необходимости
try:
    import yfinance as yf
except ImportError:
    yf = None


# ----------------------------- Парсер аргументов -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Volatility Analyzer — расчет и визуализация волатильности"
    )
    parser.add_argument("--ticker", type=str, help="Тикер (например AAPL, BTC-USD)")
    parser.add_argument("--source", choices=["yfinance", "csv"], default="yfinance",
                        help="Источник данных (по умолчанию yfinance)")
    parser.add_argument("--file", type=str, help="Путь к CSV файлу (для --source csv)")
    parser.add_argument("--method", type=str,
                        default="cc,parkinson,gk,rs,atr",
                        help="Методы через запятую (cc,parkinson,gk,rs,atr)")
    parser.add_argument("--window", type=int, default=None,
                        help="Окно для скользящей оценки (в днях)")
    parser.add_argument("--period", type=str, default="1y",
                        help="Период для yfinance (например 1y,6mo,3mo)")
    parser.add_argument("--start", type=str, help="Начальная дата YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Конечная дата YYYY-MM-DD")
    parser.add_argument("--trading-days", type=int, default=252,
                        help="Торговых дней в году (по умолчанию 252)")
    parser.add_argument("--out-metrics", type=str,
                        help="Файл для сохранения метрик (csv или json)")
    parser.add_argument("--out-chart", type=str,
                        help="Файл для сохранения графика (png)")
    parser.add_argument("--candles", action="store_true",
                        help="Строить свечи вместо линейного графика цены")
    parser.add_argument("--daily", action="store_true",
                        help="Выводить дневную волатильность (не годовую)")
    parser.add_argument("--date-format", type=str, default=None,
                        help="Формат даты для CSV (например %%Y-%%m-%%d)")
    parser.add_argument("--float-format", type=str, default=".4f",
                        help="Формат чисел при печати")
    parser.add_argument("--quiet", action="store_true",
                        help="Минимум логов")
    parser.add_argument("--print-series", action="store_true",
                        help="Вывод последних значений временных рядов")
    return parser.parse_args()


# ----------------------------- Загрузка данных -----------------------------
def load_data_from_yf(ticker, period, start, end):
    if yf is None:
        logging.error("yfinance не установлен, используйте --source csv")
        sys.exit(1)
    try:
        df = yf.download(ticker, period=period if not start else None,
                         start=start, end=end, progress=False)
    except Exception as e:
        logging.error(f"Ошибка загрузки с yfinance: {e}")
        sys.exit(1)
    return df


def load_data_from_csv(path, date_format=None):
    try:
        df = pd.read_csv(path, parse_dates=["Date"] if not date_format else None)
        if date_format:
            df["Date"] = pd.to_datetime(df["Date"], format=date_format)
        df = df.set_index("Date")
    except Exception as e:
        logging.error(f"Ошибка чтения CSV: {e}")
        sys.exit(1)
    return df


def validate_ohlc(df):
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        logging.error(f"Отсутствуют колонки: {missing}")
        sys.exit(2)
    df.dropna(subset=list(needed), inplace=True)
    return df


# ----------------------------- Методы волатильности -----------------------------
def vol_cc(df, window, D, daily=False):
    r = np.log(df["Close"] / df["Close"].shift(1))
    if window:
        series = r.rolling(window).std()
    else:
        series = pd.Series(r.std(), index=df.index)
    return series if daily else series * np.sqrt(D)


def vol_parkinson(df, window, D, daily=False):
    rs = (np.log(df["High"] / df["Low"])) ** 2
    if window:
        series = np.sqrt(rs.rolling(window).mean() / (4 * np.log(2)))
    else:
        series = pd.Series(np.sqrt(rs.mean() / (4 * np.log(2))), index=df.index)
    return series if daily else series * np.sqrt(D)


def vol_gk(df, window, D, daily=False):
    term = 0.5 * (np.log(df["High"] / df["Low"])) ** 2 \
        - (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"])) ** 2
    if window:
        series = np.sqrt(term.rolling(window).mean())
    else:
        series = pd.Series(np.sqrt(term.mean()), index=df.index)
    return series if daily else series * np.sqrt(D)


def vol_rs(df, window, D, daily=False):
    term = (np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"])
            + np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"]))
    if window:
        series = np.sqrt(term.rolling(window).mean())
    else:
        series = pd.Series(np.sqrt(term.mean()), index=df.index)
    return series if daily else series * np.sqrt(D)


def vol_atr(df, window, D, daily=False):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    if window:
        atr = tr.rolling(window).mean()
    else:
        atr = pd.Series(tr.mean(), index=df.index)
    rel_vol = atr / df["Close"].rolling(window).mean() if window else atr / df["Close"].mean()
    return rel_vol if daily else rel_vol * np.sqrt(D)


# ----------------------------- Вывод и сохранение -----------------------------
def build_metrics_table(vol_dict, window):
    rows = []
    for method, series in vol_dict.items():
        if series.notna().sum() == 0:
            continue
        val = series.dropna().iloc[-1]
        rows.append({
            "method": method,
            "window": window if window else "-",
            "sample_size": series.notna().sum(),
            "annualized_vol": val
        })
    return pd.DataFrame(rows)


def save_metrics(vol_dict, metrics, path):
    if path.endswith(".csv"):
        metrics.to_csv(path, index=False)
    elif path.endswith(".json"):
        out = {
            "metrics": metrics.to_dict(orient="records"),
            "series": {m: s.dropna().to_dict() for m, s in vol_dict.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        logging.error("Формат файла должен быть .csv или .json")


def plot_chart(df, vol_dict, args):
    fig, ax1 = plt.subplots(2 if vol_dict else 1, 1, figsize=(10, 6),
                            sharex=True, gridspec_kw={'height_ratios': [2, 1]} if vol_dict else None)

    if not vol_dict:
        ax1 = [ax1]

    # Верхняя панель — цена
    ax_price = ax1[0]
    if args.candles:
        # Упрощённые свечи
        for idx, row in df.iterrows():
            color = "g" if row["Close"] >= row["Open"] else "r"
            ax_price.plot([idx, idx], [row["Low"], row["High"]], color=color)
            ax_price.add_patch(
                plt.Rectangle((idx, min(row["Open"], row["Close"])),
                              0.5, abs(row["Close"] - row["Open"]),
                              facecolor=color, edgecolor=color)
            )
    else:
        ax_price.plot(df.index, df["Close"], label="Close")

    ax_price.set_title(f"{args.ticker} Price & Volatility")
    ax_price.legend()

    # Нижняя панель — волатильность
    if vol_dict:
        ax_vol = ax1[1]
        for m, s in vol_dict.items():
            ax_vol.plot(s.index, s, label=m)
        ax_vol.set_ylabel("Volatility")
        ax_vol.legend()
        ax_vol.grid(True)

    plt.tight_layout()
    if args.out_chart:
        plt.savefig(args.out_chart)
    else:
        plt.show()


# ----------------------------- Главная функция -----------------------------
def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")

    # Загрузка данных
    if args.source == "yfinance":
        if not args.ticker:
            logging.error("Для yfinance нужно указать --ticker")
            sys.exit(2)
        df = load_data_from_yf(args.ticker, args.period, args.start, args.end)
    else:
        if not args.file:
            logging.error("Для csv нужно указать --file")
            sys.exit(2)
        df = load_data_from_csv(args.file, args.date_format)

    if df.empty:
        logging.error("Нет данных для анализа")
        sys.exit(2)

    df = validate_ohlc(df)

    # Методы
    methods = list(dict.fromkeys([m.strip() for m in args.method.split(",")]))
    funcs = {
        "cc": vol_cc,
        "parkinson": vol_parkinson,
        "gk": vol_gk,
        "rs": vol_rs,
        "atr": vol_atr
    }

    vol_dict = {}
    for m in methods:
        if m not in funcs:
            logging.warning(f"Метод {m} не поддерживается")
            continue
        series = funcs[m](df, args.window, args.trading_days, daily=args.daily)
        vol_dict[m] = series

    metrics = build_metrics_table(vol_dict, args.window)

    # Печать результатов
    if not metrics.empty:
        print(metrics.to_string(index=False,
                                float_format=lambda x: format(x, args.float_format)))
    else:
        logging.error("Не удалось рассчитать волатильность")
        sys.exit(2)

    # Дополнительный вывод
    if args.print_series:
        for m, s in vol_dict.items():
            print(f"\nПоследние значения {m}:")
            print(s.dropna().tail(5).to_string(float_format=lambda x: format(x, args.float_format)))

    # Сохранение
    if args.out_metrics:
        save_metrics(vol_dict, metrics, args.out_metrics)

    # График
    plot_chart(df, vol_dict if args.window else {}, args)


if __name__ == "__main__":
    main()