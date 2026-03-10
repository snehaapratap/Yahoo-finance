from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252


@dataclass
class StrategyResult:
    total_strategy_return: float
    total_buy_hold_return: float
    annualized_strategy_volatility: float
    annualized_buy_hold_volatility: float


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    dirs = {
        "raw": base_dir / "data" / "raw",
        "processed": base_dir / "data" / "processed",
        "reports": base_dir / "reports",
        "figures": base_dir / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def download_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Ticker '{ticker}' is missing required columns: {missing_cols}")

        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df["Ticker"] = ticker
        data[ticker] = df
    return data


def clean_and_align_data(raw_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    cleaned: Dict[str, pd.DataFrame] = {}
    cleaning_report_rows = []

    for ticker, df in raw_data.items():
        work = df.copy()
        rows_before = len(work)
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work = work.dropna(subset=["Date"]).drop_duplicates().sort_values("Date")
        rows_after_dedup = len(work)
        work = work.set_index("Date")

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_before = int(work[numeric_cols].isna().sum().sum())
        work[numeric_cols] = work[numeric_cols].interpolate(method="time").ffill().bfill()
        missing_after = int(work[numeric_cols].isna().sum().sum())

        cleaning_report_rows.append(
            {
                "Ticker": ticker,
                "Rows_Before": rows_before,
                "Rows_After_Dedup": rows_after_dedup,
                "Missing_Before_Fill": missing_before,
                "Missing_After_Fill": missing_after,
            }
        )
        cleaned[ticker] = work

    common_dates = None
    for df in cleaned.values():
        common_dates = set(df.index) if common_dates is None else common_dates.intersection(set(df.index))

    if not common_dates:
        raise ValueError("No common date range exists across the selected tickers.")

    common_index = pd.DatetimeIndex(sorted(common_dates))
    for ticker in cleaned:
        cleaned[ticker] = cleaned[ticker].loc[common_index].copy()
        cleaned[ticker] = cleaned[ticker][~cleaned[ticker].index.duplicated(keep="first")]

    cleaning_report = pd.DataFrame(cleaning_report_rows)
    return cleaned, cleaning_report


def add_features(cleaned_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    featured: Dict[str, pd.DataFrame] = {}
    for ticker, df in cleaned_data.items():
        work = df.copy()
        work["Daily_Return"] = work["Close"].pct_change()
        work["Log_Return"] = np.log(work["Close"] / work["Close"].shift(1))
        work["MA_20"] = work["Close"].rolling(window=20).mean()
        work["MA_50"] = work["Close"].rolling(window=50).mean()
        work["Volatility_30"] = work["Daily_Return"].rolling(window=30).std()
        featured[ticker] = work
    return featured


def compute_basic_analytics(featured_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    returns_df = pd.DataFrame()

    for ticker, df in featured_data.items():
        daily_returns = df["Daily_Return"].dropna()
        mean_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        annualized_vol = std_daily_return * np.sqrt(TRADING_DAYS_PER_YEAR)

        summary_rows.append(
            {
                "Ticker": ticker,
                "Mean_Daily_Return": mean_daily_return,
                "Std_Daily_Return": std_daily_return,
                "Annualized_Volatility": annualized_vol,
            }
        )
        returns_df[ticker] = df["Daily_Return"]

    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    corr_df = returns_df.corr()
    return summary_df, corr_df


def evaluate_ma_crossover_strategy(df: pd.DataFrame) -> Tuple[pd.DataFrame, StrategyResult]:
    work = df.copy()
    work["Signal"] = (work["MA_20"] > work["MA_50"]).astype(int)
    work["Position_Change"] = work["Signal"].diff().fillna(0)
    work["Strategy_Return"] = work["Signal"].shift(1).fillna(0) * work["Daily_Return"].fillna(0)
    work["Buy_Hold_Return"] = work["Daily_Return"].fillna(0)
    work["Strategy_Cumulative_Return"] = (1 + work["Strategy_Return"]).cumprod() - 1
    work["Buy_Hold_Cumulative_Return"] = (1 + work["Buy_Hold_Return"]).cumprod() - 1

    result = StrategyResult(
        total_strategy_return=float(work["Strategy_Cumulative_Return"].iloc[-1]),
        total_buy_hold_return=float(work["Buy_Hold_Cumulative_Return"].iloc[-1]),
        annualized_strategy_volatility=float(work["Strategy_Return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
        annualized_buy_hold_volatility=float(work["Buy_Hold_Return"].std() * np.sqrt(TRADING_DAYS_PER_YEAR)),
    )
    return work, result


def create_visualizations(
    featured_data: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,
    strategy_ticker: str,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 6))
    for ticker, df in featured_data.items():
        plt.plot(df.index, df["Close"], label=ticker, linewidth=1.6)
    plt.title("Closing Price History")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "price_history_all_stocks.png", dpi=200)
    plt.close()

    ma_df = featured_data[strategy_ticker]
    plt.figure(figsize=(12, 6))
    plt.plot(ma_df.index, ma_df["Close"], label=f"{strategy_ticker} Close", linewidth=1.5)
    plt.plot(ma_df.index, ma_df["MA_20"], label="20-day MA", linewidth=1.2)
    plt.plot(ma_df.index, ma_df["MA_50"], label="50-day MA", linewidth=1.2)
    plt.title(f"{strategy_ticker}: Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"moving_averages_{strategy_ticker}.png", dpi=200)
    plt.close()

    # 3) Correlation heatmap of daily returns.
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Daily Return Correlation Matrix")
    plt.tight_layout()
    plt.savefig(figures_dir / "correlation_heatmap.png", dpi=200)
    plt.close()


def create_strategy_plot(strategy_df: pd.DataFrame, ticker: str, figures_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(
        strategy_df.index,
        strategy_df["Strategy_Cumulative_Return"],
        label="MA Crossover Strategy",
        linewidth=1.5,
    )
    plt.plot(
        strategy_df.index,
        strategy_df["Buy_Hold_Cumulative_Return"],
        label="Buy and Hold",
        linewidth=1.5,
    )
    plt.title(f"{ticker}: Strategy vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / f"strategy_vs_buy_hold_{ticker}.png", dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yahoo Finance quantitative assignment solution.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="List of stock tickers (minimum 3).",
    )
    parser.add_argument(
        "--start-date",
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format. Must cover at least 3 years with end-date.",
    )
    parser.add_argument(
        "--end-date",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--strategy-ticker",
        default=None,
        help="Ticker for moving-average crossover strategy. Defaults to first ticker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]
    if len(tickers) < 3:
        raise ValueError("Please provide at least three tickers.")

    strategy_ticker = args.strategy_ticker.upper() if args.strategy_ticker else tickers[0]
    if strategy_ticker not in tickers:
        raise ValueError("--strategy-ticker must be one of the provided --tickers.")

    base_dir = Path(__file__).resolve().parent
    dirs = ensure_directories(base_dir)

    raw_data = download_data(tickers, args.start_date, args.end_date)
    for ticker, df in raw_data.items():
        df.to_csv(dirs["raw"] / f"{ticker}_raw.csv", index=False)

    cleaned_data, cleaning_report = clean_and_align_data(raw_data)
    cleaning_report.to_csv(dirs["reports"] / "cleaning_report.csv", index=False)
    for ticker, df in cleaned_data.items():
        df.to_csv(dirs["processed"] / f"{ticker}_cleaned.csv")

    featured_data = add_features(cleaned_data)
    for ticker, df in featured_data.items():
        df.to_csv(dirs["processed"] / f"{ticker}_featured.csv")

    summary_df, corr_df = compute_basic_analytics(featured_data)
    summary_df.to_csv(dirs["reports"] / "basic_analytics_summary.csv")
    corr_df.to_csv(dirs["reports"] / "correlation_matrix.csv")

    strategy_df, strategy_result = evaluate_ma_crossover_strategy(featured_data[strategy_ticker])
    strategy_df.to_csv(dirs["reports"] / f"strategy_timeseries_{strategy_ticker}.csv")
    pd.DataFrame(
        [
            {
                "Ticker": strategy_ticker,
                "Total_Strategy_Return": strategy_result.total_strategy_return,
                "Total_Buy_Hold_Return": strategy_result.total_buy_hold_return,
                "Annualized_Strategy_Volatility": strategy_result.annualized_strategy_volatility,
                "Annualized_Buy_Hold_Volatility": strategy_result.annualized_buy_hold_volatility,
            }
        ]
    ).to_csv(dirs["reports"] / f"strategy_performance_{strategy_ticker}.csv", index=False)

    create_visualizations(featured_data, corr_df, strategy_ticker, dirs["figures"])
    create_strategy_plot(strategy_df, strategy_ticker, dirs["figures"])

    print("\nBasic analytics per stock:")
    print(summary_df.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nCorrelation matrix of daily returns:")
    print(corr_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nStrategy results for {strategy_ticker}:")
    print(f"Total Strategy Return: {strategy_result.total_strategy_return:.4%}")
    print(f"Total Buy-and-Hold Return: {strategy_result.total_buy_hold_return:.4%}")
    print(f"Annualized Strategy Volatility: {strategy_result.annualized_strategy_volatility:.4%}")
    print(f"Annualized Buy-and-Hold Volatility: {strategy_result.annualized_buy_hold_volatility:.4%}")

    print("\nArtifacts generated:")
    print(f"- Raw data: {dirs['raw']}")
    print(f"- Processed data: {dirs['processed']}")
    print(f"- Reports: {dirs['reports']}")
    print(f"- Figures: {dirs['figures']}")


if __name__ == "__main__":
    main()
