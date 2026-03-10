# Quantitative Developer Assignment - Yahoo Finance Analysis

This project implements a complete end-to-end financial data pipeline using Yahoo Finance data, including:
- data extraction,
- cleaning and preprocessing,
- feature engineering,
- basic quantitative analytics,
- visualization,
- and a moving-average crossover strategy (bonus section).

The implementation is designed to satisfy all requirements from the assignment document.

## Project Structure

- `quant_analysis.py` - main script for extraction, processing, analysis, and plotting
- `requirements.txt` - Python dependencies
- `data/raw/` - downloaded raw data per ticker
- `data/processed/` - cleaned and feature-engineered datasets
- `reports/` - analytics and strategy output CSVs
- `figures/` - generated charts

## Environment Setup

1. Create and activate a Python 3 virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run with default tickers (`AAPL`, `MSFT`, `GOOGL`) and default date range:

```bash
python quant_analysis.py
```

Run with custom tickers and date range:

```bash
python quant_analysis.py --tickers AAPL NVDA TSLA --start-date 2022-01-01 --end-date 2026-01-01 --strategy-ticker NVDA
```

## Assignment Requirements Mapping

### Part 1 - Data Extraction
- Uses `yfinance` to download daily OHLCV data.
- Includes `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
- Saves raw data to `data/raw/<TICKER>_raw.csv`.

### Part 2 - Data Cleaning and Preprocessing
- Converts date column to `datetime` and sets it as index.
- Handles missing values using time interpolation, then forward/backward fill.
- Removes duplicate rows.
- Aligns all stocks to a common date range (intersection of dates).
- Saves cleaned data to `data/processed/<TICKER>_cleaned.csv`.
- Saves cleaning details to `reports/cleaning_report.csv`.

### Part 3 - Feature Engineering
Adds the following columns for each stock:
- `Daily_Return` (`pct_change`)
- `Log_Return` (`log(Close / Close.shift(1))`)
- `MA_20` (20-day moving average)
- `MA_50` (50-day moving average)
- `Volatility_30` (30-day rolling std of daily returns)

Feature datasets are saved to `data/processed/<TICKER>_featured.csv`.

### Part 4 - Basic Analytics
Computes per-stock:
- mean daily return,
- standard deviation of daily returns,
- annualized volatility (`std * sqrt(252)`).

Computes for all stocks:
- correlation matrix of daily returns.

Outputs:
- `reports/basic_analytics_summary.csv`
- `reports/correlation_matrix.csv`

### Part 5 - Visualization
Creates:
1. `figures/price_history_all_stocks.png` - closing prices over time
2. `figures/moving_averages_<TICKER>.png` - close vs MA20/MA50 for one stock
3. `figures/correlation_heatmap.png` - correlation heatmap

### Part 6 - Optional Bonus (Strategy)
Implements moving-average crossover strategy for one selected stock:
- Buy signal when `MA_20 > MA_50`
- Exit when crossover reverses
- Strategy uses previous-day signal to avoid look-ahead bias

Compares:
- strategy cumulative return vs buy-and-hold cumulative return

Outputs:
- `reports/strategy_timeseries_<TICKER>.csv`
- `reports/strategy_performance_<TICKER>.csv`
- `figures/strategy_vs_buy_hold_<TICKER>.png`

## Assumptions

- Data is pulled from Yahoo Finance and depends on market calendar/trading availability.
- At least 3 tickers should be provided.
- Annualization factor is 252 trading days.
- Date alignment uses the strict common period across all selected tickers.

## Notes on Missing Values

Missing values are handled in this order:
1. Time-based interpolation (`interpolate(method="time")`)
2. Forward fill (`ffill`)
3. Backward fill (`bfill`)

This ensures continuity for rolling computations while minimizing data loss.
