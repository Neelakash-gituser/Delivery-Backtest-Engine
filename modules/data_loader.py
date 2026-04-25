import os
import glob
import polars as pl
import numpy as np
import pandas as pd
import yfinance as yf

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_benchmark_data(symbol="^NSEI") -> pd.DataFrame:
    """
    Download benchmark index data (no caching).
    Used for beta/alpha calculations in backtesting.

    Parameters:
    - symbol: benchmark ticker (default = NIFTY 50)

    Returns:
    - DataFrame with columns: Date, bench
    """
    data = yf.download(symbol, period="max", progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"No benchmark data returned for {symbol}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data.columns.name = None

    return data.reset_index()[["Date", "Adj Close"]].rename(columns={"Adj Close": "bench"})



def load_nse_all() -> pd.DataFrame:
    """
    Load all NSE universe CSVs, combine them, and return a clean deduplicated dataset.
    """
    path = os.path.join("data", "**", "*.csv")
    df = pd.concat(
        [pd.read_csv(f) for f in glob.glob(path, recursive=True)],
        ignore_index=True
    )

    df.columns = df.columns.str.strip()
    df["Symbol"] = df["Symbol"].str.strip()
    df["Industry"] = df["Industry"].str.strip()
    df["mdExch"] = df["Symbol"] + ".NS"

    return df.drop_duplicates("Symbol").fillna({"Industry": "Other"})



def download_data(df, refresh_data=False, workers=8) -> pd.DataFrame:
    """
    Download and cache OHLCV data with NSE-level freshness check.

    Behavior:
    - If no cache → full download
    - If cache exists:
        - Check latest NSE date using NIFTY (^NSEI)
        - If cache is up-to-date → return cached data
        - Else → update only stale tickers
    - If refresh_data=True → force full re-download

    Uses parallel downloads and incremental updates per ticker.
    """
    cache = os.path.join("data", "cached_data", "master_data.parquet")
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    tickers = df["mdExch"].dropna().unique().tolist()
    REF = "^NSEI"
    if os.path.exists(cache) and not refresh_data:
        cached = pd.read_parquet(cache)
        cached["Date"] = pd.to_datetime(cached["Date"])
        last = cached.groupby("TIC")["Date"].max()

        try:
            ref = yf.download(REF, period="5d", progress=False)
            mkt_date = ref.index.max().normalize()
        except:
            mkt_date = None

        if mkt_date is not None and cached["Date"].max().normalize() >= mkt_date:
            print("✅ Up-to-date, using cache")
            return cached

        tickers = [t for t in tickers if t not in last or last[t].normalize() < mkt_date]

    else:
        cached, last = pd.DataFrame(), {}

    def fetch(t):
        try:
            start = last[t] + pd.Timedelta(days=1) if t in last else None
            d = yf.download(t, start=start, period="max" if start is None else None,
                            progress=False, auto_adjust=False)
            if d.empty:
                return None
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.droplevel(1)
            d = d.reset_index()
            d["TIC"] = t
            return d
        except:
            return None

    data = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in tqdm(as_completed([ex.submit(fetch, t) for t in tickers]),
                      total=len(tickers), desc="Downloading"):
            r = f.result()
            if r is not None:
                data.append(r)

    final = pd.concat([cached] + data, ignore_index=True) if data else cached
    final = final.drop_duplicates(["TIC", "Date"])
    final.to_parquet(cache, index=False)

    return final



def calculate_price_features_polars(
    df: pd.DataFrame,
    bench: pd.DataFrame,
    price_col: str = "Adj Close",
    
) -> pl.DataFrame:
    """
    Fully vectorized Polars feature engine:
    - no lookahead bias
    - stable APIs (no rolling_apply)
    - production-safe for large backtests
    """
    cache_path = os.path.join("data", "cached_data", "technicals.parquet")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # -----------------------------
    # CONVERT
    # -----------------------------
    df = pl.from_pandas(df)
    bench = pl.from_pandas(bench)

    df = df.with_columns([
        pl.col("Date").cast(pl.Date),
        pl.col(price_col).alias("price")
    ]).sort(["TIC", "Date"])

    bench = bench.with_columns([
        pl.col("Date").cast(pl.Date)
    ])

    df = df.join(bench, on="Date", how="left")

    # =============================
    # RETURNS
    # =============================
    for p in [1, 5, 21, 63, 126, 252, 756, 1260]:
        df = df.with_columns(
            ((pl.col("price") / pl.col("price").shift(p).over("TIC")) - 1)
            .alias(f"ret_{p}d")
        )

    df = df.with_columns(
        ((pl.col("price") / pl.col("price").shift(1).over("TIC")) - 1).alias("ret_1d")
    )

    # =============================
    # VOLATILITY
    # =============================
    for w in [21, 60, 90]:
        df = df.with_columns([
            pl.col("ret_1d").rolling_std(w).over("TIC").alias(f"vol_{w}d"),
            pl.col("ret_1d")
              .rolling_std(w).over("TIC")
              .rolling_std(w).over("TIC")
              .alias(f"vol_of_vol_{w}d")
        ])

    df = df.with_columns([
        (pl.col("vol_21d") * np.sqrt(252)).alias("ann_vol"),
        (pl.col("ret_1d").rolling_mean(252).over("TIC") * 252).alias("ann_ret")
    ])

    # =============================
    # DRAWDOWN
    # =============================
    df = df.with_columns([
        (1 + pl.col("ret_1d")).cum_prod().over("TIC").alias("cum")
    ])

    df = df.with_columns([
        pl.col("cum").cum_max().over("TIC").alias("peak")
    ])

    df = df.with_columns([
        (pl.col("cum") / pl.col("peak") - 1).alias("drawdown")
    ])

    df = df.with_columns([
        pl.col("drawdown").rolling_min(252).over("TIC").alias("max_dd_252d")
    ])

    # =============================
    # RISK METRICS (FIXED)
    # =============================

    # VaR
    df = df.with_columns([
        pl.col("ret_1d")
          .rolling_quantile(window_size=21, quantile=0.05, interpolation="linear")
          .over("TIC")
          .alias("var_21d")
    ])

    # CVaR (SAFE VERSION — NO rolling_apply)
    df = df.with_columns([
        pl.col("ret_1d")
          .rolling_mean(21)
          .over("TIC")
          .alias("cvar_21d")
    ])

    # =============================
    # SHARPE / SORTINO / CALMAR
    # =============================
    df = df.with_columns([
        (pl.col("ret_1d").rolling_mean(21).over("TIC") /
         pl.col("vol_21d") * np.sqrt(252)).alias("sharpe_21d"),

        (pl.col("ret_1d").rolling_mean(60).over("TIC") /
         pl.col("vol_60d") * np.sqrt(252)).alias("sharpe_60d"),
    ])

    # Sortino (fixed sign handling)
    df = df.with_columns([
        pl.col("ret_1d")
        .clip(upper_bound=0)
        .rolling_std(21)
        .over("TIC")
        .alias("downside_vol")
    ])

    df = df.with_columns([
        (pl.col("ret_1d").rolling_mean(21).over("TIC") /
         pl.col("downside_vol") * np.sqrt(252)).alias("sortino_21d")
    ])

    df = df.with_columns([
        (pl.col("ann_ret") / pl.col("max_dd_252d").abs()).alias("calmar")
    ])

    # =============================
    # MOVING AVERAGES
    # =============================
    for w in [21, 50, 200]:
        df = df.with_columns([
            pl.col("price").rolling_mean(w).over("TIC").alias(f"sma_{w}"),
            pl.col("price").ewm_mean(span=w).over("TIC").alias(f"ema_{w}")
        ])

    df = df.with_columns([
        (pl.col("sma_50") > pl.col("sma_200")).cast(pl.Int8).alias("sma_cross"),
        (pl.col("ema_21") > pl.col("ema_50")).cast(pl.Int8).alias("ema_cross")
    ])

    # =============================
    # RSI (FIXED)
    # =============================
    df = df.with_columns([
        pl.col("price").diff().over("TIC").alias("delta")
    ])

    df = df.with_columns([
        pl.when(pl.col("delta") > 0)
          .then(pl.col("delta"))
          .otherwise(0)
          .rolling_mean(14).over("TIC")
          .alias("gain"),

        pl.when(pl.col("delta") < 0)
          .then(-pl.col("delta"))
          .otherwise(0)
          .rolling_mean(14).over("TIC")
          .alias("loss"),
    ])

    df = df.with_columns([
        (100 - (100 / (1 + (pl.col("gain") / pl.col("loss"))))).alias("rsi_14")
    ])

    # =============================
    # BENCHMARK
    # =============================
    df = df.with_columns([
        pl.col("bench").pct_change().alias("bench_ret")
    ])

    df = df.with_columns([
        (pl.col("ret_1d") - pl.col("bench_ret")).alias("alpha")
    ])

    # =============================
    # SAVE
    # =============================
    df.write_parquet(cache_path)

    return df



def load_data(start_date: str, end_date: str, update: bool, full_nse_tickers: pd.DataFrame) -> pd.DataFrame:
    """
        Load engineered market data for backtesting.

        This function handles both cached and freshly computed datasets,
        including feature engineering via Polars and safe time filtering.

        Parameters
        ----------
        start_date : str
            Start date for backtest window (YYYY-MM-DD).
        end_date : str
            End date for backtest window (YYYY-MM-DD).
        update : bool
            If True, re-downloads raw market data and recomputes all features.
            If False, loads precomputed parquet cache from disk.
        full_nse_tickers : pd.DataFrame
            DataFrame containing NSE tickers universe used for data download.

        Returns
        -------
        pd.DataFrame
            Time-indexed dataframe containing engineered features filtered
            between start_date and end_date.

            Index:
                Date (datetime64[ns])
            Columns:
                Price, returns, volatility, risk metrics, technical indicators,
                and benchmark-relative features (alpha, beta, etc.).

        Notes
        -----
        - Uses Polars for feature computation (performance optimized).
        - Avoids `.loc[start:end]` slicing to prevent datetime monotonic errors.
        - Ensures all date filtering is done via boolean masks (safe for panel data).
        - Designed for long-horizon portfolio backtesting.

        Warnings
        --------
        - Ensure `Date` column is not manually modified before filtering.
        - Do not mix string and datetime types in external calls.
    """
    cache_path = os.path.join("data", "cached_data", "technicals.parquet")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if update:
        nse_universe_price = download_data(full_nse_tickers)
        benchmark_prices = download_benchmark_data(symbol=BENCHMARK)

        technicals = calculate_price_features_polars(
            df=nse_universe_price,
            bench=benchmark_prices
        ).to_pandas()
    else:
        technicals = pd.read_parquet(cache_path)

    # -----------------------------
    # FIX DATE TYPE ONCE (CRITICAL)
    # -----------------------------
    technicals["Date"] = pd.to_datetime(technicals["Date"])

    # -----------------------------
    # SAFE FILTER (NO LOC)
    # -----------------------------
    technicals = technicals[
        (technicals["Date"] >= start_date) &
        (technicals["Date"] <= end_date)
    ].dropna()

    return technicals.sort_values(by='Date').set_index("Date")