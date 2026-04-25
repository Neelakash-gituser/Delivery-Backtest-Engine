import yaml
import pandas as pd

def load_config(path: str) -> dict:
    """
    Load backtesting configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    dict
        Configuration dictionary containing parameters like:
        start_date, end_date, benchmark, update flag, etc.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def build_window(prices: pd.DataFrame, end_date, lookback=252):
    """
    Returns clean price window up to end_date.
    """
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.Timedelta(days=lookback * 2)

    window = prices[
        (prices["Date"] <= end_date) &
        (prices["Date"] >= start_date)
    ].copy()

    return window


def generate_rebalance_dates(all_dates, freq: str):
    """
    Generate rebalance dates aligned to actual trading days.

    Parameters
    ----------
    all_dates : array-like
        Unique trading dates from price data
    freq : str
        One of ["daily", "weekly", "monthly", "quarterly"]

    Returns
    -------
    set of pd.Timestamp
        Dates on which rebalancing should occur
    """
    FREQ_MAP = {
            "daily": "D",
            "weekly": "W",
            "monthly": "M",
            "quarterly": "Q"
    }

    if freq not in FREQ_MAP:
        raise ValueError(f"Unsupported freq: {freq}")

    pandas_freq = FREQ_MAP[freq]
    dates = pd.Series(pd.to_datetime(all_dates)).sort_values()

    rebalance_dates = (
        dates.groupby(dates.dt.to_period(pandas_freq))
        .max()
        .values
    )

    return set(pd.to_datetime(rebalance_dates))


def default_rank(df, candidates, ranker_dict=None):
    """
    fallback ranking if no custom function provided
    """
    df = df[df["TIC"].isin(candidates)]
    df["score"] = 0

    if ranker_dict:
        for feature, weight in ranker_dict.items():
            if feature in df.columns:
                df["score"] += df[feature] * weight
        df["rank"] = df["score"].rank(ascending=False)
    else:
        df["score"] = 1

    return df.sort_values("rank").head(10)  # default to top 10


def ensure_date_column(df):
    """
    Ensure DataFrame has a 'Date' column of datetime type.
    If index is DatetimeIndex, reset it to a 'Date' column.
    """
    df = df.copy()

    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        else:
            raise ValueError("No Date column or DatetimeIndex found")

    df["Date"] = pd.to_datetime(df["Date"])
    return df