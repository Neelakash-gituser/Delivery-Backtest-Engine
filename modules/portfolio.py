from pypfopt import expected_returns, risk_models, EfficientFrontier
from modules.utils import build_window
import pandas as pd

def estimate_moments(price_window: pd.DataFrame):
    """
    Estimate expected returns + covariance matrix.
    """
    price_matrix = price_window.pivot(
        index="Date",
        columns="TIC",
        values="Adj Close"
    ).ffill().dropna(axis=1, how="any")

    mu = expected_returns.mean_historical_return(price_matrix, frequency=252)
    cov = risk_models.CovarianceShrinkage(price_matrix).ledoit_wolf()

    return mu, cov


def optimize_portfolio(mu, cov, method="max_sharpe"):
    """
    Portfolio optimizer switch.
    """
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    if method == "max_sharpe":
        ef.max_sharpe()
    elif method == "min_vol":
        ef.min_volatility()
    else:
        raise ValueError("Unsupported optimizer")

    return ef.clean_weights()


def allocate_portfolio(
    ranked,
    prices,
    date,
    method="max_sharpe",
    lookback=252,
    custom_ranker=None
):
    """
    Unified portfolio allocator:

    Supports:
    - max_sharpe / min_vol (PyPortfolioOpt)
    - equal_weight
    - custom ranking-based allocation
    """

    # -------------------------
    # 1. NON-OPTIMIZER PATHS
    # -------------------------
    if method == "equal_weight":
        tickers = ranked["TIC"].unique().tolist()
        w = {t: 1 / len(tickers) for t in tickers}
        return w

    if method == "custom":
        if custom_ranker is None:
            raise ValueError("custom_ranker must be provided")

        ranked_out = custom_ranker(ranked)
        scores = ranked_out.set_index("TIC")["score"]
        scores = scores / scores.sum()

        return scores.to_dict()

    # -------------------------
    # 2. BUILD WINDOW (for MVO/MinVol)
    # -------------------------
    window = build_window(prices, date, lookback)
    tickers = ranked["TIC"].unique().tolist()
    window = window[window["TIC"].isin(tickers)]

    # -------------------------
    # 3. ESTIMATION
    # -------------------------
    mu, cov = estimate_moments(window)

    # -------------------------
    # 4. OPTIMIZATION
    # -------------------------
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))

    if method == "max_sharpe":
        ef.max_sharpe()
    elif method == "min_vol":
        ef.min_volatility()
    else:
        raise ValueError("Unknown allocation method")

    return ef.clean_weights()