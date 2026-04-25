import pandas as pd
import numpy as np
from modules.screen import eval_rule
from modules.portfolio import allocate_portfolio
from modules.trade import execute_trades
from modules.utils import generate_rebalance_dates, default_rank, ensure_date_column
from modules.trade import indian_cost_model

def run_backtest(
    price_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_cash: float,
    rebalance_freq: str,
    screen_rule: dict,
    allocator: str = "max_sharpe",
    ranker=None,
    cost_model=indian_cost_model,
    ranker_dict=None,
    verbose: bool = True   # 👈 ADD THIS
):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    price_df = ensure_date_column(price_df)
    feature_df = ensure_date_column(feature_df)

    price_df = price_df[(price_df["Date"] >= start_date) & (price_df["Date"] <= end_date)]
    feature_df = feature_df[(feature_df["Date"] >= start_date) & (feature_df["Date"] <= end_date)]

    cash = initial_cash
    portfolio = {}
    trade_blotter = []
    equity_curve = []

    all_dates = sorted(price_df["Date"].unique())
    rebalance_dates = generate_rebalance_dates(all_dates, rebalance_freq)

    # =============================
    # MAIN LOOP
    # =============================
    for i, date in enumerate(all_dates):

        if verbose and i % 10 == 0:
            print(f"\n📅 Progress: {i}/{len(all_dates)} | Date: {date.date()}")

        prices_today = price_df[price_df["Date"] == date].set_index("TIC")
        # -----------------------------
        # 1. MTM
        # -----------------------------
        port_value = cash
        for tic, qty in portfolio.items():
            if tic in prices_today.index:
                port_value += (qty * prices_today.loc[tic, "Adj Close"])

        # -----------------------------
        # 2. REBALANCE
        # -----------------------------
        if date in rebalance_dates:

            if verbose:
                print(f"\n🔁 Rebalance triggered on {date.date()}")
                print(f"   💰 Portfolio value: {port_value:,.2f}")
                print(f"   🧾 Cash: {cash:,.2f}")

            features_today = feature_df[feature_df["Date"] == date]
            screened = eval_rule(features_today, screen_rule)
            if verbose:
                print(f"   🎯 Screened stocks: {len(screened)}")

            candidates = screened["TIC"].unique().tolist()
            if len(candidates) > 0:
                # -----------------------------
                # RANKING
                # -----------------------------
                if ranker:
                    ranked = ranker(features_today[features_today["TIC"].isin(candidates)])
                else:
                    ranked = default_rank(features_today, candidates, ranker_dict)
                if verbose:
                    print(f"   📊 Ranked stocks: {len(ranked)}")
                # -----------------------------
                # ALLOCATION
                # -----------------------------
                weights = allocate_portfolio(
                    ranked,
                    prices=price_df,
                    date=date,
                    method=allocator
                )

                if verbose:
                    top_weights = sorted(weights.items(), key=lambda x: -x[1])[:5]
                    print(f"   🧠 Top allocations: {top_weights}")
                # -----------------------------
                # EXECUTION
                # -----------------------------
                portfolio, cash, trade_blotter = execute_trades(
                    portfolio,
                    weights,
                    prices_today,
                    cash,
                    cost_model,
                    date,
                    trade_blotter=trade_blotter
                )

                if verbose:
                    print(f"   💼 Trades executed | New cash: {cash:,.2f}")

        # -----------------------------
        # 3. STORE EQUITY
        # -----------------------------
        equity_curve.append({
            "Date": date,
            "equity": port_value,
            "cash": cash
        })

    if verbose:
        print("\n✅ Backtest completed")

    return pd.DataFrame(equity_curve), pd.DataFrame(trade_blotter)