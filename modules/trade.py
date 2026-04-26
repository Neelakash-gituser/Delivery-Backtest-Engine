import pandas as pd
import numpy as np
from modules.slippage import SlippageModel


def indian_cost_model(trade_value):
    """
    Approx NSE equity costs (simplified)
    """
    stt          = 0.001       # ~0.1%
    brokerage    = 0.0003
    exchange_tx  = 0.0000345
    sebi_fee     = 0.000001

    total = stt + brokerage + exchange_tx + sebi_fee
    return trade_value * total


def execute_trades(
    portfolio,
    weights,
    prices_today,
    cash,
    cost_model,
    date,
    trade_blotter,
    slippage_model=None,   # SlippageModel instance — pass None to disable
    open_prices=None,      # pd.Series indexed by TIC for next day's open prices
    exec_mode="close",     # "close" | "open_next" | "random_open"
    noise_sigma=0.003,     # gaussian noise std — only used when exec_mode="random_open"
):
    # ── EXECUTION PRICE RESOLVER ──────────────────────────────────────────────
    def get_exec_price(tic, signal_price, direction, approx_trade_value):
        """
        Resolves base execution price then layers slippage on top.
        direction : +1 = buy, -1 = sell
        """
        # Step 1: base price from execution mode
        if exec_mode == "close" or open_prices is None:
            base = signal_price

        elif exec_mode == "open_next":
            base = open_prices.get(tic, np.nan)
            if pd.isna(base) or base <= 0:
                base = signal_price       # fallback: last-day or missing open

        elif exec_mode == "random_open":
            base = open_prices.get(tic, np.nan)
            if pd.isna(base) or base <= 0:
                base = signal_price
            base = base * (1 + np.random.normal(0, noise_sigma))

        else:
            base = signal_price

        # Step 2: apply slippage on top of base price
        if slippage_model is not None:
            row = prices_today.loc[tic] if tic in prices_today.index else None

            # ADV and DailyVol are pre-computed columns in prices_today
            # (added in data_loader). Falls back to None gracefully.
            adv       = row["ADV"]      if row is not None and "ADV"      in row.index else None
            daily_vol = row["DailyVol"] if row is not None and "DailyVol" in row.index else None

            factor = slippage_model.get_factor(
                ticker      = tic,
                direction   = direction,
                trade_value = approx_trade_value,
                adv         = adv,
                daily_vol   = daily_vol,
            )
            base = base * (1 + direction * factor)

        return base

    # ─────────────────────────────────────────────────────────────────────────

    new_portfolio = portfolio.copy()
    total_cost    = 0

    # -----------------------------
    # 1. PORTFOLIO VALUE
    # -----------------------------
    portfolio_value = cash + sum(
        qty * prices_today.loc[t, "Adj Close"]
        for t, qty in portfolio.items()
        if t in prices_today.index
    )

    # -----------------------------
    # 2. TARGET POSITIONS
    # -----------------------------
    target_positions = {}

    for tic, w in weights.items():
        if tic not in prices_today.index or w <= 0:
            continue
        price                  = prices_today.loc[tic, "Adj Close"]
        target_value           = portfolio_value * w
        target_positions[tic]  = target_value / price   # float shares → int later

    # -----------------------------
    # 3. SELL FIRST (DELTA ONLY)
    # -----------------------------
    for tic, current_qty in portfolio.items():

        if tic not in prices_today.index:
            continue

        signal_price = prices_today.loc[tic, "Adj Close"]
        target_qty   = target_positions.get(tic, 0)
        qty_diff     = current_qty - target_qty    # positive → need to sell

        if qty_diff <= 0:
            continue

        sell_qty = int(qty_diff)

        if sell_qty <= 0:
            continue

        approx_value = sell_qty * signal_price
        exec_price   = get_exec_price(tic, signal_price, direction=-1,
                                      approx_trade_value=approx_value)

        trade_value  = sell_qty * exec_price
        cost         = cost_model(trade_value)

        cash        += (trade_value - cost)
        total_cost  += cost

        new_qty = current_qty - sell_qty
        if new_qty > 0:
            new_portfolio[tic] = new_qty
        else:
            new_portfolio.pop(tic, None)

        trade_blotter.append({
            "Date":        date,
            "TIC":         tic,
            "SignalPrice": signal_price,
            "Price":       exec_price,
            "Slippage":    round(exec_price - signal_price, 6),
            "Qty":         -sell_qty,
            "TradeValue":  -trade_value,
            "Cost":        cost,
            "Type":        "SELL",
        })

    # -----------------------------
    # 4. BUY AFTER SELL (DELTA ONLY)
    # -----------------------------
    for tic, target_qty in target_positions.items():

        if tic not in prices_today.index:
            continue

        signal_price = prices_today.loc[tic, "Adj Close"]
        current_qty  = new_portfolio.get(tic, 0)
        qty_diff     = target_qty - current_qty    # positive → need to buy

        if qty_diff <= 0:
            continue

        buy_qty = int(qty_diff)

        if buy_qty <= 0:
            continue

        approx_value = buy_qty * signal_price
        exec_price   = get_exec_price(tic, signal_price, direction=+1,
                                      approx_trade_value=approx_value)

        trade_value  = buy_qty * exec_price
        cost         = cost_model(trade_value)

        # Cash constraint BEFORE execution
        if trade_value + cost > cash:
            max_qty = int(cash / (exec_price * (1 + cost_model(1))))
            buy_qty = min(buy_qty, max_qty)

            if buy_qty <= 0:
                continue

            trade_value = buy_qty * exec_price
            cost        = cost_model(trade_value)

        cash       -= (trade_value + cost)
        total_cost += cost

        new_portfolio[tic] = current_qty + buy_qty

        trade_blotter.append({
            "Date":        date,
            "TIC":         tic,
            "SignalPrice": signal_price,
            "Price":       exec_price,
            "Slippage":    round(exec_price - signal_price, 6),
            "Qty":         buy_qty,
            "TradeValue":  trade_value,
            "Cost":        cost,
            "Type":        "BUY",
        })

    return new_portfolio, cash, trade_blotter