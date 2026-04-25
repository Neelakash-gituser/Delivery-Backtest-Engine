import pandas as pd
import numpy as np


def indian_cost_model(trade_value):
    """
    Approx NSE equity costs (simplified)
    """
    stt = 0.001  # ~0.1%
    brokerage = 0.0003
    exchange_tx = 0.0000345
    sebi_fee = 0.000001

    total = stt + brokerage + exchange_tx + sebi_fee

    return trade_value * total


def execute_trades(
    portfolio,
    weights,
    prices_today,
    cash,
    cost_model,
    date,
    trade_blotter
):
    new_portfolio = portfolio.copy()
    total_cost = 0

    # -----------------------------
    # 1. PORTFOLIO VALUE
    # -----------------------------
    portfolio_value = cash + sum(
        qty * prices_today.loc[t, "Adj Close"]
        for t, qty in portfolio.items()
        if t in prices_today.index
    )

    # -----------------------------
    # 2. TARGET POSITIONS (KEEP VALUE-BASED ONLY)
    # -----------------------------
    target_positions = {}

    for tic, w in weights.items():
        if tic not in prices_today.index or w <= 0:
            continue

        price = prices_today.loc[tic, "Adj Close"]
        target_value = portfolio_value * w

        # 🔥 FIX 1: DO NOT use float shares directly
        target_positions[tic] = target_value / price

    # -----------------------------
    # 3. SELL FIRST (DELTA ONLY)
    # -----------------------------
    for tic, current_qty in portfolio.items():

        if tic not in prices_today.index:
            continue

        price = prices_today.loc[tic, "Adj Close"]
        target_qty = target_positions.get(tic, 0)

        # 🔥 FIX 2: REAL DELTA TRADE
        qty_diff = current_qty - target_qty   # positive = sell

        if qty_diff <= 0:
            continue

        sell_qty = int(qty_diff)   # 🔥 integer enforcement

        if sell_qty <= 0:
            continue

        trade_value = sell_qty * price
        cost = cost_model(trade_value)

        cash += (trade_value - cost)
        total_cost += cost

        new_qty = current_qty - sell_qty

        if new_qty > 0:
            new_portfolio[tic] = new_qty
        else:
            new_portfolio.pop(tic, None)

        trade_blotter.append({
            "Date": date,
            "TIC": tic,
            "Price": price,
            "Qty": -sell_qty,
            "TradeValue": -trade_value,
            "Cost": cost,
            "Type": "SELL"
        })

    # -----------------------------
    # 4. BUY AFTER SELL (DELTA ONLY)
    # -----------------------------
    for tic, target_qty in target_positions.items():

        if tic not in prices_today.index:
            continue

        price = prices_today.loc[tic, "Adj Close"]
        current_qty = new_portfolio.get(tic, 0)

        # 🔥 FIX 3: TRUE DELTA
        qty_diff = target_qty - current_qty

        if qty_diff <= 0:
            continue

        buy_qty = int(qty_diff)   # integer enforcement

        if buy_qty <= 0:
            continue

        trade_value = buy_qty * price
        cost = cost_model(trade_value)

        # 🔥 FIX 4: cash constraint BEFORE execution (not after math)
        if trade_value + cost > cash:
            max_qty = int(cash / (price * (1 + cost_model(1))))
            buy_qty = min(buy_qty, max_qty)

            if buy_qty <= 0:
                continue

            trade_value = buy_qty * price
            cost = cost_model(trade_value)

        cash -= (trade_value + cost)
        total_cost += cost

        new_portfolio[tic] = current_qty + buy_qty

        trade_blotter.append({
            "Date": date,
            "TIC": tic,
            "Price": price,
            "Qty": buy_qty,
            "TradeValue": trade_value,
            "Cost": cost,
            "Type": "BUY"
        })

    return new_portfolio, cash, trade_blotter