📊 Portfolio Rebalancing Backtester

A Python-based portfolio backtesting engine that simulates realistic rule-based portfolio rebalancing across multiple time frequencies (daily, weekly, monthly, quarterly). The system supports position-level tracking, transaction cost modeling, and cash-aware trade execution.

🔧 Key Features
Delta-based rebalancing logic (adjusts only required position changes, not full rebuilds)
Supports integer share execution for realistic trading simulation
Handles transaction costs and cash constraints
Maintains full portfolio state tracking over time
Flexible rebalance frequency support (daily, weekly, monthly, quarterly)
Includes trade blotter logging for audit and analysis
Market data-driven backtesting using adjusted close prices
📈 Simulation Design

The engine computes portfolio value at each step and rebalances holdings by moving from current positions to target weights derived from a ranking or strategy signal. It avoids unrealistic assumptions like fractional shares or perfect liquidity and ensures cash is updated strictly through executed trades.

🧠 Use Cases
Quant strategy backtesting
Portfolio optimization research
Turnover and transaction cost analysis
Factor-based investing simulation
Strategy performance attribution
⚙️ Core Idea

Instead of rebuilding the portfolio at each rebalance, the system performs incremental rebalancing using position deltas, closely mimicking real-world execution constraints and improving realism of backtest results.