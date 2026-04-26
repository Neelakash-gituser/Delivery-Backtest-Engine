# modules/slippage.py
import numpy as np

class SlippageModel:
    """
    mode = "fixed"   → flat bps on every trade
    mode = "volume"  → square-root market impact scaled by ADV
    mode = "none"    → 0 slippage (current behaviour)
    """

    def __init__(self, config: dict):
        slip_cfg = config.get("slippage", {})
        self.mode   = slip_cfg.get("mode", "fixed")       # fixed | volume | none
        self.bps    = slip_cfg.get("bps", 10) / 10_000    # default 10 bps
        self.impact_k = slip_cfg.get("impact_k", 0.1)     # market impact constant

    def get_factor(
        self,
        ticker: str,
        direction: int,          # +1 = buy, -1 = sell
        trade_value: float,      # absolute ₹ value of the delta trade
        adv: float = None,       # avg daily rupee volume for this ticker
        daily_vol: float = None, # daily return std dev for this ticker
    ) -> float:
        """
        Returns a slippage factor to multiply against price.
        exec_price = signal_price * (1 + direction * factor)
        """
        if self.mode == "none":
            return 0.0

        if self.mode == "fixed":
            return self.bps

        if self.mode == "volume":
            if adv is None or adv == 0:
                return self.bps   # fallback to fixed if no ADV
            sigma = daily_vol if daily_vol else 0.015  # fallback 1.5% daily vol
            participation = trade_value / adv
            return sigma * self.impact_k * np.sqrt(participation)

        return 0.0