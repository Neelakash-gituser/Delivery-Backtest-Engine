import pandas as pd

def eval_rule(df: pd.DataFrame, rule: dict):
    """
    Evaluate a structured screening rule on a financial dataset.

    This function applies a rule-based filtering system where each rule
    consists of multiple condition groups. Within each group, conditions
    can be combined using AND / OR logic, and groups are combined using
    AND logic by default.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing financial features (e.g., returns,
        volatility, ratios, indicators).
    rule : dict
        Nested dictionary defining screening logic.

        Example structure:
        {
            "condition1": {
                "and": [
                    ("sharpe_60d", ">", 1),
                    ("ann_ret", ">", 0.12)
                ],
                "or": [
                    ("sharpe_60d", "<=", "sharpe_21d")
                ]
            },
            "condition2": {
                "and": [
                    ("drawdown", ">", -0.2)
                ]
            }
        }

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing rows that satisfy the rule logic.

    Notes
    -----
    - Conditions inside each group are combined using AND / OR logic.
    - Groups are combined using AND logic by default.
    - Designed for backtesting and cross-sectional stock screening.
    - No time-awareness is enforced here; must be handled upstream.
    """
    group_masks = []

    for _, group in rule.items():

        group_mask = None

        # AND logic inside group
        if "and" in group:
            for col, op, val in group["and"]:
                mask = build_mask(df, col, op, val)
                group_mask = mask if group_mask is None else (group_mask & mask)

        # OR logic inside group
        if "or" in group:
            or_mask = None
            for col, op, val in group["or"]:
                mask = build_mask(df, col, op, val)
                or_mask = mask if or_mask is None else (or_mask | mask)

            group_mask = or_mask if group_mask is None else (group_mask | or_mask)

        group_masks.append(group_mask)

    # combine groups (AND by default)
    final_mask = group_masks[0]
    for m in group_masks[1:]:
        final_mask &= m

    return df[final_mask]


def build_mask(df, col, op, val):
    """
        Build a boolean mask for a single condition.

        Supports:
        - Column vs value comparisons (e.g., sharpe_60d > 1)
        - Column vs column comparisons (e.g., sharpe_60d <= sharpe_21d)

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        col : str
            Column name on left-hand side of condition.
        op : str
            Comparison operator: one of [>, <, >=, <=, ==, !=].
        val : str | float | int
            Right-hand side value or column name.

        Returns
        -------
        pd.Series
            Boolean mask representing the condition.
    """
    if isinstance(val, str) and val in df.columns:
        return eval(f"df[col] {op} df[val]")
    else:
        return eval(f"df[col] {op} val")