from __future__ import annotations

import numpy as np


def calculate_net_r(
    pnl: np.ndarray,
    size: np.ndarray,
    entry_price: np.ndarray,
    initial_risk_pct: np.ndarray,
) -> np.ndarray:
    initial_risk_cash = size * entry_price * initial_risk_pct
    return np.divide(
        pnl,
        initial_risk_cash,
        out=np.full_like(pnl, np.nan, dtype=float),
        where=(initial_risk_cash > 0) & np.isfinite(initial_risk_cash),
    )


def summarize_r(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "trades": 0,
            "winrate": np.nan,
            "avg_win_r": np.nan,
            "avg_loss_r": np.nan,
            "expectancy_r": np.nan,
            "profit_factor_r": np.nan,
            "total_r": 0.0,
            "max_drawdown_r": np.nan,
        }

    wins = values[values > 0]
    losses = values[values < 0]
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    drawdown = peaks[1:] - equity

    return {
        "trades": int(n),
        "winrate": float((values > 0).mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
        "expectancy_r": float(values.mean()),
        "profit_factor_r": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and losses.sum() != 0
            else np.inf
        ),
        "total_r": float(values.sum()),
        "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
    }


def evidence_label(trades: int) -> str:
    if trades < 30:
        return "insufficient"
    if trades < 100:
        return "very_weak"
    if trades < 300:
        return "exploratory"
    return "meaningful"
