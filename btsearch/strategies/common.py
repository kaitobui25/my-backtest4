from __future__ import annotations

import pandas as pd
from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache


def apply_direction(
    long_signal: pd.Series,
    short_signal: pd.Series,
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series]:
    if config.direction == "long":
        short_signal = pd.Series(False, index=long_signal.index)
    elif config.direction == "short":
        long_signal = pd.Series(False, index=short_signal.index)
    elif config.direction != "both":
        raise ValueError(f"Direction không hợp lệ: {config.direction}")

    conflict = long_signal & short_signal
    return (
        long_signal.mask(conflict, False).fillna(False),
        short_signal.mask(conflict, False).fillna(False),
    )


def regime_filter(
    cache: IndicatorCache,
    ema_window: int,
) -> tuple[pd.Series, pd.Series]:
    if ema_window <= 0:
        true = pd.Series(True, index=cache.df.index)
        return true, true
    trend = cache.ema(ema_window)
    return cache.close > trend, cache.close < trend


def crossed_above(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, pd.Series):
        return (a > b) & (a.shift(1) <= b.shift(1))
    return (a > b) & (a.shift(1) <= b)


def crossed_below(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if isinstance(b, pd.Series):
        return (a < b) & (a.shift(1) >= b.shift(1))
    return (a < b) & (a.shift(1) >= b)


def apply_time_filter(
    long_signal: pd.Series,
    short_signal: pd.Series,
    config: StrategyConfig,
) -> tuple[pd.Series, pd.Series]:
    session = str(config.params.get("session", "all"))
    weekday_filter = str(config.params.get("weekday_filter", "all"))
    index = long_signal.index

    if session == "all":
        session_mask = pd.Series(True, index=index)
    else:
        hours = index.hour
        ranges = {
            "asia": (0, 8),
            "london": (7, 16),
            "new_york": (13, 22),
            "overlap": (13, 16),
            "high_liquidity": (7, 22),
        }
        if session not in ranges:
            raise ValueError(f"Session không hợp lệ: {session}")
        start, end = ranges[session]
        session_mask = pd.Series(
            (hours >= start) & (hours < end),
            index=index,
        )

    weekdays = index.dayofweek
    if weekday_filter == "all":
        weekday_mask = pd.Series(True, index=index)
    elif weekday_filter == "weekdays":
        weekday_mask = pd.Series(weekdays < 5, index=index)
    elif weekday_filter == "mon_thu":
        weekday_mask = pd.Series(weekdays < 4, index=index)
    else:
        raise ValueError(
            f"weekday_filter không hợp lệ: {weekday_filter}"
        )

    mask = session_mask & weekday_mask
    return long_signal & mask, short_signal & mask
