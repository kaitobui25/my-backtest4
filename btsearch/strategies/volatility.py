from __future__ import annotations

from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .common import apply_direction, crossed_above, crossed_below


def atr_expansion_breakout(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    atr = cache.atr(int(p["atr_window"]))
    atr_mean = atr.rolling(
        int(p["atr_mean_window"]),
        min_periods=int(p["atr_mean_window"]),
    ).mean()
    expansion = atr > atr_mean * float(p["expansion_mult"])
    upper = cache.rolling_high(int(p["lookback"]))
    lower = cache.rolling_low(int(p["lookback"]))
    long_signal = crossed_above(cache.close, upper) & expansion
    short_signal = crossed_below(cache.close, lower) & expansion
    return apply_direction(long_signal, short_signal, cfg)


def squeeze_breakout(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    bandwidth = cache.bandwidth(int(p["bb_window"]), float(p["std_mult"]))
    lookback = int(p["bandwidth_lookback"])
    threshold = bandwidth.rolling(
        lookback, min_periods=lookback
    ).quantile(float(p["quantile"]))
    squeeze_recent = (bandwidth.shift(1) <= threshold.shift(1)).rolling(
        int(p["squeeze_memory"]),
        min_periods=1,
    ).max().astype(bool)
    upper = cache.rolling_high(int(p["breakout_lookback"]))
    lower = cache.rolling_low(int(p["breakout_lookback"]))
    long_signal = crossed_above(cache.close, upper) & squeeze_recent
    short_signal = crossed_below(cache.close, lower) & squeeze_recent
    return apply_direction(long_signal, short_signal, cfg)
