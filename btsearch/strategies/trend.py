from __future__ import annotations

from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .common import apply_direction, crossed_above, crossed_below, regime_filter


def donchian_breakout(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    upper = cache.rolling_high(int(p["lookback"]))
    lower = cache.rolling_low(int(p["lookback"]))
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(cache.close, upper) & long_regime
    short_signal = crossed_below(cache.close, lower) & short_regime
    return apply_direction(long_signal, short_signal, cfg)


def rolling_range_breakout(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    upper = cache.rolling_high(int(p["lookback"]))
    lower = cache.rolling_low(int(p["lookback"]))
    atr = cache.atr(int(p["atr_window"]))
    buffer = float(p["breakout_atr_buffer"]) * atr
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(cache.close, upper + buffer) & long_regime
    short_signal = crossed_below(cache.close, lower - buffer) & short_regime
    return apply_direction(long_signal, short_signal, cfg)


def ema_cross(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    fast = cache.ema(int(p["fast_ema"]))
    slow = cache.ema(int(p["slow_ema"]))
    long_signal = crossed_above(fast, slow)
    short_signal = crossed_below(fast, slow)
    return apply_direction(long_signal, short_signal, cfg)


def ema_slope_pullback(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    trend = cache.ema(int(p["ema_window"]))
    slope = trend - trend.shift(int(p["slope_lookback"]))
    atr = cache.atr(int(p["atr_window"]))
    distance = (cache.close - trend) / atr.replace(0.0, float("nan"))
    max_distance = float(p["pullback_atr"])
    long_signal = (
        (cache.close > trend)
        & (slope > 0)
        & (distance >= 0)
        & (distance <= max_distance)
        & (distance.shift(1) > max_distance)
        & (cache.close > cache.close.shift(1))
    )
    short_signal = (
        (cache.close < trend)
        & (slope < 0)
        & (distance <= 0)
        & (distance >= -max_distance)
        & (distance.shift(1) < -max_distance)
        & (cache.close < cache.close.shift(1))
    )
    return apply_direction(long_signal, short_signal, cfg)
