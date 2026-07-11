from __future__ import annotations

from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .common import apply_direction, crossed_above, crossed_below


def breakout_volume(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    volume_ok = cache.df["volume"] > (
        cache.volume_mean(int(p["volume_window"]))
        * float(p["volume_mult"])
    )
    upper = cache.rolling_high(int(p["lookback"]))
    lower = cache.rolling_low(int(p["lookback"]))
    long_signal = crossed_above(cache.close, upper) & volume_ok
    short_signal = crossed_below(cache.close, lower) & volume_ok
    return apply_direction(long_signal, short_signal, cfg)


def momentum_volume(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    value = cache.roc(int(p["roc_window"]))
    threshold = float(p["threshold"])
    volume_ok = cache.df["volume"] > (
        cache.volume_mean(int(p["volume_window"]))
        * float(p["volume_mult"])
    )
    long_signal = crossed_above(value, threshold) & volume_ok
    short_signal = crossed_below(value, -threshold) & volume_ok
    return apply_direction(long_signal, short_signal, cfg)
