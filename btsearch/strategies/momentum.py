from __future__ import annotations

from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .common import apply_direction, crossed_above, crossed_below, regime_filter


def rsi_momentum(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    value = cache.rsi(int(p["rsi_window"]))
    threshold = float(p["threshold"])
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(value, threshold) & long_regime
    short_signal = crossed_below(value, 100.0 - threshold) & short_regime
    return apply_direction(long_signal, short_signal, cfg)


def roc_momentum(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    value = cache.roc(int(p["roc_window"]))
    threshold = float(p["threshold"])
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(value, threshold) & long_regime
    short_signal = crossed_below(value, -threshold) & short_regime
    return apply_direction(long_signal, short_signal, cfg)


def macd_momentum(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    macd_line, signal_line, histogram = cache.macd(
        int(p["fast"]),
        int(p["slow"]),
        int(p["signal"]),
    )
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(macd_line, signal_line) & (histogram > 0) & long_regime
    short_signal = crossed_below(macd_line, signal_line) & (histogram < 0) & short_regime
    return apply_direction(long_signal, short_signal, cfg)
