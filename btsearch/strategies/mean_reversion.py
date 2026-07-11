from __future__ import annotations

from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .common import apply_direction, crossed_above, crossed_below, regime_filter


def bollinger_reentry(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    lower, _, upper = cache.bollinger(
        int(p["window"]), float(p["std_mult"])
    )
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(cache.close, lower) & (cache.close.shift(1) < lower.shift(1))
    short_signal = crossed_below(cache.close, upper) & (cache.close.shift(1) > upper.shift(1))
    return apply_direction(
        long_signal & long_regime,
        short_signal & short_regime,
        cfg,
    )


def bollinger_fade(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    lower, _, upper = cache.bollinger(
        int(p["window"]), float(p["std_mult"])
    )
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = (cache.df["low"] <= lower) & (cache.close > lower) & long_regime
    short_signal = (cache.df["high"] >= upper) & (cache.close < upper) & short_regime
    return apply_direction(long_signal, short_signal, cfg)


def zscore_reversion(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    mean = cache.sma(int(p["window"]))
    std = cache.std(int(p["window"])).replace(0.0, float("nan"))
    zscore = (cache.close - mean) / std
    threshold = float(p["zscore_threshold"])
    long_signal = crossed_above(zscore, -threshold)
    short_signal = crossed_below(zscore, threshold)
    return apply_direction(long_signal, short_signal, cfg)


def rsi_extreme(cache: IndicatorCache, cfg: StrategyConfig):
    p = cfg.params
    value = cache.rsi(int(p["rsi_window"]))
    level = float(p["level"])
    long_regime, short_regime = regime_filter(cache, int(p["regime_ema"]))
    long_signal = crossed_above(value, level) & (value.shift(1) < level) & long_regime
    short_signal = crossed_below(value, 100.0 - level) & (
        value.shift(1) > 100.0 - level
    ) & short_regime
    return apply_direction(long_signal, short_signal, cfg)
