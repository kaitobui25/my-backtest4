from __future__ import annotations

from collections.abc import Callable
from btsearch.config import StrategyConfig
from btsearch.indicators import IndicatorCache
from .trend import (
    donchian_breakout,
    rolling_range_breakout,
    ema_cross,
    ema_slope_pullback,
)
from .momentum import rsi_momentum, roc_momentum, macd_momentum
from .mean_reversion import (
    bollinger_reentry,
    bollinger_fade,
    zscore_reversion,
    rsi_extreme,
)
from .volatility import atr_expansion_breakout, squeeze_breakout
from .volume import breakout_volume, momentum_volume


SignalFunction = Callable[
    [IndicatorCache, StrategyConfig],
    tuple[object, object],
]

REGISTRY: dict[str, SignalFunction] = {
    "donchian_breakout": donchian_breakout,
    "rolling_range_breakout": rolling_range_breakout,
    "ema_cross": ema_cross,
    "ema_slope_pullback": ema_slope_pullback,
    "rsi_momentum": rsi_momentum,
    "roc_momentum": roc_momentum,
    "macd_momentum": macd_momentum,
    "bollinger_reentry": bollinger_reentry,
    "bollinger_fade": bollinger_fade,
    "zscore_reversion": zscore_reversion,
    "rsi_extreme": rsi_extreme,
    "atr_expansion_breakout": atr_expansion_breakout,
    "squeeze_breakout": squeeze_breakout,
    "breakout_volume": breakout_volume,
    "momentum_volume": momentum_volume,
}


def make_signals(cache: IndicatorCache, config: StrategyConfig):
    try:
        fn = REGISTRY[config.family]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy family: {config.family}") from exc
    return fn(cache, config)
