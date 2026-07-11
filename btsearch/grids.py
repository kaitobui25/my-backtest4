from __future__ import annotations

from itertools import product
from typing import Iterable, Any

from btsearch.config import StrategyConfig


DIRECTIONS = ("long", "short", "both")
SL_ATR = (1.25, 1.75, 2.25)
RR = (1.5, 2.0, 2.5)
ATR_WINDOWS = (14,)


def _configs(
    family: str,
    grid: dict[str, Iterable[Any]],
    directions: Iterable[str] = DIRECTIONS,
) -> list[StrategyConfig]:
    names = list(grid)
    values = [list(grid[name]) for name in names]
    out: list[StrategyConfig] = []
    for combo in product(*values):
        params = dict(zip(names, combo, strict=True))
        if family == "ema_cross" and params["fast_ema"] >= params["slow_ema"]:
            continue
        if family == "macd_momentum" and params["fast"] >= params["slow"]:
            continue
        for direction in directions:
            out.append(StrategyConfig(family, direction, params))
    return out


def build_coarse_configs(has_volume: bool) -> list[StrategyConfig]:
    configs: list[StrategyConfig] = []

    configs += _configs("donchian_breakout", {
        "lookback": (20, 48, 96),
        "regime_ema": (50, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("rolling_range_breakout", {
        "lookback": (12, 24, 48, 96),
        "breakout_atr_buffer": (0.0, 0.1, 0.25),
        "regime_ema": (50, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("ema_cross", {
        "fast_ema": (8, 13, 21, 34),
        "slow_ema": (34, 55, 89, 144),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("ema_slope_pullback", {
        "ema_window": (34, 50, 100, 200),
        "slope_lookback": (3, 6, 12),
        "pullback_atr": (0.25, 0.5, 0.75),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("rsi_momentum", {
        "rsi_window": (7, 14, 21),
        "threshold": (50.0, 55.0, 60.0),
        "regime_ema": (0, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("roc_momentum", {
        "roc_window": (3, 6, 12, 24),
        "threshold": (0.0025, 0.005, 0.01),
        "regime_ema": (0, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("macd_momentum", {
        "fast": (8, 12),
        "slow": (21, 26, 34),
        "signal": (5, 9),
        "regime_ema": (0, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("bollinger_reentry", {
        "window": (20, 40, 60),
        "std_mult": (1.5, 2.0, 2.5),
        "regime_ema": (0, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    }, ("long", "short"))
    configs += _configs("bollinger_fade", {
        "window": (20, 40, 60),
        "std_mult": (1.5, 2.0, 2.5),
        "regime_ema": (0, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    }, ("long", "short"))
    configs += _configs("zscore_reversion", {
        "window": (20, 40, 80),
        "zscore_threshold": (1.5, 2.0, 2.5),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    }, ("long", "short"))
    configs += _configs("rsi_extreme", {
        "rsi_window": (7, 14, 21),
        "level": (20.0, 25.0, 30.0),
        "regime_ema": (0, 100, 200),
        "atr_window": ATR_WINDOWS,
        "sl_atr": SL_ATR,
        "rr": RR,
    }, ("long", "short"))
    configs += _configs("atr_expansion_breakout", {
        "atr_window": (7, 14),
        "atr_mean_window": (48, 96, 192),
        "expansion_mult": (1.1, 1.25, 1.5),
        "lookback": (20, 48, 96),
        "sl_atr": SL_ATR,
        "rr": RR,
    })
    configs += _configs("squeeze_breakout", {
        "bb_window": (20, 40),
        "std_mult": (1.5, 2.0),
        "bandwidth_lookback": (96, 192),
        "quantile": (0.1, 0.2),
        "squeeze_memory": (4, 12),
        "breakout_lookback": (12, 24, 48),
        "atr_window": ATR_WINDOWS,
        "sl_atr": (1.25, 2.0),
        "rr": (1.5, 2.5),
    })
    if has_volume:
        configs += _configs("breakout_volume", {
            "lookback": (20, 48, 96),
            "volume_window": (20, 48, 96),
            "volume_mult": (1.0, 1.25, 1.5),
            "atr_window": ATR_WINDOWS,
            "sl_atr": SL_ATR,
            "rr": RR,
        })
        configs += _configs("momentum_volume", {
            "roc_window": (3, 6, 12),
            "threshold": (0.0025, 0.005),
            "volume_window": (20, 48, 96),
            "volume_mult": (1.0, 1.5),
            "atr_window": ATR_WINDOWS,
            "sl_atr": SL_ATR,
            "rr": RR,
        })

    return deduplicate(configs)


def build_full_configs(has_volume: bool) -> list[StrategyConfig]:
    base = build_coarse_configs(has_volume)
    # Full mode adds neighborhoods around every coarse configuration.
    return deduplicate(base + refine_configs(base))


def _neighbors(key: str, value: Any) -> list[Any]:
    if isinstance(value, bool) or value == 0:
        return [value]
    if key in {"rr", "sl_atr", "std_mult", "expansion_mult", "volume_mult"}:
        return sorted({round(max(0.25, value - 0.25), 4), value, round(value + 0.25, 4)})
    if key in {"breakout_atr_buffer", "pullback_atr"}:
        return sorted({round(max(0.0, value - 0.1), 4), value, round(value + 0.1, 4)})
    if key in {"threshold"}:
        if value < 0.1:
            return sorted({round(value * 0.75, 6), value, round(value * 1.25, 6)})
        return sorted({max(1.0, value - 2.5), value, value + 2.5})
    if key in {"level", "zscore_threshold"}:
        step = 2.5 if value > 5 else 0.25
        return sorted({max(step, value - step), value, value + step})
    if key == "quantile":
        return sorted({round(max(0.02, value - 0.05), 3), value, round(min(0.5, value + 0.05), 3)})
    if isinstance(value, int):
        return sorted({max(2, round(value * 0.8)), value, max(3, round(value * 1.2))})
    if isinstance(value, float):
        return sorted({round(value * 0.8, 6), value, round(value * 1.2, 6)})
    return [value]


def refine_configs(seeds: Iterable[StrategyConfig]) -> list[StrategyConfig]:
    out: list[StrategyConfig] = []
    for seed in seeds:
        keys = list(seed.params)
        # Change one parameter at a time around the seed. This avoids a
        # combinatorial explosion while still exploring each local axis.
        out.append(seed)
        for key in keys:
            for neighbor in _neighbors(key, seed.params[key]):
                params = dict(seed.params)
                params[key] = neighbor
                if seed.family == "ema_cross" and params["fast_ema"] >= params["slow_ema"]:
                    continue
                if seed.family == "macd_momentum" and params["fast"] >= params["slow"]:
                    continue
                out.append(StrategyConfig(seed.family, seed.direction, params))

        # Time filters are refined only around promising seeds so broad
        # search stays fast instead of multiplying the entire coarse grid.
        for session in (
            "asia", "london", "new_york", "overlap", "high_liquidity"
        ):
            params = dict(seed.params)
            params["session"] = session
            out.append(StrategyConfig(seed.family, seed.direction, params))
        for weekday_filter in ("weekdays", "mon_thu"):
            params = dict(seed.params)
            params["weekday_filter"] = weekday_filter
            out.append(StrategyConfig(seed.family, seed.direction, params))
    return deduplicate(out)


def deduplicate(configs: Iterable[StrategyConfig]) -> list[StrategyConfig]:
    unique: dict[str, StrategyConfig] = {}
    for config in configs:
        unique[config.config_id] = config
    return list(unique.values())
