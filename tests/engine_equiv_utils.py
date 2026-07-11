from __future__ import annotations

import math

import numpy as np
import pandas as pd

from btsearch.config import RunSettings, StrategyConfig
from btsearch.engine import VectorBTBatchEngine
from btsearch.grids import build_coarse_configs

BASELINE_FIELDS = [
    "trades",
    "winrate",
    "avg_win_r",
    "avg_loss_r",
    "expectancy_r",
    "profit_factor_r",
    "total_r",
    "max_drawdown_r",
    "cost_r_per_trade",
]


def make_test_data(seed: int = 42, n_rows: int = 3000) -> pd.DataFrame:
    """Deterministic synthetic OHLCV with volume (no external data needed)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_rows, freq="15min", tz="UTC")
    ret = rng.normal(0.0, 0.001, n_rows)
    close = 100.0 * np.exp(np.cumsum(ret))
    noise = np.abs(rng.normal(0.0, 0.001, n_rows))
    high = close * (1.0 + noise)
    low = close * (1.0 - noise)
    open_ = close * (1.0 + rng.normal(0.0, 0.0005, n_rows))
    open_ = np.maximum(open_, low)
    volume = rng.integers(500, 1500, n_rows).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def make_test_configs(n_per_family: int = 4) -> list[StrategyConfig]:
    """~60 configs spanning every family and long/short/both directions."""
    all_cfg = build_coarse_configs(has_volume=True)
    by_family: dict[str, list[StrategyConfig]] = {}
    for c in all_cfg:
        by_family.setdefault(c.family, []).append(c)
    chosen: list[StrategyConfig] = []
    for cfgs in by_family.values():
        chosen.extend(cfgs[:n_per_family])
    return chosen


def run_and_collect(
    df: pd.DataFrame,
    configs: list[StrategyConfig],
    path,
    batch_size: int = 48,
    fee: float = 0.0005,
    slippage: float = 0.0002,
    group: bool = True,
) -> dict[str, dict[str, float]]:
    settings = RunSettings(batch_size=batch_size, fee=fee, slippage=slippage)
    engine = VectorBTBatchEngine(settings)
    try:
        result = engine.run(df, configs, path, False, "BENCH", group=group)
    except TypeError:
        # Older engine builds without the grouping flag; results are
        # order-independent so this is equivalent for baseline capture.
        result = engine.run(df, configs, path, False, "BENCH")
    return {
        row["config_id"]: {f: row[f] for f in BASELINE_FIELDS}
        for _, row in result.iterrows()
    }


def compare_results(
    baseline: dict[str, dict[str, float]],
    actual: dict[str, dict[str, float]],
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> None:
    assert set(baseline) == set(actual), (
        "config_id sets differ: "
        f"only_baseline={set(baseline) - set(actual)} "
        f"only_actual={set(actual) - set(baseline)}"
    )
    for cid in baseline:
        b = baseline[cid]
        a = actual[cid]
        assert b["trades"] == a["trades"], f"{cid}: trades differ"
        for f in BASELINE_FIELDS:
            if f == "trades":
                continue
            # Baseline JSON may serialize inf/nan as strings; coerce to float.
            bv = float(b[f]) if b[f] is not None else float("nan")
            av = float(a[f]) if a[f] is not None else float("nan")
            if pd.isna(bv) and pd.isna(av):
                continue
            if math.isinf(bv) or math.isinf(av):
                # infinities are only equal to themselves (same sign).
                assert bv == av, f"{cid}: {f} differs ({bv} vs {av})"
                continue
            diff = abs(bv - av)
            assert diff <= atol + rtol * abs(bv), (
                f"{cid}: {f} differs by {diff} ({bv} vs {av})"
            )
