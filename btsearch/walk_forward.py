from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from btsearch.config import RunSettings, StrategyConfig
from btsearch.engine import VectorBTBatchEngine

_VALIDATION_COLUMNS = {
    "trades": "validation_trades",
    "expectancy_r": "validation_expectancy_r",
    "winrate": "validation_winrate",
    "avg_win_r": "validation_avg_win_r",
    "avg_loss_r": "validation_avg_loss_r",
    "profit_factor_r": "validation_profit_factor_r",
    "max_drawdown_r": "validation_max_drawdown_r",
    "cost_r_per_trade": "validation_cost_r_per_trade",
}


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def build_expanding_folds(
    index: pd.DatetimeIndex,
    min_train_months: int = 24,
    validation_months: int = 6,
    first_validation_start: pd.Timestamp | None = None,
) -> list[Fold]:
    start = pd.Timestamp(index.min()).tz_convert("UTC")
    end = pd.Timestamp(index.max()).tz_convert("UTC")

    earliest = start + pd.DateOffset(months=min_train_months)
    validation_start = (
        max(earliest, first_validation_start)
        if first_validation_start is not None
        else earliest
    )
    folds: list[Fold] = []
    number = 1
    while validation_start < end:
        validation_end = min(
            validation_start + pd.DateOffset(months=validation_months),
            end + pd.Timedelta(microseconds=1),
        )
        if validation_end <= validation_start:
            break
        folds.append(Fold(
            fold_id=f"F{number:02d}",
            train_start=start,
            train_end=validation_start,
            validation_start=validation_start,
            validation_end=validation_end,
        ))
        number += 1
        validation_start = validation_end
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    candidate_configs: list[StrategyConfig],
    settings: RunSettings,
    output_dir: Path,
    resume: bool,
    first_validation_start: pd.Timestamp,
) -> pd.DataFrame:
    """Validate a fixed shortlist of configs on every walk-forward fold.

    The shortlist (``candidate_configs``) is frozen by the caller after Phase 1
    and Phase 2. Every config in that shortlist is evaluated on the validation
    window of each fold using the same out-of-sample data; no per-fold
    re-selection is performed. The returned frame has one row per
    (config, fold) so downstream aggregation can confirm each config was
    tested on every fold.
    """
    engine = VectorBTBatchEngine(settings)
    all_rows: list[pd.DataFrame] = []

    for fold in build_expanding_folds(
        df.index,
        first_validation_start=first_validation_start,
    ):
        validation = df.loc[
            (df.index >= fold.validation_start)
            & (df.index < fold.validation_end)
        ]
        if len(validation) < 500:
            continue

        val_path = output_dir / "walk_forward_checkpoints" / (
            f"{fold.fold_id}_validation.parquet"
        )
        validation_results = engine.run(
            validation,
            candidate_configs,
            val_path,
            resume,
            f"{fold.fold_id}-VALIDATION",
        )

        renamed = validation_results.rename(
            columns=_VALIDATION_COLUMNS
        )
        renamed["fold_id"] = fold.fold_id
        renamed["validation_start"] = fold.validation_start
        renamed["validation_end"] = fold.validation_end
        all_rows.append(renamed)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
