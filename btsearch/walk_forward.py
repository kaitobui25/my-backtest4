from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from btsearch.config import RunSettings, StrategyConfig
from btsearch.engine import VectorBTBatchEngine
from btsearch.ranking import main_ranking


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


def _row_to_config(row: pd.Series) -> StrategyConfig:
    return StrategyConfig(
        family=str(row["family"]),
        direction=str(row["direction"]),
        params=json.loads(row["params_json"]),
    )


def run_walk_forward(
    df: pd.DataFrame,
    candidate_configs: list[StrategyConfig],
    settings: RunSettings,
    output_dir: Path,
    resume: bool,
    first_validation_start: pd.Timestamp,
) -> pd.DataFrame:
    engine = VectorBTBatchEngine(settings)
    all_rows: list[pd.DataFrame] = []

    for fold in build_expanding_folds(
        df.index,
        first_validation_start=first_validation_start,
    ):
        train = df.loc[
            (df.index >= fold.train_start)
            & (df.index < fold.train_end)
        ]
        validation = df.loc[
            (df.index >= fold.validation_start)
            & (df.index < fold.validation_end)
        ]
        if len(train) < 1_000 or len(validation) < 500:
            continue

        train_path = output_dir / "walk_forward_checkpoints" / (
            f"{fold.fold_id}_train.parquet"
        )
        train_results = engine.run(
            train,
            candidate_configs,
            train_path,
            resume,
            f"{fold.fold_id}-TRAIN",
        )
        ranked = main_ranking(
            train_results,
            min_trades=max(30, settings.min_trades_ranking // 2),
        )
        selected = ranked.head(
            settings.walk_forward_select_per_fold
        )
        if selected.empty:
            continue

        selected_configs = [
            _row_to_config(row)
            for _, row in selected.iterrows()
        ]
        val_path = output_dir / "walk_forward_checkpoints" / (
            f"{fold.fold_id}_validation.parquet"
        )
        validation_results = engine.run(
            validation,
            selected_configs,
            val_path,
            resume,
            f"{fold.fold_id}-VALIDATION",
        )

        merged = selected[[
            "config_id", "family", "direction", "params_json",
            "trades", "expectancy_r",
        ]].merge(
            validation_results[[
                "config_id", "trades", "expectancy_r",
                "winrate", "avg_win_r", "avg_loss_r",
                "profit_factor_r", "max_drawdown_r",
                "cost_r_per_trade",
            ]],
            on="config_id",
            suffixes=("_train", "_validation"),
        )
        merged = merged.rename(columns={
            "trades_train": "train_trades",
            "expectancy_r_train": "train_expectancy_r",
            "trades_validation": "validation_trades",
            "expectancy_r_validation": "validation_expectancy_r",
            "winrate": "validation_winrate",
            "avg_win_r": "validation_avg_win_r",
            "avg_loss_r": "validation_avg_loss_r",
            "profit_factor_r": "validation_profit_factor_r",
            "max_drawdown_r": "validation_max_drawdown_r",
            "cost_r_per_trade": "validation_cost_r_per_trade",
        })
        merged["fold_id"] = fold.fold_id
        merged["train_start"] = fold.train_start
        merged["train_end"] = fold.train_end
        merged["validation_start"] = fold.validation_start
        merged["validation_end"] = fold.validation_end
        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
