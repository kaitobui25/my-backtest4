from __future__ import annotations

import numpy as np
import pandas as pd


def main_ranking(
    results: pd.DataFrame,
    min_trades: int = 100,
) -> pd.DataFrame:
    eligible = results.loc[
        (results["trades"] >= min_trades)
        & results["expectancy_r"].notna()
    ].copy()
    return eligible.sort_values(
        ["expectancy_r", "profit_factor_r", "trades"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def top_by_family(
    results: pd.DataFrame,
    min_trades: int,
    per_family: int,
) -> pd.DataFrame:
    ranked = main_ranking(results, min_trades)
    if ranked.empty:
        return ranked
    return (
        ranked.groupby("family", group_keys=False)
        .head(per_family)
        .reset_index(drop=True)
    )


def aggregate_walk_forward(
    folds: pd.DataFrame,
) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()

    grouped = folds.groupby(
        ["config_id", "family", "direction", "params_json"],
        dropna=False,
    )
    rows: list[dict] = []
    for keys, group in grouped:
        validation = group["validation_expectancy_r"].astype(float)
        trades = group["validation_trades"].astype(int)
        positive_ratio = float((validation > 0).mean())
        mean = float(validation.mean())
        std = float(validation.std(ddof=0))
        min_value = float(validation.min())
        total_trades = int(trades.sum())

        # Robust score:
        # mean expectancy
        # - 0.5 * fold dispersion
        # - 0.10R for every fraction of non-positive folds
        # - up to 0.10R small-sample penalty below 300 validation trades.
        negative_fold_penalty = 0.10 * (1.0 - positive_ratio)
        sample_penalty = 0.10 * max(0.0, (300 - total_trades) / 300)
        robust_score = (
            mean
            - 0.5 * std
            - negative_fold_penalty
            - sample_penalty
        )

        rows.append({
            "config_id": keys[0],
            "family": keys[1],
            "direction": keys[2],
            "params_json": keys[3],
            "validation_mean_expectancy_r": mean,
            "validation_median_expectancy_r": float(validation.median()),
            "validation_min_expectancy_r": min_value,
            "validation_std_expectancy_r": std,
            "validation_positive_fold_ratio": positive_ratio,
            "total_validation_trades": total_trades,
            "folds_selected": int(len(group)),
            "validation_mean_winrate": float(
                group["validation_winrate"].mean()
            ),
            "validation_mean_avg_win_r": float(
                group["validation_avg_win_r"].mean()
            ),
            "validation_mean_avg_loss_r": float(
                group["validation_avg_loss_r"].mean()
            ),
            "validation_mean_profit_factor_r": float(
                group["validation_profit_factor_r"].mean()
            ),
            "validation_max_drawdown_r": float(
                group["validation_max_drawdown_r"].max()
            ),
            "cost_r_per_trade": float(
                group["validation_cost_r_per_trade"].mean()
            ),
            "robust_score": float(robust_score),
        })

    return pd.DataFrame(rows)


def rank_walk_forward_expectancy(
    aggregate: pd.DataFrame,
    min_trades: int = 100,
) -> pd.DataFrame:
    if aggregate.empty:
        return aggregate
    return aggregate.loc[
        aggregate["total_validation_trades"] >= min_trades
    ].sort_values(
        [
            "validation_mean_expectancy_r",
            "validation_positive_fold_ratio",
            "total_validation_trades",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def rank_walk_forward_robust(
    aggregate: pd.DataFrame,
    min_trades: int = 100,
) -> pd.DataFrame:
    if aggregate.empty:
        return aggregate
    return aggregate.loc[
        aggregate["total_validation_trades"] >= min_trades
    ].sort_values(
        [
            "robust_score",
            "validation_mean_expectancy_r",
            "total_validation_trades",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
