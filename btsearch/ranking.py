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

    total_folds = int(folds["fold_id"].nunique())
    grouped = folds.groupby(
        ["config_id", "family", "direction", "params_json"],
        dropna=False,
    )
    rows: list[dict] = []
    for keys, group in grouped:
        validation = group["validation_expectancy_r"].astype(float)
        trades = group["validation_trades"].astype(int)
        valid = validation.notna() & trades.notna()
        exp_valid = validation[valid]
        trade_valid = trades[valid]

        macro = float(validation.mean()) if len(validation) else float("nan")
        std = float(validation.std(ddof=0))
        median_value = float(validation.median())
        min_value = float(validation.min())
        positive_ratio = float((validation > 0).mean())
        total_trades = int(trade_valid.sum())
        folds_tested = int(valid.sum())
        all_folds_tested = folds_tested == total_folds

        # micro_expectancy_r is trade-weighted across folds:
        #   sum(validation_expectancy_r * validation_trades)
        #   / sum(validation_trades)
        if total_trades > 0:
            micro = float(
                (exp_valid * trade_valid).sum() / total_trades
            )
        else:
            micro = float("nan")

        # Robust score:
        # macro expectancy
        # - 0.5 * fold dispersion
        # - 0.10R for every fraction of non-positive folds
        # - up to 0.10R small-sample penalty below 300 validation trades.
        negative_fold_penalty = 0.10 * (1.0 - positive_ratio)
        sample_penalty = 0.10 * max(0.0, (300 - total_trades) / 300)
        robust_score = (
            macro
            - 0.5 * std
            - negative_fold_penalty
            - sample_penalty
        )

        rows.append({
            "config_id": keys[0],
            "family": keys[1],
            "direction": keys[2],
            "params_json": keys[3],
            "macro_expectancy_r": macro,
            "micro_expectancy_r": micro,
            "median_expectancy_r": median_value,
            "worst_fold_expectancy_r": min_value,
            "validation_std_expectancy_r": std,
            "positive_fold_ratio": positive_ratio,
            "total_validation_trades": total_trades,
            "folds_tested": folds_tested,
            "total_folds": total_folds,
            "all_folds_tested": bool(all_folds_tested),
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
    eligible = aggregate.loc[
        (aggregate["all_folds_tested"])
        & (aggregate["total_validation_trades"] >= min_trades)
    ]
    return eligible.sort_values(
        [
            "micro_expectancy_r",
            "positive_fold_ratio",
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
    eligible = aggregate.loc[
        (aggregate["all_folds_tested"])
        & (aggregate["total_validation_trades"] >= min_trades)
    ]
    return eligible.sort_values(
        [
            "robust_score",
            "micro_expectancy_r",
            "total_validation_trades",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
