import pandas as pd
import pytest
from btsearch.ranking import (
    aggregate_walk_forward,
    main_ranking,
    rank_walk_forward_expectancy,
    rank_walk_forward_robust,
)


def test_three_trade_config_cannot_top_main_ranking():
    data = pd.DataFrame([
        {
            "config_id": "noise",
            "expectancy_r": 2.0,
            "profit_factor_r": 10.0,
            "trades": 3,
        },
        {
            "config_id": "usable",
            "expectancy_r": 0.1,
            "profit_factor_r": 1.2,
            "trades": 200,
        },
    ])
    ranked = main_ranking(data, min_trades=100)
    assert ranked["config_id"].tolist() == ["usable"]


def _fold_row(config_id, family, fold_id, exp, trades):
    return {
        "config_id": config_id,
        "family": family,
        "direction": "long",
        "params_json": "{}",
        "validation_expectancy_r": exp,
        "validation_trades": trades,
        "validation_winrate": 0.5,
        "validation_avg_win_r": 0.3,
        "validation_avg_loss_r": -0.2,
        "validation_profit_factor_r": 1.5,
        "validation_max_drawdown_r": 1.0,
        "validation_cost_r_per_trade": 0.05,
        "fold_id": fold_id,
    }


def _build_folds(rows):
    return pd.DataFrame(rows)


def test_micro_expectancy_weighted_by_trades():
    folds = _build_folds([
        _fold_row("A", "trend", "F01", 0.1, 100),
        _fold_row("A", "trend", "F02", 0.3, 300),
        _fold_row("B", "momentum", "F01", 0.2, 50),
        _fold_row("B", "momentum", "F02", 0.4, 50),
    ])
    agg = aggregate_walk_forward(folds).set_index("config_id")

    # A: macro = mean(0.1, 0.3) = 0.2
    #    micro = (0.1*100 + 0.3*300) / 400 = 0.25
    assert agg.loc["A", "macro_expectancy_r"] == 0.2
    assert agg.loc["A", "micro_expectancy_r"] == 0.25
    # B: macro = 0.3, micro = (0.2*50 + 0.4*50) / 100 = 0.3
    assert agg.loc["B", "macro_expectancy_r"] == pytest.approx(0.3)
    assert agg.loc["B", "micro_expectancy_r"] == pytest.approx(0.3)


def test_macro_expectancy_unweighted_fold_average():
    folds = _build_folds([
        _fold_row("A", "trend", "F01", 0.0, 10),
        _fold_row("A", "trend", "F02", 0.2, 1000),
        _fold_row("A", "trend", "F03", 0.4, 10),
    ])
    agg = aggregate_walk_forward(folds).set_index("config_id")
    # Simple unweighted mean of the three fold expectancies.
    assert agg.loc["A", "macro_expectancy_r"] == (0.0 + 0.2 + 0.4) / 3


def test_all_folds_tested_flags_missing_fold():
    folds = _build_folds([
        _fold_row("A", "trend", "F01", 0.2, 200),
        _fold_row("A", "trend", "F02", 0.2, 200),
        _fold_row("B", "momentum", "F01", 0.5, 200),
    ])
    agg = aggregate_walk_forward(folds).set_index("config_id")
    assert bool(agg.loc["A", "all_folds_tested"]) is True
    assert agg.loc["A", "folds_tested"] == 2
    assert agg.loc["A", "total_folds"] == 2
    assert bool(agg.loc["B", "all_folds_tested"]) is False
    assert agg.loc["B", "folds_tested"] == 1


def test_missing_fold_config_excluded_from_rankings():
    folds = _build_folds([
        _fold_row("A", "trend", "F01", 0.2, 200),
        _fold_row("A", "trend", "F02", 0.2, 200),
        _fold_row("B", "momentum", "F01", 0.9, 200),
    ])
    agg = aggregate_walk_forward(folds)
    # B has a higher expectancy but only traded one of two folds.
    top_exp = rank_walk_forward_expectancy(agg, min_trades=100)
    assert "B" not in top_exp["config_id"].tolist()
    assert "A" in top_exp["config_id"].tolist()

    top_rob = rank_walk_forward_robust(agg, min_trades=100)
    assert "B" not in top_rob["config_id"].tolist()

