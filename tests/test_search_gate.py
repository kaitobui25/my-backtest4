from __future__ import annotations

import pandas as pd

import search_gate


def test_add_flags_rejects_high_expectancy_with_too_few_trades() -> None:
    frame = pd.DataFrame(
        {
            "trades": [3, 300],
            "expectancy_r": [0.80, 0.23],
        }
    )

    result = search_gate.add_flags(frame, target=0.225, min_trades=300)

    three_trade_row = result.loc[result["trades"] == 3].iloc[0]
    valid_row = result.loc[result["trades"] == 300].iloc[0]
    assert bool(three_trade_row["target_hit"])
    assert not bool(three_trade_row["sample_ok"])
    assert not bool(three_trade_row["eligible"])
    assert bool(valid_row["eligible"])


def test_dedupe_configs_removes_bollinger_ema_duplicates() -> None:
    configs = [
        search_gate.core.Config("A", "bollinger_reentry", 100, 20, 2.0, 14, 1.5, 2.0, "long"),
        search_gate.core.Config("B", "bollinger_reentry", 200, 20, 2.0, 14, 1.5, 2.0, "long"),
    ]

    result = search_gate.dedupe_configs(configs)

    assert len(result) == 1
    assert result[0].ema_window == 100
