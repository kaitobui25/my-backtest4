import pandas as pd
from btsearch.ranking import main_ranking


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
