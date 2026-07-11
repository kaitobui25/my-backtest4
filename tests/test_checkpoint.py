import pandas as pd

from btsearch.checkpoint import CheckpointStore
from btsearch.config import StrategyConfig


def test_checkpoint_pending_does_not_repeat_completed(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path / "results.parquet")
    configs = [
        StrategyConfig("x", "long", {"sl_atr": 1.0, "rr": 2.0}),
        StrategyConfig("x", "short", {"sl_atr": 1.0, "rr": 2.0}),
    ]

    completed = pd.DataFrame([{
        "config_id": configs[0].config_id,
        "expectancy_r": 0.1,
    }])
    completed.to_csv(store.csv_path, index=False)

    pending = store.pending(configs)
    assert [item.config_id for item in pending] == [configs[1].config_id]
