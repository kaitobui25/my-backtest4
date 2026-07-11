import pandas as pd

from btsearch.config import StrategyConfig
from btsearch.strategies.common import apply_time_filter


def test_asia_session_uses_utc_hours():
    index = pd.date_range(
        "2026-01-01", periods=24, freq="1h", tz="UTC"
    )
    signal = pd.Series(True, index=index)
    cfg = StrategyConfig(
        "x", "both", {"session": "asia"}
    )
    long_signal, short_signal = apply_time_filter(
        signal, signal, cfg
    )
    assert long_signal.sum() == 8
    assert short_signal.sum() == 8
    assert long_signal.loc[index[0]]
    assert not long_signal.loc[index[8]]
